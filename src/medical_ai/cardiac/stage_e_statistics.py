"""Efficient paired patient bootstrap of fixed, nested held-out predictions.

Weighted ranking handles tied predictions exactly. Bootstrap weights are identical
for comparisons on identical patient rows. No models or thresholds are refitted.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from .metrics import calibration


def weighted_metrics(y, p, decisions, weights):
    y,p,w=np.asarray(y,int),np.asarray(p,float),np.asarray(weights,float)
    n=w.sum(axis=1);positive=w@y;negative=n-positive
    order=np.argsort(p,kind='stable');ys=y[order];ps=p[order];ws=w[:,order]
    starts=np.r_[0, np.flatnonzero(np.diff(ps))+1]
    gp=np.add.reduceat(ws*ys,starts,axis=1)
    gn=np.add.reduceat(ws*(1-ys),starts,axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        auc=(gp*(np.cumsum(gn,axis=1)-.5*gn)).sum(axis=1)/(positive*negative)
        rp,rn=gp[:,::-1],gn[:,::-1]
        cp=np.cumsum(rp,axis=1);ct=np.cumsum(rp+rn,axis=1)
        precision=np.divide(cp,ct,out=np.zeros_like(cp),where=ct>0)
        ap=(rp*precision).sum(axis=1)/positive
        clipped=np.clip(p,1e-15,1-1e-15)
        result={'auc':auc,'average_precision':ap,'brier':w@((p-y)**2)/n,
                'log_loss':w@(-y*np.log(clipped)-(1-y)*np.log(1-clipped))/n}
        for target,d in decisions.items():
            tp=w@(d & (y==1));fp=w@(d & (y==0));tn=negative-fp;fn=positive-tp
            suffix=f'{target:g}'
            result.update({f'sensitivity_{suffix}':tp/positive,f'specificity_{suffix}':tn/negative,
                           f'ppv_{suffix}':tp/(tp+fp),f'npv_{suffix}':tn/(tn+fn),
                           f'flagged_fraction_{suffix}':(tp+fp)/n})
    for threshold in [.05,.1,.15,.2,.3]:
        positive_actions=(p>=threshold)&(y==1)
        negative_actions=(p>=threshold)&(y==0)
        result[f'net_benefit_{threshold:g}']=(w@positive_actions-w@negative_actions*threshold/(1-threshold))/n
    return result


def bootstrap_draws(y,p,decisions,repeats=2000,seed=20261112,batch=96):
    rng=np.random.default_rng(seed);chunks={}
    for offset in range(0,repeats,batch):
        size=min(batch,repeats-offset)
        indices=rng.integers(0,len(y),size=(size,len(y)))
        w=np.vstack([np.bincount(row,minlength=len(y)) for row in indices])
        r=weighted_metrics(y,p,decisions,w)
        for k,v in r.items():chunks.setdefault(k,[]).append(v)
    return {k:np.concatenate(v) for k,v in chunks.items()}


def percentile(values):
    values=np.asarray(values);valid=values[np.isfinite(values)]
    return (float(np.quantile(valid,.025)),float(np.quantile(valid,.975)),len(valid)) if len(valid) else (np.nan,np.nan,0)


def analyze(private,output):
    private,out=Path(private),Path(output)
    config=json.loads((out/'config.json').read_text());repeats=config['bootstrap_repeats']
    with np.load('private-data/cardiac_stages_ab_v2/predictors.npz') as z:
        X,y,names=z['X'].copy(),z['y'].copy(),z['names'].tolist()
    with np.load(private/'subset_masks.npz') as z:
        conservative=z['conservative'].copy();no_history=z['no_known_history_conflict'].copy()
    nt=np.isfinite(X[:,names.index('ntprobnp')])
    rows=[];bins=[];curves=[];choices=[];cache={}
    drawdir=private/'bootstrap_draws';drawdir.mkdir(exist_ok=True)
    clinical_nt_panels={'ecg_age_sex_routine','ecg_age_sex_routine_ntprobnp',
                        'ecg_age_sex_routine_ntprobnp_availability_only'}
    for checkpoint in sorted((private/'models').glob('*.npz')):
        record=json.loads(checkpoint.with_suffix('.json').read_text())
        training,panel,family=record['dataset'],record['panel'],record['family']
        with np.load(checkpoint,allow_pickle=False) as z:
            included=z['included'].copy();p=z['p'].copy()
            decisions={t:z[f'decision_{t:g}'].copy() for t in config['sensitivity_targets']}
            np.testing.assert_array_equal(z['y'],y)
        evaluations={'full':included} if training=='full' else {'conservative':included}
        if training=='full':evaluations.update({'conservative':conservative,'no_known_history_conflict':no_history})
        if panel in clinical_nt_panels:
            evaluations.update({'ntprobnp_available':included & nt,'ntprobnp_missing':included & ~nt})
        for group,mask in evaluations.items():
            if not mask.any():continue
            yy,pp=y[mask],p[mask];dd={t:d[mask] for t,d in decisions.items()}
            if not np.isfinite(pp).all():raise ValueError('Missing held-out predictions')
            key=f'{training}__{panel}__{family}__{group}'
            drawfile=drawdir/f'{key}.npz'
            point={k:float(v[0]) for k,v in weighted_metrics(yy,pp,dd,np.ones((1,len(yy)))).items()}
            if drawfile.exists():
                with np.load(drawfile) as z:draws={k:z[k].copy() for k in z.files}
            else:
                draws=bootstrap_draws(yy,pp,dd,repeats,config['seed']);np.savez_compressed(drawfile,**draws)
            stats={'training_cohort':training,'evaluation_cohort':group,'panel':panel,'family':family,
                   'n':len(yy),'positive':int(yy.sum()),'prevalence':float(yy.mean()),**point}
            for metric,v in draws.items():
                lo,hi,count=percentile(v);stats.update({metric+'_low':lo,metric+'_high':hi,metric+'_valid_bootstrap':count})
            if len(np.unique(yy))==2:
                cal,b=calibration(yy,pp);stats.update(cal)
                for r in b:
                    # Descriptive Wilson bin interval, not model-refitting uncertainty.
                    n=r['n'];q=r['observed'];den=1+1.96**2/n
                    center=(q+1.96**2/(2*n))/den
                    radius=1.96*np.sqrt(q*(1-q)/n+1.96**2/(4*n*n))/den
                    bins.append({'training_cohort':training,'evaluation_cohort':group,'panel':panel,'family':family,
                                 **r,'observed_low':center-radius,'observed_high':center+radius})
            else:stats.update({'calibration_intercept':np.nan,'calibration_slope':np.nan,'ece':np.nan})
            rows.append(stats)
            cache[(training,panel,family,group)]={'point':point,'draws':draws,'n':len(yy),
                  'row_hash':hashlib.sha256(np.flatnonzero(mask).tobytes()).hexdigest()}
            if group in ('full','conservative'):
                fpr,tpr,_=roc_curve(yy,pp);precision,recall,_=precision_recall_curve(yy,pp)
                curves.extend({'training_cohort':training,'evaluation_cohort':group,'panel':panel,'family':family,
                               'curve':'roc','x':float(a),'y':float(b)} for a,b in zip(fpr,tpr))
                curves.extend({'training_cohort':training,'evaluation_cohort':group,'panel':panel,'family':family,
                               'curve':'pr','x':float(a),'y':float(b)} for a,b in zip(recall,precision))
        for r in record['fold_choices']:
            choices.append({'training_cohort':training,'panel':panel,'family':family,**r})
        print(f'Summarized {checkpoint.stem}',flush=True)
    contrasts=json.loads(Path('config/cardiac/comparison_contrasts.v2.json').read_text())
    pairs=[(tag,a,b) for tag,pairs in contrasts.items() if isinstance(pairs,list) for a,b in pairs]
    pairs += [('ordering_control','ecg_age_sex_routine','ecg_age_sex_routine_ntprobnp_availability_only'),
              ('raw_legacy_context','raw_ecg_only','raw_ecg_routine')]
    paired=[]
    def add_pair(a,b,metadata):
        if a['row_hash']!=b['row_hash']:raise ValueError('Paired comparison uses different patients')
        for metric in a['point']:
            lo,hi,count=percentile(b['draws'][metric]-a['draws'][metric])
            paired.append({**metadata,'n':a['n'],'metric':metric,
                           'difference_b_minus_a':b['point'][metric]-a['point'][metric],
                           'low':lo,'high':hi,'valid_bootstrap':count})
    for training in config['cohort_training']:
        for family in config['families']:
            for group in ['full','conservative','no_known_history_conflict','ntprobnp_available','ntprobnp_missing']:
                for tag,aa,bb in pairs:
                    ka,kb=(training,aa,family,group),(training,bb,family,group)
                    if ka in cache and kb in cache:
                        add_pair(cache[ka],cache[kb],{'training_cohort':training,'evaluation_cohort':group,
                                 'family':family,'panel_a':aa,'panel_b':bb,'contrast':tag})
    retraining=[]
    for (training,panel,family,group),a in cache.items():
        if training!='full' or group!='conservative':continue
        key=('conservative',panel,family,'conservative')
        if key in cache:
            b=cache[key]
            for metric in a['point']:
                lo,hi,count=percentile(b['draws'][metric]-a['draws'][metric])
                retraining.append({'evaluation_cohort':'conservative','family':family,'panel':panel,'n':a['n'],
                                   'metric':metric,'strict_fit_minus_full_fit':b['point'][metric]-a['point'][metric],
                                   'low':lo,'high':hi,'valid_bootstrap':count})
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    pd.DataFrame(retraining).to_csv(out/'conservative_retraining_differences.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'roc_pr_curves.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'fold_choices.csv',index=False)
    metrics.loc[metrics.panel.isin(clinical_nt_panels)].to_csv(out/'ntprobnp_ordering_analysis.csv',index=False)
    legacy=[]
    for panel,auc,ap,brier in [('raw_ecg_only',.7914,.3302,.0929),('raw_ecg_routine',.7967,.3487,.0918)]:
        legacy.append({'source':'historical_aggregate_not_independently_reproduced','cohort':'legacy',
                       'panel':panel,'n':2017,'positive':244,'auc':auc,'average_precision':ap,'brier':brier,
                       'comparability':'different eligibility, unknown identity overlap; no valid paired old/new CI'})
    for _,r in metrics.loc[(metrics.family=='hist_gradient_boosting') &
                          metrics.panel.isin(['ecg_only','ecg_routine','raw_ecg_only','raw_ecg_routine']) &
                          metrics.evaluation_cohort.isin(['full','conservative'])].iterrows():
        legacy.append({'source':'stage_e_nested_oof','cohort':r.evaluation_cohort,
                       'training_cohort':r.training_cohort,'panel':r.panel,'n':r.n,'positive':r.positive,
                       **{k:r[k] for k in ['auc','auc_low','auc_high','average_precision','brier']},
                       'comparability':'descriptive only; raw source representation sensitivity, no legacy membership equivalence'})
    pd.DataFrame(legacy).to_csv(out/'legacy_comparison.csv',index=False)
    (out/'statistics_notes.json').write_text(json.dumps({
        'bootstrap_repeats':repeats,'method':'ordinary patient bootstrap; paired weights for identical patient rows; percentile 95% intervals',
        'auc_ties':'weighted midrank U-statistic','average_precision':'weighted threshold-group precision recall increments',
        'uncertainty':'conditional on trained nested OOF models and fixed training-selected thresholds; not full pipeline refitting',
        'calibration_intervals':'Wilson intervals for descriptive calibration bins only; slope/intercept are descriptive point estimates',
        'models_covered':len(list((private/'models').glob('*.npz'))),'aggregate_rows':len(rows),
        'multiple_comparisons':'primary contrast prespecified; secondary comparisons exploratory, no multiplicity adjustment'},indent=2))
