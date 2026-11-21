"""Aggregate-only seed/patient uncertainty and independent consistency audits."""
import itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import sha256
from .stage_g_run import clinical_inputs
from .stage_g_model import Encoder,decision_loss
from .clinical_revision_models import independent_audit
from .stage_e_statistics import weighted_metrics,percentile
from .metrics import calibration

FAMILIES=['standard_relu','shared_mask_relu','selected_coherence_relu']
THRESHOLD=1/21


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def seed_weights(c):
    rng=np.random.default_rng(c['bootstrap_seed']+17);s=len(c['training_seeds'])
    idx=rng.integers(0,s,(c['bootstrap_repeats'],s))
    return np.vstack([np.bincount(r,minlength=s) for r in idx])/s


def interval(a):
    lo,hi,_=percentile(a);return lo,hi


def point_and_draws(y,p,c):
    def calculate(w):
        result=weighted_metrics(y,p,{THRESHOLD:p>=THRESHOLD},w)
        suffix=f'{THRESHOLD:g}'
        for name in ['sensitivity','specificity','ppv','npv','flagged_fraction']:
            result[name]=result.pop(name+'_'+suffix)
        result['diagnostic_loss']=w@decision_loss(y,p)/w.sum(1)
        return result
    point={k:float(v[0]) for k,v in calculate(np.ones((1,len(y)))).items()}
    cal,_=calibration(y,p);point.update(cal);point['prediction_variance']=float(np.var(p))
    rng=np.random.default_rng(c['bootstrap_seed']);draws={}
    for offset in range(0,c['bootstrap_repeats'],96):
        size=min(96,c['bootstrap_repeats']-offset);idx=rng.integers(0,len(y),(size,len(y)))
        weights=np.vstack([np.bincount(i,minlength=len(y)) for i in idx])
        for name,v in calculate(weights).items():draws.setdefault(name,[]).append(v)
    return point,{k:np.concatenate(v) for k,v in draws.items()}


def summary_row(points,draws,sw,meta):
    points=np.asarray(points);patient=draws.mean(1);crossed=(draws*sw).sum(1)
    pl,ph=interval(patient);cl,ch=interval(crossed)
    return {**meta,'mean':float(points.mean()),'seed_sd':float(points.std(ddof=1)),
            'seed_min':float(points.min()),'seed_max':float(points.max()),
            'patient_low':pl,'patient_high':ph,'crossed_low':cl,'crossed_high':ch}


def load_predictions(private,c):
    X,y,folds,panels,names,features=clinical_inputs()
    result={f:np.full((len(c['training_seeds']),len(y),len(panels)),np.nan) for f in FAMILIES}
    choices=[]
    for si,seed in enumerate(c['training_seeds']):
        for outer in range(5):
            te=folds[f'outer_{outer}_test'];file=private/f'seed_{seed}/fold_{outer}.npz'
            with np.load(file) as z:
                np.testing.assert_array_equal(z['test_indices'],te);np.testing.assert_array_equal(z['y'],y[te])
                for family in FAMILIES:result[family][si,te]=z[family]
    if not all(np.isfinite(p).all() for p in result.values()):raise ValueError('Incomplete seed predictions')
    return X,y,folds,panels,names,features,result


def build_audits(private,out,c,X,folds,panels,names):
    """Assessment-only fit, using no disease outcomes. Runs before scoring labels."""
    for seed in c['training_seeds']:
        folder=private/f'seed_{seed}'
        for outer in range(5):
            target=folder/f'audit_{outer}.npz';meta=out/f'audit_seed_{seed}_fold_{outer}.json'
            if target.exists() and meta.exists():
                if sha256(target)!=json.loads(meta.read_text())['sha256']:raise ValueError('Audit modified')
                continue
            tr,te=folds[f'outer_{outer}_train'],folds[f'outer_{outer}_test'];enc=Encoder().fit(X[tr]);data={};rows=[]
            with np.load(folder/f'fold_{outer}.npz') as z:
                for fi,family in enumerate(FAMILIES):
                    p=z[family]
                    for edge,(parent,child) in enumerate(c['edges']):
                        a=independent_audit(X[te],X[tr],enc,panels[parent],p[:,parent],p[:,child],c)
                        prefix=f'f{fi}_e{edge}'
                        data[prefix+'_projection']=a['projection_squared_error'].astype('float32')
                        data[prefix+'_child_projection']=(p[:,child]-a['projection'])**2
                        data[prefix+'_nested_difference']=a['raw_nested_difference'].astype('float32')
                        data[prefix+'_moment_products']=a['moment_products'].astype('float32')
                        rows.append({'family':family,'parent':names[parent],'child':names[child],'n':len(te),'instrument_names':a['moment_names'],'projection_mse':float(np.mean(a['projection_squared_error'])),'child_projection_mse':float(np.mean((p[:,child]-a['projection'])**2)),'moment_score':float(np.mean(a['moments']**2))})
            np.savez_compressed(target,test_indices=te,**data)
            dump(meta,{'seed':seed,'outer_fold':outer,'sha256':sha256(target),'audit_uses_outcomes':False,'audit_feedback_to_prediction_or_selection':False,'rows':rows})
            print(json.dumps({'independent_audit_complete_seed':seed,'outer_fold':outer}),flush=True)


def audit_statistics(private,out,c,y,folds,names,sw):
    cache={};rows=[];moments=[];summaries=[];paired=[]
    for si,seed in enumerate(c['training_seeds']):
        files=[];metadata=[]
        for outer in range(5):
            with np.load(private/f'seed_{seed}/audit_{outer}.npz') as z:files.append({k:z[k].copy() for k in z.files})
            metadata.append(json.loads((out/f'audit_seed_{seed}_fold_{outer}.json').read_text()))
        for fi,family in enumerate(FAMILIES):
            edge_draws=[];edge_points=[]
            for edge,(parent,child) in enumerate(c['edges']):
                prefix=f'f{fi}_e{edge}';vectors=np.zeros((len(y),3));products=[]
                for outer,z in enumerate(files):
                    te=z['test_indices'];vectors[te]=np.column_stack([z[prefix+'_projection'],z[prefix+'_child_projection'],z[prefix+'_nested_difference']]);products.append(z[prefix+'_moment_products'])
                    info=next(r for r in metadata[outer]['rows'] if r['family']==family and r['parent']==names[parent] and r['child']==names[child])
                    for j,instrument in enumerate(info['instrument_names']):
                        values=products[-1][:,j]
                        moments.append({'seed':seed,'outer_fold':outer,'family':family,'parent':names[parent],'child':names[child],'instrument':instrument,'n':len(te),'mean_moment':float(values.mean()),'descriptive_se':float(values.std(ddof=1)/np.sqrt(len(te)))})
                metric_names=['projection_mse','child_projection_mse','mean_nested_difference','moment_score']
                point=dict(zip(metric_names[:3],vectors.mean(0)));point['moment_score']=float(np.mean([np.mean(a.mean(0)**2) for a in products]))
                draws={k:[] for k in metric_names};rng=np.random.default_rng(c['bootstrap_seed'])
                for offset in range(0,c['bootstrap_repeats'],96):
                    size=min(96,c['bootstrap_repeats']-offset);idx=rng.integers(0,len(y),(size,len(y)));w=np.vstack([np.bincount(i,minlength=len(y)) for i in idx])
                    values=w@vectors/len(y)
                    for j,k in enumerate(metric_names[:3]):draws[k].append(values[:,j])
                    score=np.zeros(size)
                    for outer,a in enumerate(products):
                        te=files[outer]['test_indices'];ww=w[:,te];mu=ww@a/ww.sum(1)[:,None];score+=(mu*mu).mean(1)/5
                    draws['moment_score'].append(score)
                draws={k:np.concatenate(v) for k,v in draws.items()}
                label=names[parent]+' -> '+names[child]
                rows.append({'seed':seed,'family':family,'edge':label,'n':len(y),**{k:float(v) for k,v in point.items()}})
                cache[(family,label,seed)]=(point,draws);edge_draws.append(draws);edge_points.append(point)
            # Across-edge summary preserves within-fold conditional moment scoring.
            point={k:np.mean([p[k] for p in edge_points]) for k in edge_points[0]};draws={k:np.mean([d[k] for d in edge_draws],axis=0) for k in edge_draws[0]}
            rows.append({'seed':seed,'family':family,'edge':'all_five_edges','n':len(y),**{k:float(v) for k,v in point.items()}});cache[(family,'all_five_edges',seed)]=(point,draws)
        print(json.dumps({'audit_statistics_complete_seed':seed}),flush=True)
    edges=list(dict.fromkeys(r['edge'] for r in rows))
    for edge in edges:
        for family in FAMILIES:
            for metric in ['projection_mse','child_projection_mse','mean_nested_difference','moment_score']:
                ps=np.array([cache[family,edge,s][0][metric] for s in c['training_seeds']]);ds=np.column_stack([cache[family,edge,s][1][metric] for s in c['training_seeds']])
                summaries.append(summary_row(ps,ds,sw,{'family':family,'edge':edge,'metric':metric,'n':len(y),'training_seeds':5}))
        for a,b in itertools.combinations(FAMILIES,2):
            for metric in ['projection_mse','child_projection_mse','mean_nested_difference','moment_score']:
                ps=np.array([cache[b,edge,s][0][metric]-cache[a,edge,s][0][metric] for s in c['training_seeds']]);ds=np.column_stack([cache[b,edge,s][1][metric]-cache[a,edge,s][1][metric] for s in c['training_seeds']])
                paired.append(summary_row(ps,ds,sw,{'comparison':b+' minus '+a,'edge':edge,'metric':metric,'n':len(y),'training_seeds':5}))
    pd.DataFrame(rows).to_csv(out/'independent_consistency_by_seed.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'independent_consistency_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'independent_consistency_paired.csv',index=False)
    pd.DataFrame(moments).to_csv(out/'parent_moments_aggregate.csv',index=False)


def calibration_bins(y,p):
    groups=np.minimum((p*10).astype(int),9);pending=[];bins=[]
    for j in range(10):
        idx=np.flatnonzero(groups==j);pending.extend(idx.tolist())
        if len(pending)>=20 and y[pending].sum()>=5 and (1-y[pending]).sum()>=5:bins.append(pending);pending=[]
    if pending:
        if not bins:raise ValueError('Insufficient calibration support')
        bins[-1].extend(pending)
    return [{'bin':j,'n':len(idx),'positive':int(y[idx].sum()),'mean_predicted':float(p[idx].mean()),'observed':float(y[idx].mean())} for j,idx in enumerate(bins)]


def analyze(private,out,c):
    private,out=Path(private),Path(out)
    hashes={str(Path(__file__)):sha256(__file__)};receipt=out/'analysis_source_hashes.json'
    if receipt.exists() and json.loads(receipt.read_text())!=hashes:raise ValueError('Analysis source changed; use versioned analysis outputs')
    dump(receipt,hashes)
    X,y,folds,panels,names,features,P=load_predictions(private,c)
    build_audits(private,out,c,X,folds,panels,names)
    with np.load('private-data/cardiac_stage_e_v1/subset_masks.npz') as z:conservative=z['conservative'].copy()
    groups={'full':np.ones(len(y),bool),'conservative':conservative,'ntprobnp_available':np.isfinite(X[:,16]),'ntprobnp_unavailable':~np.isfinite(X[:,16])}
    rows=[];summaries=[];pairs=[];seedpairs=[];ensembles=[];bins=[];cache={};sw=seed_weights(c)
    for group,keep in groups.items():
        label=y[keep]
        for j,panel in enumerate(names):
            for family in FAMILIES:
                points=[];draws=[]
                for si,seed in enumerate(c['training_seeds']):
                    point,draw=point_and_draws(label,P[family][si,keep,j],c)
                    points.append(point);draws.append(draw)
                    rows.append({'group':group,'panel':panel,'family':family,'seed':seed,'n':len(label),'positive':int(label.sum()),**point})
                for metric in draws[0]:
                    ps=np.array([p[metric] for p in points]);ds=np.column_stack([d[metric] for d in draws])
                    summaries.append(summary_row(ps,ds,sw,{'group':group,'panel':panel,'family':family,'metric':metric,'n':len(label),'positive':int(label.sum()),'training_seeds':5}))
                for metric in points[0]:
                    if metric not in draws[0]:
                        values=np.array([pt[metric] for pt in points]);summaries.append({'group':group,'panel':panel,'family':family,'metric':metric,'n':len(label),'positive':int(label.sum()),'training_seeds':5,'mean':float(values.mean()),'seed_sd':float(values.std(ddof=1)),'seed_min':float(values.min()),'seed_max':float(values.max()),'patient_low':np.nan,'patient_high':np.nan,'crossed_low':np.nan,'crossed_high':np.nan})
                cache[(group,panel,family)]=(points,draws)
                if group=='full':
                    prob=P[family][:,:,j].mean(0);point,draw=point_and_draws(y,prob,c)
                    record={'group':group,'panel':panel,'family':family,'estimand':'uniform_5_seed_probability_ensemble','n':len(y),'positive':int(y.sum()),**point}
                    for metric,values in draw.items():lo,hi=interval(values);record.update({metric+'_patient_low':lo,metric+'_patient_high':hi})
                    ensembles.append(record)
                    if panel=='plus_both':bins.extend({'family':family,'panel':panel,'estimand':'uniform_5_seed_probability_ensemble',**b} for b in calibration_bins(y,prob))
            for a,b in itertools.combinations(FAMILIES,2):
                ap,ad=cache[group,panel,a];bp,bd=cache[group,panel,b]
                for metric in ad[0]:
                    ps=np.array([bb[metric]-aa[metric] for aa,bb in zip(ap,bp)]);ds=np.column_stack([bb[metric]-aa[metric] for aa,bb in zip(ad,bd)])
                    pairs.append(summary_row(ps,ds,sw,{'group':group,'panel':panel,'comparison':b+' minus '+a,'metric':metric,'n':len(label),'training_seeds':5}))
                    if group=='full' and a=='shared_mask_relu' and b=='selected_coherence_relu':
                        for si,seed in enumerate(c['training_seeds']):
                            lo,hi=interval(ds[:,si]);seedpairs.append({'group':group,'panel':panel,'seed':seed,'comparison':b+' minus '+a,'metric':metric,'difference':float(ps[si]),'patient_low':lo,'patient_high':hi,'n':len(label)})
        print(json.dumps({'clinical_metrics_complete_group':group,'n':len(label)}),flush=True)
    # Incremental-panel comparisons use the SAME paired patient/seed draws.
    panelpairs=[]
    for group,keep in groups.items():
        for a,b in [('ecg','ecg_clinical'),('ecg','ecg_routine'),('ecg_clinical','ecg_clinical_routine'),('ecg_clinical_routine','plus_ntprobnp'),('ecg_clinical_routine','plus_troponin_t'),('plus_ntprobnp','plus_both')]:
            for family in FAMILIES:
                ap,ad=cache[group,a,family];bp,bd=cache[group,b,family]
                for metric in ad[0]:
                    ps=np.array([bb[metric]-aa[metric] for aa,bb in zip(ap,bp)]);ds=np.column_stack([bb[metric]-aa[metric] for aa,bb in zip(ad,bd)])
                    panelpairs.append(summary_row(ps,ds,sw,{'group':group,'family':family,'panel_a':a,'panel_b':b,'metric':metric,'n':int(keep.sum()),'training_seeds':5}))
    pd.DataFrame(rows).to_csv(out/'metrics_by_seed.csv',index=False);pd.DataFrame(summaries).to_csv(out/'metrics_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_differences.csv',index=False);pd.DataFrame(seedpairs).to_csv(out/'paired_by_seed.csv',index=False)
    pd.DataFrame(panelpairs).to_csv(out/'incremental_panel_comparisons.csv',index=False)
    pd.DataFrame(ensembles).to_csv(out/'ensemble_metrics.csv',index=False);pd.DataFrame(bins).to_csv(out/'ensemble_calibration_bins.csv',index=False)
    choices=[]
    for seed in c['training_seeds']:
        for outer in range(5):
            m=json.loads((out/f'seed_{seed}_fold_{outer}.json').read_text());row={'seed':seed,'outer_fold':outer,'selected_beta':m['selected_beta'],'n_train':m['n_train'],'n_test':m['n_test']}
            for beta,loss in zip(c['coherence_candidates'],m['inner_brier']):row['inner_brier_beta_'+str(beta)]=float(np.mean(loss))
            choices.append(row)
    pd.DataFrame(choices).to_csv(out/'penalty_selection.csv',index=False)
    availability=[]
    for j,name in enumerate(features):
        observed=np.isfinite(X[:,j]);availability.append({'feature':name,'n':len(y),'available':int(observed.sum()),'missing':int((~observed).sum()),'available_positive':int(y[observed].sum()),'missing_percent':float(100*(~observed).mean())})
    pd.DataFrame(availability).to_csv(out/'predictor_availability.csv',index=False)
    dump(out/'cohort.json',{'n':len(y),'positive':int(y.sum()),'negative':int((1-y).sum()),'conservative_n':int(conservative.sum()),'conservative_positive':int(y[conservative].sum()),'training_seeds':c['training_seeds'],'patient_seeds_not_independent_cases':True})
    audit_statistics(private,out,c,y,folds,names,sw)
    dump(out/'uncertainty_notes.json',{'primary':'mean of five single-seed metrics','patient_intervals':'1000 paired patient resamples with fixed five trained runs','crossed_intervals':'same patient draw and resampled complete training-seed runs; five observed seeds only','ensemble':'separate secondary uniform probability ensemble; patient intervals only','not_captured':'full model/conditional/audit refitting, cohort/timing uncertainty or causal testing effects','audit':'different estimator/held-out cases and sampler-free moments; not known clinical ground truth; audit not used for tuning','calibration':'slope/intercept/ECE descriptive; seed variability table contains individual point estimates','multiplicity':'exploratory; no adjustment'})
