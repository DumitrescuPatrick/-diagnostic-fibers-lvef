"""Stage G local reproducible experiments; public outputs contain aggregates only."""
import argparse
import itertools
import json
import os
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss
from .data import sha256
from .staged_audit import snapshot
from .stage_g_model import (Encoder,fit,predict,masks_from_bits,exact_constraints,kernel_constraints,decision_loss)
from .stage_g_synthetic import MedicalSpace
from .stage_e_statistics import bootstrap_draws,weighted_metrics,percentile
from .metrics import calibration

PRIV=Path('private-data/cardiac_stage_g_v1');OUT=Path('results/cardiac_stage_g_v1')
CONFIG=Path('config/cardiac/stage_g.v1.json')
FAMILIES=['standard_relu','shared_mask_relu','coherent_relu']


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def start():
    if (OUT/'config.json').exists():raise ValueError('Run already initialized; preserve outputs, use a separately versioned run for changes')
    PRIV.mkdir(exist_ok=True);os.chmod(PRIV,0o700);OUT.mkdir(exist_ok=True)
    protected=[p for p in Path('private-data').iterdir() if p!=PRIV]+[p for p in Path('results').iterdir() if p!=OUT]
    protected += [Path('docs/cardiac/STAGE_F_INTERFACE.md'),Path('src/medical_ai/cardiac/stage_f_interface.py')]
    dump(PRIV/'preservation_before.json',snapshot(protected))
    cfg=json.loads(CONFIG.read_text());dump(OUT/'config.json',cfg)
    import importlib.metadata
    dump(OUT/'versions.json',{'python':platform.python_version(),'platform':platform.platform(),**{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','matplotlib']}})
    dump(OUT/'source_hashes.json',{str(p):sha256(p) for p in sorted(Path('src/medical_ai/cardiac').glob('stage_g*.py'))})


def cfg():
    c=json.loads((OUT/'config.json').read_text())
    if c!=json.loads(CONFIG.read_text()):raise ValueError('Config changed')
    for p,h in json.loads((OUT/'source_hashes.json').read_text()).items():
        if sha256(p)!=h:raise ValueError('Source changed since initialization')
    return c


def train(X,y,panels,base,seed,c,beta=0,constraints=None,fixed=None):
    return fit(X,y,panels,base,seed,c['epochs'],beta,constraints,fixed,c['width'])


def save_model(path,fitted):
    model,e=fitted
    torch.save({'state_dict':model.state_dict(),'center':torch.tensor(e.center),'scale':torch.tensor(e.scale),'width':64},path)


def synthetic():
    c=cfg();rows=[];tower=[];selection=[]
    for wi,world in enumerate(c['worlds']):
        s=MedicalSpace(world);seed=c['seed']+wi*100
        X,y,_=s.sample(c['synthetic_train'],seed);V,vy,_=s.sample(c['synthetic_validation'],seed+1);T,ty,idx=s.sample(c['synthetic_test'],seed+2)
        panels=masks_from_bits(np.arange(16),6,2);e=Encoder().fit(X)
        constraints=exact_constraints(s,e,2,seed+3,c['constraint_rows'],c['conditional_particles'])
        trained={};pp={};population={}
        standard=[train(X,y,panels,2,seed,c,fixed=m) for m in panels]
        shared=train(X,y,panels,2,seed,c)
        candidates=[train(X,y,panels,2,seed,c,beta,constraints) for beta in c['coherence_candidates']]
        losses=[float(np.mean([brier_score_loss(vy,predict(f,V,m)) for m in panels])) for f in candidates]
        winner=int(np.argmin(losses));selection.append({'world':world,'candidate_betas':c['coherence_candidates'],'validation_brier':losses,'selected_beta':c['coherence_candidates'][winner]})
        for family,fitted in zip(FAMILIES,[standard,shared,candidates[winner]]):
            pp[family]=np.column_stack([predict(fitted[j] if family=='standard_relu' else fitted,T,m) for j,m in enumerate(panels)])
            population[family]=np.column_stack([predict(fitted[j] if family=='standard_relu' else fitted,s.X,m) for j,m in enumerate(panels)])
            for j,m in enumerate(panels):
                p=pp[family][:,j];truth=s.truth(m)[idx]
                rows.append({'world':world,'family':family,'panel':j,'n':len(ty),'conditional_probability_mse':float(np.mean((p-truth)**2)),'brier':float(np.mean((p-ty)**2)),'diagnostic_loss':float(decision_loss(ty,p).mean()),'exact_population_conditional_mse':float(np.sum(s.mass*(population[family][:,j]-s.truth(m))**2))})
            for j,m in enumerate(panels):
                for feature in np.flatnonzero(m==0):
                    child=m.copy();child[feature]=1;ci=int(np.flatnonzero(np.all(panels==child,axis=1))[0])
                    residual=population[family][:,j]-s.project(population[family][:,ci],m)
                    tower.append({'world':world,'family':family,'parent':j,'child':ci,'exact_population_tower_mse':float(np.dot(s.mass,residual**2))})
        truth=np.column_stack([s.truth(m)[idx] for m in panels])
        np.savez_compressed(PRIV/f'synthetic_{world}.npz',y=ty,truth=truth,**pp)
        print(json.dumps({'synthetic_world_complete':world,'n':len(ty),'selected_beta':c['coherence_candidates'][winner]}),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'synthetic_panel_metrics.csv',index=False);pd.DataFrame(tower).to_csv(OUT/'synthetic_tower.csv',index=False);dump(OUT/'synthetic_selection.json',selection)


def clinical_inputs():
    plan=json.loads(Path('results/cardiac_stage_c_v2/experiment_plan.json').read_text())
    for p,h in [(plan['input_bundle'],plan['input_sha256']),(plan['folds_private_path'],plan['folds_sha256'])]:
        if sha256(p)!=h:raise ValueError('Frozen clinical inputs changed')
    features=plan['panels']['ecg_age_sex_routine_ntprobnp_troponin_t']
    with np.load(plan['input_bundle']) as z:
        names=z['names'].tolist();X=z['X'][:,[names.index(n) for n in features]].copy();y=z['y'].copy();ids=z['subject_id'].copy()
    with np.load(plan['folds_private_path']) as z:folds={n:z[n].copy() for n in z.files}
    np.testing.assert_array_equal(ids,folds['subject_id']);np.testing.assert_array_equal(y,folds['y'])
    panel_columns={'ecg':list(range(9)),'age_recorded_sex':list(range(9,12)),'ecg_clinical':list(range(12)),
                   'ecg_routine':list(range(9))+list(range(12,16)),'ecg_clinical_routine':list(range(16)),
                   'plus_ntprobnp':list(range(17)),'plus_troponin_t':list(range(16))+[17],'plus_both':list(range(18))}
    panels=[]
    for columns in panel_columns.values():
        m=np.zeros(18,np.float32);m[columns]=1;panels.append(m)
    return X,y,folds,np.array(panels),list(panel_columns),features


def clinical():
    c=cfg();X,y,folds,panels,names,features=clinical_inputs();selection=[];tower=[]
    predictions={f:np.full((len(y),len(panels)),np.nan) for f in FAMILIES}
    for outer in range(5):
        checkpoint=PRIV/f'clinical_fold_{outer}.npz';meta=OUT/f'clinical_fold_{outer}.json'
        test=folds[f'outer_{outer}_test'];tr=folds[f'outer_{outer}_train'];seed=c['seed']+1000+outer
        if checkpoint.exists():
            with np.load(checkpoint) as z:
                for f in FAMILIES:predictions[f][test]=z[f]
            mm=json.loads(meta.read_text());selection.append(mm);tower.extend(mm['tower']);continue
        losses=np.zeros((len(c['coherence_candidates']),4))
        for inner in range(4):
            a=folds[f'outer_{outer}_inner_{inner}_train'];b=folds[f'outer_{outer}_inner_{inner}_validation']
            if np.intersect1d(a,b).size or np.intersect1d(np.r_[a,b],test).size:raise ValueError('Fold leakage')
            e=Encoder().fit(X[a]);constraints=kernel_constraints(X[a],e,12,seed+inner,c['constraint_rows'],c['conditional_particles'])
            for j,beta in enumerate(c['coherence_candidates']):
                model=train(X[a],y[a],panels,12,seed+inner,c,beta,constraints)
                losses[j,inner]=np.mean([brier_score_loss(y[b],predict(model,X[b],m)) for m in panels])
        winner=int(np.argmin(losses.mean(1)));beta=c['coherence_candidates'][winner]
        e=Encoder().fit(X[tr]);constraints=kernel_constraints(X[tr],e,12,seed,c['constraint_rows'],c['conditional_particles'])
        standard=[train(X[tr],y[tr],panels,12,seed,c,fixed=m) for m in panels]
        shared=train(X[tr],y[tr],panels,12,seed,c);coherent=train(X[tr],y[tr],panels,12,seed,c,beta,constraints)
        # Evaluation law fitted only outer training records; held-out parents only.
        # Pool training donors with held-out queries, but fit neighbors ONLY on train.
        eval_a,eval_b,eval_w=heldout_constraints(X[tr],X[test],e,seed,c)
        foldtower=[]
        for family,fitted in zip(FAMILIES,[standard,shared,coherent]):
            p=np.column_stack([predict(fitted[j] if family=='standard_relu' else fitted,X[test],m) for j,m in enumerate(panels)])
            predictions[family][test]=p
            if family!='standard_relu':
                model=fitted[0]
                with torch.no_grad():
                    parent=torch.sigmoid(model(torch.from_numpy(eval_a))).numpy()
                    child=torch.sigmoid(model(torch.from_numpy(eval_b.reshape(-1,eval_b.shape[-1])))).numpy().reshape(len(eval_a),-1)
                foldtower.append({'outer_fold':outer,'family':family,'kernel_record_tower_mse':float(np.mean((parent-(child*eval_w).sum(1))**2)),'n':len(eval_a)})
                save_model(PRIV/f'{family}_fold_{outer}.pt',fitted)
            else:
                for j,ff in enumerate(fitted):save_model(PRIV/f'{family}_{names[j]}_fold_{outer}.pt',ff)
        # Standard per-panel MLP evaluated on comparable supported nested panels.
        # Same training donor law and same held-out parents as shared networks below.
        for family,fitted in [('standard_relu',standard)]:
            smse=[]
            for parent_idx,child_idx in [(2,4),(4,5),(4,6),(5,7),(6,7)]:
                aa,bb,ww=heldout_constraints(X[tr],X[test],e,seed,c,panels[parent_idx],panels[child_idx])
                with torch.no_grad():
                    pa=torch.sigmoid(fitted[parent_idx][0](torch.from_numpy(aa))).numpy()
                    pb=torch.sigmoid(fitted[child_idx][0](torch.from_numpy(bb.reshape(-1,bb.shape[-1])))).numpy().reshape(len(aa),-1)
                smse.append(np.mean((pa-(pb*ww).sum(1))**2))
            foldtower.append({'outer_fold':outer,'family':family,'kernel_record_tower_mse':float(np.mean(smse)),'n':len(test)*5})
        # All three models on the identical five supported nested edges (primary).
        matched=[]
        for parent_idx,child_idx in [(2,4),(4,5),(4,6),(5,7),(6,7)]:
            aa,bb,ww=heldout_constraints(X[tr],X[test],e,seed,c,panels[parent_idx],panels[child_idx])
            for family,fitted in zip(FAMILIES,[standard,shared,coherent]):
                ma=fitted[parent_idx][0] if family=='standard_relu' else fitted[0]
                mb=fitted[child_idx][0] if family=='standard_relu' else fitted[0]
                with torch.no_grad():
                    pa=torch.sigmoid(ma(torch.from_numpy(aa))).numpy()
                    pb=torch.sigmoid(mb(torch.from_numpy(bb.reshape(-1,bb.shape[-1])))).numpy().reshape(len(aa),-1)
                matched.append({'outer_fold':outer,'family':family,'parent':names[parent_idx],'child':names[child_idx],'kernel_record_tower_mse':float(np.mean((pa-(pb*ww).sum(1))**2)),'n':len(aa)})
        dump(OUT/f'clinical_matched_tower_{outer}.json',matched)
        metadata={'outer_fold':outer,'n_train':len(tr),'n_test':len(test),'selected_beta':beta,'inner_brier':losses.tolist(),'tower':foldtower}
        dump(meta,metadata);selection.append(metadata);tower.extend(foldtower)
        np.savez_compressed(checkpoint,**{f:predictions[f][test] for f in FAMILIES})
        print(json.dumps({'clinical_fold_complete':outer,'n':len(test),'selected_beta':beta}),flush=True)
    if not all(np.isfinite(p).all() for p in predictions.values()):raise ValueError('Incomplete predictions')
    np.savez_compressed(PRIV/'clinical_predictions.npz',y=y,**predictions)
    dump(OUT/'clinical_selection.json',selection);dump(OUT/'clinical_panels.json',{'names':names,'features':features,'masks':panels.tolist()})
    pd.DataFrame(tower).to_csv(OUT/'clinical_tower_exploratory.csv',index=False)
    pd.DataFrame([r for k in range(5) for r in json.loads((OUT/f'clinical_matched_tower_{k}.json').read_text())]).to_csv(OUT/'clinical_tower_matched.csv',index=False)


def heldout_constraints(trainX,testX,encoder,seed,c,parent=None,child=None):
    rng=np.random.default_rng(seed);d=trainX.shape[1];n=len(testX);k=c['conditional_particles']
    pm=masks_from_bits(rng.integers(0,63,n),d,12) if parent is None else np.tile(parent,(n,1))
    cm=pm.copy() if child is None else np.tile(child,(n,1))
    if child is None:
        for i in range(n):cm[i,rng.choice(np.flatnonzero(pm[i]==0))]=1
    from sklearn.neighbors import NearestNeighbors
    donors=np.empty((n,k),int);distance=np.empty((n,k))
    for mask in np.unique(pm,axis=0):
        rows=np.flatnonzero(np.all(pm==mask,axis=1));encoded=encoder.encode(trainX,mask)
        dd,ii=NearestNeighbors(n_neighbors=k,n_jobs=1).fit(encoded).kneighbors(encoder.encode(testX[rows],mask));donors[rows]=ii;distance[rows]=dd
    weights=np.exp(-.5*(distance/np.maximum(distance[:,-1:],1e-6))**2);weights/=weights.sum(1,keepdims=True)
    return encoder.encode(testX,pm),encoder.encode(trainX[donors].reshape(-1,d),np.repeat(cm,k,axis=0)).reshape(n,k,-1),weights


def statistics():
    c=cfg();rows=[];pairs=[];bins=[]
    X,y,_,panels,names,_=clinical_inputs()
    with np.load('private-data/cardiac_stage_e_v1/subset_masks.npz') as z:conservative=z['conservative'].copy()
    for kind in ['clinical']+['synthetic_'+w for w in c['worlds']]:
        with np.load(PRIV/f'{kind}_predictions.npz' if kind=='clinical' else PRIV/f'{kind}.npz') as z:data={k:z[k].copy() for k in z.files}
        yy=data['y'];panels_=names if kind=='clinical' else [str(j) for j in range(16)]
        groups={'full':np.ones(len(yy),bool)}
        if kind=='clinical':groups.update({'conservative':conservative,'ntprobnp_recorded':np.isfinite(X[:,16]),'ntprobnp_unrecorded':~np.isfinite(X[:,16])})
        for group,keep in groups.items():
            for j,panel in enumerate(panels_):
                draws_by_family={};points={}
                for family in FAMILIES:
                    p=data[family][keep,j];label=yy[keep];dec={1/21:p>=1/21}
                    metrics=weighted_metrics(label,p,dec,np.ones((1,len(label))));point={k:float(v[0]) for k,v in metrics.items()}
                    draws=bootstrap_draws(label,p,dec,c['bootstrap_repeats'],c['seed'])
                    # Recover weighted decision loss from paired sensitivity/spec draws.
                    prev=float(label.mean());point['diagnostic_loss']=float(decision_loss(label,p).mean())
                    # Bootstrap prevalence must vary with the resampled labels.
                    rng=np.random.default_rng(c['seed']);loss=decision_loss(label,p);lossdraw=[];mse=[]
                    for start in range(0,c['bootstrap_repeats'],96):
                        ii=rng.integers(0,len(label),(min(96,c['bootstrap_repeats']-start),len(label)))
                        lossdraw.extend(loss[ii].mean(1))
                        if kind!='clinical':mse.extend(((p-data['truth'][keep,j])**2)[ii].mean(1))
                    draws['diagnostic_loss']=np.array(lossdraw)
                    if kind!='clinical':point['conditional_probability_mse']=float(np.mean((p-data['truth'][keep,j])**2));draws['conditional_probability_mse']=np.array(mse)
                    cal,calbins=calibration(label,p);point.update(cal)
                    row={'dataset':kind,'group':group,'panel':panel,'family':family,'n':len(label),'positive':int(label.sum()),**point}
                    for metric,v in draws.items():
                        lo,hi,_=percentile(v);row[metric+'_low']=lo;row[metric+'_high']=hi
                    # Calibration helper's details retained verbatim as aggregate JSON.
                    bins.append({'dataset':kind,'group':group,'panel':panel,'family':family,'calibration':cal,'bins':calbins})
                    rows.append(row);draws_by_family[family]=draws;points[family]=point
                for a,b in itertools.combinations(FAMILIES,2):
                    for metric in draws_by_family[a]:
                        delta=draws_by_family[b][metric]-draws_by_family[a][metric];lo,hi,_=percentile(delta)
                        pairs.append({'dataset':kind,'group':group,'panel':panel,'comparison':b+' minus '+a,'metric':metric,'difference':points[b][metric]-points[a][metric],'low':lo,'high':hi})
        print(json.dumps({'statistics_complete':kind}),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
    # Some descriptive calibration fits can return undefined values; JSON nulls.
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        if isinstance(x,np.integer):return int(x)
        return x
    dump(OUT/'calibration.json',clean(bins))


def finalize():
    cfg();before=json.loads((PRIV/'preservation_before.json').read_text())
    after=snapshot(before.keys())
    if before!=after:raise ValueError('Protected prior artifacts changed')
    dump(OUT/'_SUCCESS.json',{'status':'COMPLETE','protected_files_unchanged':len(before),'public_sha256':{p.name:sha256(p) for p in OUT.iterdir() if p.is_file() and p.name!='_SUCCESS.json'},'code_sha256':json.loads((OUT/'source_hashes.json').read_text())})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','synthetic','clinical','statistics','finalize']);args=p.parse_args()
    {'prepare':start,'synthetic':synthetic,'clinical':clinical,'statistics':statistics,'finalize':finalize}[args.stage]()
if __name__=='__main__':main()
