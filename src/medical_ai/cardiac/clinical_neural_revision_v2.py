"""MIMIC-only multis-seed revision; original cohorts/results remain untouched."""
import argparse,json,os,platform
from pathlib import Path
import numpy as np
import torch
from .data import sha256
from .staged_audit import snapshot
from .stage_g_run import clinical_inputs
from .stage_g_model import Encoder,kernel_constraints,predict
from .clinical_revision_models import fit_revision,independent_audit

PRIVATE=Path('private-data/cardiac_clinical_revision_v2');PUBLIC=Path('results/cardiac_clinical_revision_v2')
CONFIG=Path('config/cardiac/clinical_neural_revision.v2.json')
FAMILIES=['standard_relu','shared_mask_relu','selected_coherence_relu']


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def source_paths():
    return [Path(__file__),Path(__file__).with_name('clinical_revision_models.py'),Path('docs/cardiac/CLINICAL_NEURAL_REVISION_V2.md'),CONFIG,Path('tests/test_cardiac_clinical_revision_v2.py')]

def prepare():
    if PRIVATE.exists() or PUBLIC.exists():raise FileExistsError('Versioned revision output already exists')
    PRIVATE.mkdir(mode=0o700);PUBLIC.mkdir()
    protected=[p for root in ['private-data','results'] for p in Path(root).iterdir() if p not in [PRIVATE,PUBLIC]]
    protected += [Path('publication'),Path('submission'),Path('src/medical_ai/cardiac/stage_g_model.py'),Path('src/medical_ai/cardiac/stage_g_run.py')]
    dump(PRIVATE/'preservation_before.json',snapshot(protected))
    dump(PUBLIC/'config.json',json.loads(CONFIG.read_text()))
    dump(PUBLIC/'source_hashes.json',{str(p):sha256(p) for p in source_paths()})
    import importlib.metadata
    dump(PUBLIC/'versions.json',{'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','matplotlib']}})
    X,y,folds,panels,names,features=clinical_inputs()
    assert len(y)==9025 and y.sum()==919
    dump(PUBLIC/'panels.json',{'names':names,'masks':panels.tolist(),'features':features})
    print(json.dumps({'prepared_patients':len(y),'positive':int(y.sum()),'training_seeds':5,'synthetic_experiments':False}),flush=True)


def config():
    c=json.loads((PUBLIC/'config.json').read_text())
    if c!=json.loads(CONFIG.read_text()):raise ValueError('Config changed')
    for p,h in json.loads((PUBLIC/'source_hashes.json').read_text()).items():
        if sha256(p)!=h:raise ValueError('Revision source changed: '+p)
    return c


def constraints(X,key,seed,c):
    folder=PRIVATE/'training_only_constraints';folder.mkdir(exist_ok=True)
    file=folder/(key+'.npz')
    if file.exists():
        with np.load(file) as z:return z['a'].copy(),z['b'].copy(),z['w'].copy()
    result=kernel_constraints(X,Encoder().fit(X),12,seed,c['constraint_rows'],c['conditional_particles'])
    np.savez_compressed(file,a=result[0],b=result[1],w=result[2]);return result


def cache_constraints():
    c=config();X,y,folds,panels,names,features=clinical_inputs()
    for outer in range(5):
        for inner in range(4):
            a=folds[f'outer_{outer}_inner_{inner}_train']
            constraints(X[a],f'outer_{outer}_inner_{inner}',c['constraint_seed_base']+outer*10+inner,c)
        tr=folds[f'outer_{outer}_train']
        constraints(X[tr],f'outer_{outer}_full',c['constraint_seed_base']+outer*10+9,c)
        print(json.dumps({'training_constraint_pool_outer_complete':outer}),flush=True)

def fit_seed(training_seed):
    c=config()
    if training_seed not in c['training_seeds']:raise ValueError('Unplanned seed')
    X,y,folds,panels,names,features=clinical_inputs()
    folder=PRIVATE/f'seed_{training_seed}';folder.mkdir(exist_ok=True)
    for outer in range(c['outer_folds']):
        checkpoint=folder/f'fold_{outer}.npz';meta=PUBLIC/f'seed_{training_seed}_fold_{outer}.json'
        if checkpoint.exists() and meta.exists():
            m=json.loads(meta.read_text())
            if sha256(checkpoint)!=m['prediction_sha256']:raise ValueError('Changed checkpoint')
            continue
        tr,te=folds[f'outer_{outer}_train'],folds[f'outer_{outer}_test'];fitseed=training_seed+1000+outer
        losses=np.zeros((len(c['coherence_candidates']),c['inner_folds']))
        for inner in range(c['inner_folds']):
            a=folds[f'outer_{outer}_inner_{inner}_train'];b=folds[f'outer_{outer}_inner_{inner}_validation']
            if np.intersect1d(a,b).size or np.intersect1d(np.r_[a,b],te).size:raise ValueError('Patient leakage')
            cons=constraints(X[a],f'outer_{outer}_inner_{inner}',c['constraint_seed_base']+outer*10+inner,c)
            for j,beta in enumerate(c['coherence_candidates']):
                fitted=fit_revision(X[a],y[a],panels,fitseed+inner,c,beta,cons if beta else None)
                losses[j,inner]=np.mean([np.mean((predict(fitted,X[b],mask)-y[b])**2) for mask in panels])
        choice=int(np.argmin(losses.mean(1)));beta=c['coherence_candidates'][choice]
        cons=constraints(X[tr],f'outer_{outer}_full',c['constraint_seed_base']+outer*10+9,c)
        standard=[fit_revision(X[tr],y[tr],panels,fitseed,c,fixed=mask) for mask in panels]
        shared=fit_revision(X[tr],y[tr],panels,fitseed,c)
        selected=shared if beta==0 else fit_revision(X[tr],y[tr],panels,fitseed,c,beta,cons)
        predictions={}
        for family,model in zip(FAMILIES,[standard,shared,selected]):
            predictions[family]=np.column_stack([predict(model[j] if family=='standard_relu' else model,X[te],mask) for j,mask in enumerate(panels)])
        if beta==0:np.testing.assert_array_equal(predictions['shared_mask_relu'],predictions['selected_coherence_relu'])
        np.savez_compressed(checkpoint,y=y[te],test_indices=te,**predictions)
        # Store trained weights privately to support independent audit/replay.
        for family,model in zip(FAMILIES,[standard,shared,selected]):
            models=model if family=='standard_relu' else [model]
            for j,fitted in enumerate(models):
                net,encoder=fitted
                torch.save({'state_dict':net.state_dict(),'center':torch.tensor(encoder.center),'scale':torch.tensor(encoder.scale)},folder/f'fold_{outer}_{family}_{j}.pt')
        dump(meta,{'seed':training_seed,'outer_fold':outer,'n_train':len(tr),'n_test':len(te),'positive_test':int(y[te].sum()),'beta_grid':c['coherence_candidates'],'inner_brier':losses.tolist(),'selected_beta':beta,'zero_equals_shared':bool(beta==0),'prediction_sha256':sha256(checkpoint)})
        print(json.dumps({'seed':training_seed,'outer_fold_complete':outer,'n_test':len(te),'selected_beta':beta}),flush=True)


def audit_seed(seed):
    c=config();X,y,folds,panels,names,_=clinical_inputs();folder=PRIVATE/f'seed_{seed}'
    for outer in range(5):
        target=folder/f'audit_{outer}.npz';meta=PUBLIC/f'audit_seed_{seed}_fold_{outer}.json'
        if target.exists() and meta.exists():
            if sha256(target)!=json.loads(meta.read_text())['sha256']:raise ValueError('Audit checkpoint changed')
            continue
        tr,te=folds[f'outer_{outer}_train'],folds[f'outer_{outer}_test'];enc=Encoder().fit(X[tr]);data={};aggregate=[]
        with np.load(folder/f'fold_{outer}.npz') as z:
            for fi,family in enumerate(FAMILIES):
                p=z[family]
                for edge,(parent,child) in enumerate(c['edges']):
                    result=independent_audit(X[te],X[tr],enc,panels[parent],p[:,parent],p[:,child],c)
                    prefix=f'f{fi}_e{edge}'
                    data[prefix+'_projection']=result['projection_squared_error'].astype('float32')
                    data[prefix+'_nested_difference']=result['raw_nested_difference'].astype('float32')
                    data[prefix+'_moment_products']=result['moment_products'].astype('float32')
                    aggregate.append({'family':family,'parent':names[parent],'child':names[child],'n':len(te),'projection_mse':float(np.mean(result['projection_squared_error'])),'mean_nested_difference':float(np.mean(result['raw_nested_difference'])),'moment_score':float(np.mean(result['moments']**2)),'instrument_names':result['moment_names']})
        np.savez_compressed(target,test_indices=te,**data)
        dump(meta,{'seed':seed,'outer_fold':outer,'sha256':sha256(target),'audit_uses_outcomes':False,'audit_feedback_to_prediction_or_selection':False,'rows':aggregate})
        print(json.dumps({'audit_seed':seed,'outer_fold_complete':outer}),flush=True)


def finalize():
    c=config();before=json.loads((PRIVATE/'preservation_before.json').read_text())
    if snapshot(before.keys())!=before:raise ValueError('A frozen artifact changed')
    for name in ['metrics_by_seed.csv','metrics_summary.csv','paired_differences.csv','independent_consistency_by_seed.csv','validation.json','FINDINGS.md']:
        if not (PUBLIC/name).exists():raise ValueError('Incomplete revision')
    dump(PUBLIC/'_SUCCESS.json',{'status':'COMPLETE','n':9025,'positive':919,'training_seeds':c['training_seeds'],'frozen_files_unchanged':len(before),'source_sha256':json.loads((PUBLIC/'source_hashes.json').read_text()),'public_sha256':{p.name:sha256(p) for p in PUBLIC.iterdir() if p.is_file() and p.name!='_SUCCESS.json'}})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','fit','audit','statistics','finalize']);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='cache':cache_constraints()
    elif a.stage=='fit':fit_seed(a.seed)
    elif a.stage=='audit':audit_seed(a.seed)
    elif a.stage=='statistics':
        from .clinical_revision_statistics import analyze
        analyze(PRIVATE,PUBLIC,config())
    else:finalize()
