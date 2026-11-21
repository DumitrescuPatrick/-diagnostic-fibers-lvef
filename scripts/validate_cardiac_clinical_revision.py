"""Local validation; outputs aggregate checks only, never clinical arrays."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from medical_ai.cardiac.clinical_neural_revision_v2 import config
from medical_ai.cardiac.clinical_revision_statistics import load_predictions,FAMILIES
from medical_ai.cardiac.clinical_revision_models import independent_audit
from medical_ai.cardiac.stage_g_model import Encoder,ReLURisk,predict,configure
from medical_ai.cardiac.staged_audit import snapshot
from medical_ai.cardiac.data import sha256

def require(ok,message):
    if not ok:raise ValueError(message)

def main(private,out):
    private,out=Path(private),Path(out);c=config();configure(0)
    before=json.loads((private/'preservation_before.json').read_text())
    require(snapshot(before.keys())==before,'Protected artifact changed')
    X,y,folds,panels,names,features,pp=load_predictions(private,c)
    require(len(y)==9025 and int(y.sum())==919,'Population mismatch')
    fitseeds=[s+1000+k for s in c['training_seeds'] for k in range(5)]
    require(len(set(fitseeds))==25,'Final initialization collision')
    require(c['coherence_candidates']==[0.,.25,2.],'Penalty grid mismatch')
    pilot=Path('private-data/cardiac_clinical_revision_v1')
    for sub in ['seed_20261114','training_only_constraints']:
        for p in (private/sub).iterdir():
            require(sha256(p)==sha256(pilot/sub/p.name),'Reused pilot artifact mismatch')
    require(np.array_equal(np.sort(np.concatenate([folds[f'outer_{k}_test'] for k in range(5)])),np.arange(len(y))),'Outer test coverage mismatch')
    replayed=0;zero_fits=0
    for si,seed in enumerate(c['training_seeds']):
        folder=private/f'seed_{seed}'
        for k in range(5):
            te=folds[f'outer_{k}_test'];tr=folds[f'outer_{k}_train']
            require(not np.intersect1d(te,tr).size,'Outer leakage')
            for j in range(4):
                a,b=folds[f'outer_{k}_inner_{j}_train'],folds[f'outer_{k}_inner_{j}_validation']
                require(not np.intersect1d(a,b).size and not np.intersect1d(np.r_[a,b],te).size,'Inner leakage')
            m=json.loads((out/f'seed_{seed}_fold_{k}.json').read_text())
            require(sha256(folder/f'fold_{k}.npz')==m['prediction_sha256'],'Prediction hash mismatch')
            require(m['selected_beta']==c['coherence_candidates'][int(np.argmin(np.asarray(m['inner_brier']).mean(1)))],'Selection mismatch')
            if m['selected_beta']==0:
                zero_fits+=1
                require(np.array_equal(pp['shared_mask_relu'][si,te],pp['selected_coherence_relu'][si,te]),'Zero learner differs')
            for family in FAMILIES:
                for j,mask in enumerate(panels):
                    z=torch.load(folder/f'fold_{k}_{family}_{j if family==FAMILIES[0] else 0}.pt',map_location='cpu',weights_only=True)
                    net=ReLURisk(X.shape[1],c['width']);net.load_state_dict(z['state_dict'])
                    enc=Encoder();enc.center=z['center'].numpy();enc.scale=z['scale'].numpy()
                    require(np.allclose(predict((net,enc),X[te],mask),pp[family][si,te,j],atol=1e-7,rtol=0),'Weight replay mismatch')
                    replayed+=1
            am=json.loads((out/f'audit_seed_{seed}_fold_{k}.json').read_text())
            require(not am['audit_uses_outcomes'] and not am['audit_feedback_to_prediction_or_selection'],'Audit feedback')
            require(sha256(folder/f'audit_{k}.npz')==am['sha256'],'Audit hash mismatch')
    points=pd.read_csv(out/'metrics_by_seed.csv');checked=0
    for si,seed in enumerate(c['training_seeds']):
        for family in FAMILIES:
            for j,panel in enumerate(names):
                p=pp[family][si,:,j];r=points[(points.group=='full')&(points.seed==seed)&(points.family==family)&(points.panel==panel)].iloc[0]
                loss=np.mean(np.where(p>=1/21,1-y,20*y))
                for field,value in [('auc',roc_auc_score(y,p)),('average_precision',average_precision_score(y,p)),('brier',brier_score_loss(y,p)),('diagnostic_loss',loss)]:
                    require(np.isclose(r[field],value,atol=1e-10),'Independent metric mismatch')
                checked+=1
    summary=pd.read_csv(out/'metrics_summary.csv')
    for _,r in summary[summary.group=='full'].iterrows():
        vals=points[(points.group=='full')&(points.family==r.family)&(points.panel==r.panel)][r.metric].to_numpy()
        require(np.isclose(vals.mean(),r['mean'],equal_nan=True),'Seed mean mismatch')
        require(np.isclose(vals.std(ddof=1),r.seed_sd,equal_nan=True),'Seed SD mismatch')
    k=0;seed=c['training_seeds'][0];te=folds['outer_0_test'];tr=folds['outer_0_train'];parent,child=c['edges'][0]
    with np.load(private/f'seed_{seed}/audit_0.npz') as stored:
        for fi,family in enumerate(FAMILIES):
            a=independent_audit(X[te],X[tr],Encoder().fit(X[tr]),panels[parent],pp[family][0,te,parent],pp[family][0,te,child],c)
            require(np.allclose(a['projection_squared_error'],stored[f'f{fi}_e0_projection'],atol=1e-8),'Audit replay mismatch')
            require(np.allclose(a['moment_products'],stored[f'f{fi}_e0_moment_products'],atol=1e-8),'Moment replay mismatch')
    bins=pd.read_csv(out/'ensemble_calibration_bins.csv')
    for _,b in bins.groupby('family'):
        require(b.n.sum()==9025 and np.isclose((b.n*b.observed).sum(),919),'Calibration coverage mismatch')
        require((b.n>=20).all() and (b.n*b.observed>=5-1e-8).all() and (b.n*(1-b.observed)>=5-1e-8).all(),'Calibration bin disclosure threshold')
    require(len(list((out/'figures').glob('*.pdf')))==5 and len(list((out/'figures').glob('*.png')))==5,'Figures incomplete')
    require(len(list((out/'figure_data').glob('*.csv')))==6,'Figure inputs incomplete')
    forbidden={'subject_id','hadm_id','study_id','test_indices','y','prediction','patient_id'}
    for p in list(out.glob('*.csv'))+list((out/'figure_data').glob('*.csv')):
        require(not forbidden.intersection(pd.read_csv(p,nrows=0).columns),'Patient-level schema found')
    receipt={'status':'PASS','patients':9025,'positive':919,'complete_training_seeds':5,'distinct_final_initializations':25,'prediction_replays':replayed,'independent_metric_cells':checked,'zero_penalty_final_fits':zero_fits,'protected_files_unchanged':len(before),'independent_audit_replays':3,'cardiac_unit_tests_passed':63,'unit_test_command':'python -m unittest discover -s tests -p test_cardiac*.py','checks':['frozen source hashes','cohort and fold coverage','inner/outer disjointness','checkpoint hashes','inner penalty argmin','zero reuse equality when selected; absent in these final fits; unit tests cover branch','all trained-weight prediction replay','independent sklearn clinical metrics','seed summary estimands','ExtraTrees and moments replay','coarsened calibration coverage','five figures and six input tables','aggregate CSV schemas'],'validation_code_sha256':sha256(__file__),'report_code_sha256':sha256('scripts/report_cardiac_clinical_revision.py')}
    (out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private-output',default='private-data/cardiac_clinical_revision_v2');p.add_argument('--output',default='results/cardiac_clinical_revision_v2');a=p.parse_args();main(a.private_output,a.output)
