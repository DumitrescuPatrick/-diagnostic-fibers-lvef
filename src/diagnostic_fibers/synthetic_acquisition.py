"""Shared synthetic potential outcomes, Stage E learner, and matched policy benchmarks."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

def estimator(family,parameter,seed,calibration_folds):
    if family!='hist_gradient_boosting':raise ValueError('Synthetic benchmark uses native HGB only')
    return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('model',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.05,
            l2_regularization=parameter,early_stopping=False,random_state=seed))])
from .acquisition import ParticleFiber,Policy,SealedEnvironment,LossSpec,run_episode,N_TESTS

PHYSICAL_VALUES=np.array([[130.,145.],[3.2,5.4],[.8,2.4],[9.,14.],[200.,4000.],[.005,.2]])


def table(world):
    base=((np.arange(8)[:,None]>>np.arange(3))&1)
    labs=((np.arange(64)[:,None]>>np.arange(6))&1)
    B=np.repeat(base,64,axis=0);L=np.tile(labs,(8,1))
    mass=[];prob=[]
    for b,l in zip(B,L):
        b0,age,sex=b
        pb=(.3 if b0 else .7)*(.45 if age else .55)*.5
        if world=='interaction':
            ps=[.5,.5,.25,.5,.2,.2]
            pl=np.prod([p if x else 1-p for p,x in zip(ps,l)])
            logit=-4.5+1.2*b0+.6*age+.1*sex+5*(l[0]!=l[1])*(.3+.7*b0)+l[4]
        elif world in ('redundancy','null_laboratory'):
            p0=.15+.35*b0;p2=.15+.3*age
            pl=(p0 if l[0] else 1-p0)*(.9 if l[1]==l[0] else .1)*(p2 if l[2] else 1-p2)*.5*(.92 if l[4]==l[0] else .08)*(.85 if l[5]==l[4] else .15)
            logit=(-4+1.4*b0+.8*age+.2*sex+2.2*l[0]+1.2*l[2]+1.5*l[4]) if world=='redundancy' else (-3.5+1.6*b0+.6*age+.25*sex)
        else:raise ValueError('Unknown prespecified synthetic world')
        mass.append(pb*pl);prob.append(expit(logit))
    mass=np.asarray(mass);mass/=mass.sum()
    return B,L,mass,np.asarray(prob)


def features(B,L):
    b0,age,sex=B.T
    baseline=np.column_stack([70+20*b0,160+20*b0,90+30*b0,380+20*b0,430+20*b0,
                              100+10*b0,45+15*b0,20+30*b0,30+60*b0,45+30*age,np.zeros(len(B)),sex])
    labs=np.column_stack([PHYSICAL_VALUES[j,L[:,j]] for j in range(N_TESTS)])
    return np.column_stack([baseline,labs])


def sample(world,n,seed,failure_probability):
    B,L,mass,q=table(world);rng=np.random.default_rng(seed)
    state=rng.choice(len(q),size=n,p=mass)
    return {'state':state,'X':features(B[state],L[state]),'baseline_group':state//64,
            'categories':L[state], 'y':(rng.random(n)<q[state]).astype(int),
            'failures':rng.random((n,N_TESTS))<failure_probability}


def panels(costs,budget,turnarounds,deadline):
    return [tuple(j for j in range(N_TESTS) if mask>>j&1) for mask in range(1<<N_TESTS)
            if sum(costs[j] for j in range(N_TESTS) if mask>>j&1)<=budget and
               sum(turnarounds[j] for j in range(N_TESTS) if mask>>j&1)<=deadline]


def choose_fixed_panel(models,categories,failures,y,costs,budget,turnarounds,deadline):
    candidates=panels(costs,budget,turnarounds,deadline);best=None;best_score=np.inf
    for panel in candidates:
        losses=[]
        for model,cats,failed,yy in zip(models,categories,failures,y):
            observed=tuple(int(cats[j]) if j in panel and not failed[j] else -1 for j in range(N_TESTS))
            p,ess,_=model.posterior(observed)
            if not np.isfinite(p):p=model.posterior((-1,)*N_TESTS)[0]
            losses.append(model.loss.realized(int(yy),model.loss.decision(p)))
        score=np.mean(losses)+models[0].loss.cost_multiplier*sum(costs[j] for j in panel)
        if score<best_score-1e-12:best_score,best=score,panel
    return best,float(best_score)


def run_synthetic(config):
    all_rows=[];training=[]
    loss=LossSpec(config['false_negative_penalty'],config['false_positive_penalty'],config['test_cost_multiplier'])
    for wnum,world in enumerate(config['synthetic_worlds']):
        seed=config['seed']+100*wnum
        train=sample(world,config['synthetic_train'],seed,.0)
        validation=sample(world,config['synthetic_validation'],seed+1,config['synthetic_failure_probability'])
        test=sample(world,config['synthetic_test'],seed+2,config['synthetic_failure_probability'])
        # Exactly the Stage E HGB architecture, freshly trained on synthetic data.
        learned=estimator('hist_gradient_boosting',1.,seed,3).fit(train['X'],train['y'])
        B,L,mass,true_q=table(world);learned_q=learned.predict_proba(features(B,L))[:,1]
        counts=np.bincount(train['state'],minlength=512)+.5
        models=[ParticleFiber(L[g*64:(g+1)*64],learned_q[g*64:(g+1)*64],counts[g*64:(g+1)*64],loss,
                              config['minimum_particle_ess'],config['synthetic_failure_probability']) for g in range(8)]
        for budget in config['synthetic_budgets']:
            fixed,valscore=choose_fixed_panel([models[g] for g in validation['baseline_group']],validation['categories'],
                validation['failures'],validation['y'],config['costs'],budget,config['turnaround_minutes'],config['deadline_minutes'])
            training.append({'world':world,'budget':budget,'fixed_panel':[int(j) for j in fixed],
                             'validation_objective':valscore,'train_n':len(train['y']),'validation_n':len(validation['y'])})
            strategies=config['prespecified_policy_names']+config['synthetic_additional_policies']
            for strategy in strategies:
                if strategy=='fixed_all' and sum(config['costs'])>budget:continue
                panel=(0,1,2,3) if strategy=='fixed_routine' else tuple(range(N_TESTS)) if strategy=='fixed_all' else fixed
                print(f'Synthetic {world}, budget {budget}, {strategy}',flush=True)
                policy_cache={g:Policy(models[g],strategy,config['costs'],config['turnaround_minutes'],budget,config['deadline_minutes'],panel) for g in range(8)}
                for i,(group,cats,yy,failed) in enumerate(zip(test['baseline_group'],test['categories'],test['y'],test['failures'])):
                    policy=policy_cache[group]
                    environment=SealedEnvironment(cats,test['X'][i,12:],'synthetic_order',failed)
                    result=run_episode(policy,environment)
                    result.update({'population':world,'budget':budget,'strategy':strategy,'row':i,'y':int(yy),
                                   'clinical_error_loss':loss.realized(int(yy),result['decision']),
                                   'total_objective':loss.realized(int(yy),result['decision'])+loss.cost_multiplier*result['nominal_cost'],
                                   'source':'synthetic_potential_outcomes'})
                    all_rows.append(result)
    import pandas as pd
    frame=pd.DataFrame(all_rows)
    return frame
