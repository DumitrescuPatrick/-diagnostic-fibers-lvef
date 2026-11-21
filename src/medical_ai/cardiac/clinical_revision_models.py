"""Seed-matched clinical ReLU revision and sampler-independent record audits."""
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from .stage_g_model import Encoder,ReLURisk,configure,random_masks


def fit_revision(X,y,panels,seed,config,beta=0.,constraints=None,fixed=None):
    configure(seed)
    # Independent streams keep supervised batches/masks identical for beta=0/>0.
    order_rng=np.random.default_rng(seed)
    mask_rng=np.random.default_rng(seed+1000003)
    constraint_rng=np.random.default_rng(seed+2000003)
    encoder=Encoder().fit(X);model=ReLURisk(X.shape[1],config['width'])
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    targets=torch.tensor(y,dtype=torch.float32)
    if beta:
        if constraints is None:raise ValueError('Positive penalty requires training-only constraints')
        ca,cb,cw=[torch.as_tensor(z,dtype=torch.float32) for z in constraints]
    for epoch in range(config['epochs']):
        order=order_rng.permutation(len(X))
        for start in range(0,len(X),config['batch_size']):
            ii=order[start:start+config['batch_size']]
            mask=fixed if fixed is not None else random_masks(mask_rng,len(ii),X.shape[1],12,panels)
            logits=model(torch.from_numpy(encoder.encode(X[ii],mask)))
            loss=nn.functional.binary_cross_entropy_with_logits(logits,targets[ii])
            if beta:
                jj=constraint_rng.integers(0,len(ca),min(128,len(ii)))
                parent=torch.sigmoid(model(ca[jj]));child=torch.sigmoid(model(cb[jj].reshape(-1,cb.shape[-1]))).reshape(len(jj),-1)
                loss=loss+beta*((parent-(child*cw[jj]).sum(1))**2).mean()
            opt.zero_grad();loss.backward();opt.step()
    return model,encoder


def parent_instruments(X,training,encoder,mask):
    """Bounded parent-only functions; no target/hidden availability information."""
    mask=np.asarray(mask,bool);v,r=encoder.transform(X);tv,tr=encoder.transform(training)
    visible=np.flatnonzero(mask)
    values=np.clip(v[:,visible],-3,3)/3;train_values=np.clip(tv[:,visible],-3,3)/3
    cols=[np.ones(len(X))];names=['intercept']
    for offset,j in enumerate(visible):
        for values_,train_,name in [(r[:,j],tr[:,j],f'finite_{j}'),(values[:,offset],train_values[:,offset],f'value_{j}'),(values[:,offset]**2,train_values[:,offset]**2,f'square_{j}')]:
            if np.std(train_)>1e-8:cols.append(values_);names.append(name)
    return np.column_stack(cols),names


def independent_audit(X,training,encoder,parent_mask,parent_p,child_p,config):
    """Evaluation only: honest projection on already risk-held-out cases.

    Fits no disease outcome and no training donor sampler. Audit regressors use
    other outer-test patients and must NEVER feed predictor tuning or deployment.
    """
    settings=config['audit'];encoded=encoder.encode(X,parent_mask)
    splits=KFold(settings['projection_folds'],shuffle=True,random_state=settings['projection_seed'])
    projection=np.full(len(X),np.nan)
    for j,(a,b) in enumerate(splits.split(encoded)):
        est=ExtraTreesRegressor(n_estimators=settings['n_estimators'],min_samples_leaf=settings['min_samples_leaf'],max_features=settings['max_features'],n_jobs=settings['n_jobs'],random_state=settings['projection_seed']+j)
        est.fit(encoded[a],np.asarray(child_p)[a]);projection[b]=est.predict(encoded[b])
    if not np.isfinite(projection).all():raise ValueError('Incomplete cross-fitted audit')
    instruments,names=parent_instruments(X,training,encoder,parent_mask)
    residual=np.asarray(child_p)-np.asarray(parent_p)
    moments=(residual[:,None]*instruments).mean(0)
    return {'projection':projection,'projection_squared_error':(np.asarray(parent_p)-projection)**2,
            'moment_names':names,'moments':moments,'moment_products':residual[:,None]*instruments,
            'raw_nested_difference':residual}
