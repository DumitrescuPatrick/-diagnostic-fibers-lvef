"""ReLU conditional-risk models. No unseen availability flags enter an input."""
import numpy as np
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors


def configure(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))


class Encoder:
    def fit(self, X):
        X=np.asarray(X,float)
        self.center=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in X.T])
        self.scale=np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1 for c in X.T])
        self.scale=np.maximum(self.scale,1e-6)
        return self

    def transform(self,X):
        X=np.asarray(X,float); finite=np.isfinite(X)
        return np.where(finite,(X-self.center)/self.scale,0).astype('float32'),finite.astype('float32')

    def encode(self,X,masks):
        v,r=self.transform(X);m=np.broadcast_to(np.asarray(masks,np.float32),v.shape)
        return np.concatenate([v*m,r*m,m],axis=1).astype('float32')


class ReLURisk(nn.Module):
    def __init__(self,d,width=64):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(3*d,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,x):return self.layers(x).squeeze(-1)


def masks_from_bits(bits,d,baseline):
    bits=np.asarray(bits,int).reshape(-1);m=np.ones((len(bits),d),np.float32)
    for j in range(d-baseline):m[:,baseline+j]=(bits>>j)&1
    return m


def random_masks(rng,n,d,baseline,panels):
    m=masks_from_bits(rng.integers(0,2**(d-baseline),n),d,baseline)
    choice=rng.random(n)<.5
    m[choice]=np.asarray(panels)[rng.integers(0,len(panels),choice.sum())]
    return m


def kernel_constraints(X,encoder,baseline,seed,n=2048,k=8):
    """Training-only kNN approximation to the observed-record conditional law.

    Donors retain their own baseline: kernel smoothing, NOT an exact fiber law.
    Missing assay status is a record state, never a generated biochemical result.
    Self donors are excluded. No outcome is used to choose a donor.
    """
    rng=np.random.default_rng(seed);d=X.shape[1];n=min(n,len(X))
    indices=rng.choice(len(X),n,replace=False)
    bits=rng.integers(0,2**(d-baseline)-1,n)
    parent=masks_from_bits(bits,d,baseline);child=parent.copy()
    for i in range(n):child[i,rng.choice(np.flatnonzero(parent[i]==0))]=1
    donors=np.empty((n,k),int);distances=np.empty((n,k))
    for b in np.unique(bits):
        rows=np.flatnonzero(bits==b);mask=parent[rows[0]]
        encoded=encoder.encode(X,mask);nn_=NearestNeighbors(n_neighbors=min(k+1,len(X)),algorithm='brute',n_jobs=1).fit(encoded)
        distance,idx=nn_.kneighbors(encoded[indices[rows]])
        for offset,row in enumerate(rows):
            keep=idx[offset]!=indices[row];donors[row]=idx[offset,keep][:k];distances[row]=distance[offset,keep][:k]
    # Local adaptive bandwidth; explicitly an approximation in continuous spaces.
    bandwidth=np.maximum(distances[:,-1:],1e-6)
    weights=np.exp(-.5*(distances/bandwidth)**2);weights/=weights.sum(axis=1,keepdims=True)
    a=encoder.encode(X[indices],parent)
    b=encoder.encode(X[donors].reshape(-1,d),np.repeat(child,k,axis=0)).reshape(n,k,-1)
    return a,b,weights.astype('float32')


def exact_constraints(space,encoder,baseline,seed,n=2048,k=8):
    rng=np.random.default_rng(seed);d=space.X.shape[1]
    idx=rng.choice(len(space.X),n,p=space.mass);bits=rng.integers(0,2**(d-baseline)-1,n)
    parent=masks_from_bits(bits,d,baseline);child=parent.copy();donors=np.empty((n,k),int)
    for i in range(n):
        child[i,rng.choice(np.flatnonzero(parent[i]==0))]=1
        keep=np.all(space.X[:,parent[i].astype(bool)]==space.X[idx[i],parent[i].astype(bool)],axis=1)
        candidates=np.flatnonzero(keep);w=space.mass[keep];w/=w.sum()
        donors[i]=rng.choice(candidates,k,p=w)
    return encoder.encode(space.X[idx],parent),encoder.encode(space.X[donors].reshape(-1,d),np.repeat(child,k,axis=0)).reshape(n,k,-1),np.full((n,k),1/k,np.float32)


def fit(X,y,panels,baseline,seed,epochs=40,beta=0.,constraints=None,fixed=None,width=64):
    configure(seed);rng=np.random.default_rng(seed);encoder=Encoder().fit(X);model=ReLURisk(X.shape[1],width)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    y=torch.tensor(y,dtype=torch.float32)
    if constraints is not None:ca,cb,cw=[torch.as_tensor(a,dtype=torch.float32) for a in constraints]
    for epoch in range(epochs):
        order=rng.permutation(len(X))
        for start in range(0,len(X),256):
            ii=order[start:start+256]
            mask=fixed if fixed is not None else random_masks(rng,len(ii),X.shape[1],baseline,panels)
            inputs=torch.from_numpy(encoder.encode(X[ii],mask))
            loss=nn.functional.binary_cross_entropy_with_logits(model(inputs),y[ii])
            if beta:
                jj=rng.integers(0,len(ca),min(128,len(ii)))
                # Both sides differentiated. BCE prevents constant-risk collapse.
                parent=torch.sigmoid(model(ca[jj]))
                children=torch.sigmoid(model(cb[jj].reshape(-1,cb.shape[-1]))).reshape(len(jj),-1)
                loss=loss+beta*torch.mean((parent-(children*cw[jj]).sum(1))**2)
            optimizer.zero_grad();loss.backward();optimizer.step()
    return model,encoder


def predict(fitted,X,masks):
    model,encoder=fitted;model.eval()
    with torch.no_grad():return torch.sigmoid(model(torch.from_numpy(encoder.encode(X,masks)))).numpy().astype(float)


def decision_loss(y,p,fn=20.,fp=1.):
    return np.where(p>=fp/(fn+fp),fp*(1-np.asarray(y)),fn*np.asarray(y))


def acquisition_action(p,conditional_probabilities,weights,costs,available,budget,cost_multiplier=.0025):
    """Caller supplies a declared conditional law; never accesses hidden results.

    Returns candidate index or None. A prospective caller must obey Stage F's
    timestamp/order contract. Record lookup costs are not physical order costs.
    """
    risk=lambda q:np.minimum(20*np.asarray(q),1-np.asarray(q))
    gain=[]
    for q,w,c,a in zip(conditional_probabilities,weights,costs,available):
        gain.append(float(risk(p)-np.dot(risk(q),w)-cost_multiplier*c) if a and c<=budget else -np.inf)
    winner=int(np.argmax(gain)) if gain else 0
    return winner if gain and gain[winner]>0 else None
