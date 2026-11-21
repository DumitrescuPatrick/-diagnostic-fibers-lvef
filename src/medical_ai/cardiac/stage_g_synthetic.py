"""Finite population medical spaces with exact nonlinear diagnostic fibers."""
import itertools
import numpy as np
from scipy.special import expit


class MedicalSpace:
    def __init__(self,world):
        self.world=world;self.X=np.array(list(itertools.product([-1.,0.,1.],repeat=6)))
        a,b,c,d,e,f=self.X.T
        if world=='curved_interaction':
            score=-2.8+2.3*((c-a)**2+(d-b)**2<1.6)+1.6*e*f+1.1*a*d-.7*f*f
        elif world=='disconnected_regions':
            score=-3.+2.6*((c*d>0)&(e*f<0))+1.8*((a+c)**2+(b+f)**2<1.4)+1.2*b*e
        else:raise ValueError(world)
        # Explicit nonuniform, dependent population, strictly positive support.
        density=np.exp(.35*a*c+.4*c*d+.3*b*f-.25*e*e+.2*a)
        self.mass=density/density.sum();self.q=expit(score)
        self._truth={}

    def sample(self,n,seed):
        rng=np.random.default_rng(seed);idx=rng.choice(len(self.X),n,p=self.mass)
        u=rng.random(n)
        # Disease region in augmented space (x,u): u < sigmoid(score(x)).
        y=(u<self.q[idx]).astype(int)
        return self.X[idx].copy(),y,idx

    def truth(self,mask):
        key=tuple(np.asarray(mask,int))
        if key not in self._truth:
            _,inverse=np.unique(self.X[:,np.asarray(mask,bool)],axis=0,return_inverse=True)
            den=np.bincount(inverse,weights=self.mass);num=np.bincount(inverse,weights=self.mass*self.q)
            self._truth[key]=(num/den)[inverse]
        return self._truth[key]

    def project(self,values,parent):
        _,inverse=np.unique(self.X[:,np.asarray(parent,bool)],axis=0,return_inverse=True)
        return (np.bincount(inverse,weights=self.mass*values)/np.bincount(inverse,weights=self.mass))[inverse]
