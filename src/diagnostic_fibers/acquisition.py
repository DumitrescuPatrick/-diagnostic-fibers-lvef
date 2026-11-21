"""Matched conditional-particle acquisition policies and sealed factual environments.

Geometric and standard VOI use separate one-step calculations of the SAME
expected terminal classification loss. Geometry is not a different utility.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

N_TESTS=6
MISSING_RECORD=3
TEST_NAMES=('sodium','potassium','creatinine','hemoglobin','ntprobnp','troponin_t')


@dataclass(frozen=True)
class LossSpec:
    false_negative: float=20.
    false_positive: float=1.
    cost_multiplier: float=.0025

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.false_negative,self.false_positive,self.cost_multiplier)) or min(self.false_negative,self.false_positive)<=0 or self.cost_multiplier<0:
            raise ValueError('Invalid prespecified penalties')

    @property
    def threshold(self):return self.false_positive/(self.false_positive+self.false_negative)
    def bayes_risk(self,p):return min(self.false_negative*p,self.false_positive*(1-p))
    def decision(self,p):return int(p>=self.threshold)
    def realized(self,y,decision):return self.false_negative*int(y==1 and decision==0)+self.false_positive*int(y==0 and decision==1)


class ParticleFiber:
    """A weighted conditional population with a shared frozen outcome predictor.

    Complete synthetic particles describe potential biochemical values. Clinical
    particles describe OBSERVED record patterns; missing category 3 is a record
    absence, never a generated assay value or prospective test failure.
    """
    def __init__(self,categories,probabilities,weights=None,loss=None,min_ess=1.,failure_probability=0.):
        self.categories=np.asarray(categories,int)
        self.q=np.asarray(probabilities,float)
        self.w=np.ones(len(self.q)) if weights is None else np.asarray(weights,float)
        if self.categories.shape!=(len(self.q),N_TESTS) or not np.isfinite(self.q).all() or ((self.q<0)|(self.q>1)).any():raise ValueError('Invalid conditional particle table')
        if np.any(self.w<0) or not self.w.sum()>0:raise ValueError('Invalid particle weights')
        if not 0<=failure_probability<1:raise ValueError('Invalid synthetic failure probability')
        self.loss=loss or LossSpec();self.min_ess=min_ess;self.failure_probability=failure_probability
        self.cache={}

    def posterior(self,observed):
        observed=tuple(observed)
        if observed not in self.cache:
            keep=np.ones(len(self.q),bool)
            for j,c in enumerate(observed):
                if c>=0:keep &= self.categories[:,j]==c
            w=self.w*keep;mass=w.sum();ess=mass*mass/np.sum(w*w) if mass else 0.
            p=float(w@self.q/mass) if mass else np.nan
            self.cache[observed]=(p,ess,w)
        return self.cache[observed]

    def branches(self,observed,j):
        p,ess,w=self.posterior(observed)
        if not w.sum()>0:return []
        result=[]
        for c in np.unique(self.categories[:,j]):
            mass=float(w[self.categories[:,j]==c].sum()/w.sum())
            if mass<=0:continue
            child=list(observed);child[j]=int(c);child=tuple(child)
            cp,ce,_=self.posterior(child)
            supported=np.isfinite(cp) and ce>=self.min_ess-1e-9
            result.append((int(c),mass,child,float(cp) if supported else p,supported))
        return result

    def geometric_gain(self,observed,j):
        p,_,_=self.posterior(observed)
        after=sum(mass*self.loss.bayes_risk(cp) for c,mass,child,cp,supported in self.branches(observed,j))
        return (1-self.failure_probability)*(self.loss.bayes_risk(p)-after)

    def standard_voi_gain(self,observed,j):
        # Independently enumerate optimal terminal actions on each possible result.
        p,_,_=self.posterior(observed);before=min(self.loss.false_negative*p,self.loss.false_positive*(1-p))
        after=0.
        for c,mass,child,cp,supported in self.branches(observed,j):
            negative_action=self.loss.false_negative*cp
            positive_action=self.loss.false_positive*(1-cp)
            after+=mass*min(negative_action,positive_action)
        after=(1-self.failure_probability)*after+self.failure_probability*before
        return before-after

    def entropy_gain(self,observed,j):
        def entropy(p):
            p=np.clip(p,1e-12,1-1e-12)
            return -p*np.log(p)-(1-p)*np.log(1-p)
        p,_,_=self.posterior(observed)
        after=sum(mass*entropy(cp) for c,mass,child,cp,supported in self.branches(observed,j))
        return float((1-self.failure_probability)*(entropy(p)-after))


class Policy:
    def __init__(self,model,name,costs,turnarounds,budget,deadline,panel=()):
        self.model,self.name=model,name
        self.costs=tuple(float(x) for x in costs);self.turnarounds=tuple(float(x) for x in turnarounds)
        self.budget,self.deadline=float(budget),float(deadline);self.panel=tuple(panel)
        self.memo={}

    def feasible(self,attempted):
        paid=sum(self.costs[j] for j in attempted);elapsed=sum(self.turnarounds[j] for j in attempted)
        return [j for j in range(N_TESTS) if j not in attempted and paid+self.costs[j]<=self.budget+1e-9 and elapsed+self.turnarounds[j]<=self.deadline+1e-9]

    def continuation_value(self,observed,attempted,depth):
        key=(tuple(observed),tuple(sorted(attempted)),depth)
        if key in self.memo:return self.memo[key]
        p,_,_=self.model.posterior(observed);stop=self.model.loss.bayes_risk(p)
        if depth<=0:return stop
        value=stop
        for j in self.feasible(attempted):
            nxt=tuple(sorted(attempted+(j,)))
            after=0.
            for c,mass,child,cp,supported in self.model.branches(observed,j):
                branch=self.continuation_value(child,nxt,depth-1) if supported else self.model.loss.bayes_risk(p)
                after+=mass*branch
            after=(1-self.model.failure_probability)*after+self.model.failure_probability*self.continuation_value(observed,nxt,depth-1)
            value=min(value,self.model.loss.cost_multiplier*self.costs[j]+after)
        self.memo[key]=value
        return value

    def choose(self,observed,attempted):
        candidates=self.feasible(attempted)
        if self.name=='no_tests':return None
        if self.name.startswith('fixed_'):
            return next((j for j in self.panel if j in candidates),None)
        best=None;best_utility=0.
        p,_,_=self.model.posterior(observed)
        for j in candidates:
            if self.name in ('geometric_depth2','standard_evoi_depth2'):
                nxt=tuple(sorted(attempted+(j,)))
                after=sum(mass*(self.continuation_value(child,nxt,1) if supported else self.model.loss.bayes_risk(p))
                          for c,mass,child,cp,supported in self.model.branches(observed,j))
                after=(1-self.model.failure_probability)*after+self.model.failure_probability*self.continuation_value(observed,nxt,1)
                gain=self.model.loss.bayes_risk(p)-after
            elif self.name=='geometric_myopic':gain=self.model.geometric_gain(observed,j)
            elif self.name=='standard_evoi':gain=self.model.standard_voi_gain(observed,j)
            elif self.name=='uncertainty_entropy':gain=self.model.entropy_gain(observed,j)
            else:raise ValueError('Unknown acquisition strategy')
            utility=gain-self.model.loss.cost_multiplier*self.costs[j]
            if utility>best_utility+1e-12:
                best_utility,best=utility,j
        return best


class SealedEnvironment:
    """Only reveal a requested factual result. Labels never enter the policy API."""
    def __init__(self,categories,values,mode,synthetic_failures=None,complete_probability=None):
        self._categories=np.asarray(categories,int);self._values=np.asarray(values,float)
        self._failures=np.zeros(N_TESTS,bool) if synthetic_failures is None else np.asarray(synthetic_failures,bool)
        self.mode=mode;self._complete_probability=complete_probability
        if self._values.shape!=(N_TESTS,) or self._categories.shape!=(N_TESTS,) or ((self._categories<0)|(self._categories>MISSING_RECORD)).any():raise ValueError('Invalid sealed result schema')
        if mode=='masked_record_review' and not np.array_equal(~np.isfinite(self._values),self._categories==MISSING_RECORD):raise ValueError('A missing record cannot masquerade as an observed biochemical category')
        if mode not in ('synthetic_order','masked_record_review'):raise ValueError('Unknown episode mode')
        if mode=='synthetic_order' and (not np.isfinite(self._values).all() or (self._categories==MISSING_RECORD).any()):raise ValueError('Synthetic potential outcomes must be specified')
        self.attempted=[];self.log=[]

    def reveal_after_commit(self,j):
        if j in self.attempted:raise ValueError('Test already requested')
        self.attempted.append(j)
        if self.mode=='synthetic_order' and self._failures[j]:
            result={'status':'simulated_observed_service_failure','category':None,'value':None}
        elif not np.isfinite(self._values[j]):
            result={'status':'unobserved_counterfactual_record_missing','category':MISSING_RECORD,'value':None}
        else:
            result={'status':'observed_synthetic_assay' if self.mode=='synthetic_order' else 'observed_recorded_assay',
                    'category':int(self._categories[j]),'value':float(self._values[j])}
        self.log.append((j,result['status']))
        return result

    def complete_probability(self):
        # All record statuses have been requested; no hidden-value shortcut.
        if self.mode!='masked_record_review' or len(self.attempted)!=N_TESTS:
            raise ValueError('Full Stage E risk cannot be used before all record statuses are revealed')
        return self._complete_probability


def run_episode(policy,environment):
    observed=(-1,)*N_TESTS;attempted=();p,ess,_=policy.model.posterior(observed)
    initial_p=p;cost=0.;elapsed=0.;successful=0;missing=0;failures=0;unsupported=0
    first_unknown_cost=None
    while True:
        action=policy.choose(observed,attempted)
        if action is None:break
        # Commit accounting before the environment can return the result.
        cost+=policy.costs[action];elapsed+=policy.turnarounds[action];attempted=attempted+(action,)
        result=environment.reveal_after_commit(action)
        if result['status']=='simulated_observed_service_failure':failures+=1;continue
        if result['status']=='unobserved_counterfactual_record_missing':
            missing+=1
            if first_unknown_cost is None:first_unknown_cost=cost
        else:successful+=1
        child=list(observed);child[action]=result['category'];child=tuple(child)
        cp,ce,_=policy.model.posterior(child)
        if not np.isfinite(cp) or ce<policy.model.min_ess-1e-9:
            unsupported+=1
            # Keep the prior supported fiber for prediction; the real response remains in the factual log.
            # No further adaptive action can be supported from an empty fiber.
            if not policy.name.startswith('fixed_'):break
        else:p,ess=cp,ce;observed=child
        if len(attempted)==N_TESTS and environment.mode=='masked_record_review':
            p=float(environment.complete_probability())
    decision=policy.model.loss.decision(p)
    return {'p':float(p),'decision':decision,'initial_p':float(initial_p),'nominal_cost':cost,
            'elapsed_scenario_minutes':elapsed,'tests_requested':len(attempted),'assays_observed':successful,
            'unobserved_counterfactual_requests':missing,'simulated_service_failures':failures,
            'unsupported_updates':unsupported,'effective_particles':float(ess),
            'first_unknown_committed_cost':float(first_unknown_cost) if first_unknown_cost is not None else cost,
            'action_sequence':tuple(attempted),'mode':environment.mode}
