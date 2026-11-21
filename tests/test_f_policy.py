"""Synthetic-first tests for matched adaptive acquisition and missing counterfactuals."""
import unittest
import numpy as np
from diagnostic_fibers.acquisition import ParticleFiber,Policy,SealedEnvironment,LossSpec,run_episode
from diagnostic_fibers.synthetic_acquisition import table,features,sample,PHYSICAL_VALUES


class AdaptivePolicyTests(unittest.TestCase):
    def model(self,world='interaction',group=1):
        B,L,mass,q=table(world);a=group*64;b=a+64
        return ParticleFiber(L[a:b],q[a:b],mass[a:b],LossSpec(),1.,0.)

    def policy(self,model,name,panel=()):
        return Policy(model,name,[2,2,2,2,20,12],[60,60,60,60,120,60],40,480,panel)

    def test_exact_geometric_and_standard_voi_match(self):
        rng=np.random.default_rng(3)
        for world in ['redundancy','interaction','null_laboratory']:
            for group in range(8):
                m=self.model(world,group)
                for _ in range(10):
                    cats=m.categories[rng.integers(0,len(m.q))];observed=tuple(int(c) if rng.random()<.4 else -1 for c in cats)
                    for j in range(6):
                        if observed[j]<0:self.assertAlmostEqual(m.geometric_gain(observed,j),m.standard_voi_gain(observed,j),places=12)

    def test_tower_property_and_no_negative_bayes_value(self):
        m=self.model();s=(-1,)*6;p=m.posterior(s)[0]
        for j in range(6):
            branches=m.branches(s,j)
            self.assertAlmostEqual(sum(x[1] for x in branches),1.)
            self.assertAlmostEqual(sum(x[1]*x[3] for x in branches),p)
            self.assertGreaterEqual(m.geometric_gain(s,j),-1e-12)

    def test_null_world_does_not_buy_irrelevant_labs(self):
        for group in range(8):
            m=self.model('null_laboratory',group)
            self.assertIsNone(self.policy(m,'geometric_myopic').choose((-1,)*6,()))
            self.assertIsNone(self.policy(m,'standard_evoi').choose((-1,)*6,()))

    def test_pair_lookahead_detects_symmetric_interaction(self):
        cats=np.zeros((4,6),int);cats[:,:2]=[[0,0],[0,1],[1,0],[1,1]]
        q=np.array([.001,.999,.999,.001]);m=ParticleFiber(cats,q,np.ones(4),LossSpec())
        self.assertIsNone(self.policy(m,'geometric_myopic').choose((-1,)*6,()))
        self.assertIn(self.policy(m,'geometric_depth2').choose((-1,)*6,()),[0,1])
        self.assertEqual(self.policy(m,'geometric_depth2').choose((-1,)*6,()),self.policy(m,'standard_evoi_depth2').choose((-1,)*6,()))

    def test_hidden_values_cannot_affect_first_action(self):
        m=self.model();policy=self.policy(m,'geometric_myopic')
        # Different sealed patients receive the same decision from the same current state.
        a=SealedEnvironment(np.zeros(6,int),PHYSICAL_VALUES[:,0],'synthetic_order')
        b=SealedEnvironment(np.ones(6,int),PHYSICAL_VALUES[:,1],'synthetic_order')
        choice=policy.choose((-1,)*6,())
        self.assertEqual(choice,policy.choose((-1,)*6,()))
        self.assertFalse(a.log);self.assertFalse(b.log)

    def test_missing_record_is_not_a_test_result_or_real_failure(self):
        e=SealedEnvironment(np.array([3,0,0,0,0,0]),np.array([np.nan,3.2,.8,9.,200.,.005]),'masked_record_review')
        r=e.reveal_after_commit(0)
        self.assertEqual(r['status'],'unobserved_counterfactual_record_missing');self.assertIsNone(r['value'])
        self.assertNotEqual(r['status'],'simulated_observed_service_failure')
        with self.assertRaises(ValueError):e.complete_probability()
        with self.assertRaises(ValueError):
            SealedEnvironment(np.zeros(6,int),np.array([np.nan,3.2,.8,9.,200.,.005]),'masked_record_review')
        with self.assertRaises(ValueError):e.reveal_after_commit(0)

    def test_failure_cost_retained_no_observation(self):
        m=self.model();e=SealedEnvironment(np.zeros(6,int),PHYSICAL_VALUES[:,0],'synthetic_order',np.ones(6,bool))
        r=run_episode(self.policy(m,'fixed_routine',(0,1,2,3)),e)
        self.assertEqual(r['nominal_cost'],8);self.assertEqual(r['assays_observed'],0)
        self.assertEqual(r['simulated_service_failures'],4);self.assertEqual(r['unobserved_counterfactual_requests'],0)

    def test_budget_and_deadline_are_shared_constraints(self):
        m=self.model();p=Policy(m,'fixed_all',[2,2,2,2,20,12],[60]*6,8,120,range(6))
        e=SealedEnvironment(np.zeros(6,int),PHYSICAL_VALUES[:,0],'synthetic_order')
        r=run_episode(p,e);self.assertEqual(r['tests_requested'],2);self.assertEqual(r['nominal_cost'],4)

    def test_error_penalties_determine_common_terminal_threshold(self):
        loss=LossSpec(20,1,.0025)
        self.assertAlmostEqual(loss.threshold,1/21)
        self.assertEqual(loss.realized(1,0),20);self.assertEqual(loss.realized(0,1),1)
        self.assertEqual(loss.realized(1,1),0)

    def test_synthetic_populations_and_all_potential_results_are_reproducible(self):
        for world in ['redundancy','interaction','null_laboratory']:
            B,L,mass,q=table(world)
            self.assertAlmostEqual(mass.sum(),1.)
            a=sample(world,100,12,.05);b=sample(world,100,12,.05)
            np.testing.assert_array_equal(a['X'],b['X']);np.testing.assert_array_equal(a['y'],b['y'])
            self.assertTrue(np.isfinite(a['X']).all());self.assertEqual(a['X'].shape,(100,18))


if __name__=='__main__':unittest.main()
