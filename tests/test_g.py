import unittest
import numpy as np
from diagnostic_fibers.model import Encoder,ReLURisk,exact_constraints,fit,predict,masks_from_bits,acquisition_action,kernel_constraints
from diagnostic_fibers.spaces import MedicalSpace
import torch

class StageGTests(unittest.TestCase):
    def test_exact_tower(self):
        for name in ['curved_interaction','disconnected_regions']:
            s=MedicalSpace(name)
            for bit in range(16):
                parent=masks_from_bits([bit],6,2)[0]
                for j in np.flatnonzero(parent==0):
                    child=parent.copy();child[j]=1
                    np.testing.assert_allclose(s.project(s.truth(child),parent),s.truth(parent),atol=2e-15)
            self.assertAlmostEqual(s.mass.sum(),1)
            self.assertTrue(np.all(s.mass>0))
    def test_hidden_values_and_missingness_do_not_leak(self):
        x=np.array([[1,2,np.nan],[3,4,7]],float);e=Encoder().fit(x)
        altered=x.copy();altered[:,1:]=999
        np.testing.assert_array_equal(e.encode(x,[1,0,0]),e.encode(altered,[1,0,0]))
        # Revealed record absence is different from an unqueried test.
        self.assertFalse(np.array_equal(e.encode(x,[1,0,1]),e.encode(x,[1,0,0])))
    def test_relu_and_determinism(self):
        s=MedicalSpace('curved_interaction');x,y,_=s.sample(80,1);panels=masks_from_bits(np.arange(16),6,2)
        a=fit(x,y,panels,2,12,epochs=2);b=fit(x,y,panels,2,12,epochs=2)
        np.testing.assert_array_equal(predict(a,x,panels[0]),predict(b,x,panels[0]))
        self.assertEqual(sum(isinstance(m,torch.nn.ReLU) for m in a[0].modules()),2)
    def test_conditional_donors(self):
        s=MedicalSpace('curved_interaction');e=Encoder().fit(s.X)
        a,b,w=exact_constraints(s,e,2,1,40,8)
        np.testing.assert_allclose(w.sum(1),1)
        # All donors agree exactly on visible parent coordinates, and add one test.
        for i in range(len(a)):
            parent=a[i,-6:].astype(bool);child=b[i,0,-6:].astype(bool)
            self.assertEqual(child.sum(),parent.sum()+1)
            np.testing.assert_allclose(b[i,:,:6][:,parent],np.broadcast_to(a[i,:6][parent],(8,parent.sum())))
    def test_kernel_no_outcome_argument(self):
        s=MedicalSpace('curved_interaction');x,_,_=s.sample(40,1)
        a,b,w=kernel_constraints(x,Encoder().fit(x),2,1,20,4)
        self.assertEqual(b.shape,(20,4,18));np.testing.assert_allclose(w.sum(1),1,rtol=1e-6)
    def test_cost_rule(self):
        self.assertEqual(acquisition_action(.1,[np.array([0.,1.])],[np.array([.9,.1])],[2],[True],2),0)
        self.assertIsNone(acquisition_action(.1,[np.array([0.,1.])],[np.array([.9,.1])],[2],[True],1))
        self.assertIsNone(acquisition_action(.1,[np.array([.1,.1])],[np.array([.5,.5])],[2],[True],2))
