"""Independent checks supporting the publication audit, no clinical inputs."""
import itertools,unittest
import numpy as np
from diagnostic_fibers.model import ReLURisk
from diagnostic_fibers.spaces import MedicalSpace

class PublicationMathematics(unittest.TestCase):
    def test_parameter_inventory(self):
        self.assertEqual(sum(p.numel() for p in ReLURisk(18).parameters()),7745)
        self.assertEqual(sum(p.numel() for p in ReLURisk(6).parameters()),5441)
    def test_eight_particle_variance_bias(self):
        p=.3;k=8;states=np.array(list(itertools.product([0.,1.],repeat=k)))
        weights=np.prod(np.where(states==1,p,1-p),axis=1)
        penalty=np.dot(weights,(p-states.mean(1))**2)
        self.assertAlmostEqual(penalty,p*(1-p)/k,places=14)
        self.assertGreater(penalty,0)  # Exact population tower residual is zero.
    def test_constant_is_coherent_but_inaccurate(self):
        s=MedicalSpace('curved_interaction');constant=np.full(len(s.X),np.dot(s.mass,s.q))
        np.testing.assert_allclose(s.project(constant,[1,1,0,0,0,0]),constant,atol=1e-14)
        self.assertGreater(np.dot(s.mass,(constant-s.q)**2),.001)
    def test_restricted_projection_nonidentifiability(self):
        X=np.array(list(itertools.product([0,1],repeat=3)));d=X.sum(1)%2
        for bits in range(7):
            mask=(bits>>np.arange(3))&1
            _,groups=np.unique(X[:,mask.astype(bool)],axis=0,return_inverse=True)
            for group in np.unique(groups):self.assertAlmostEqual(d[groups==group].mean(),.5)
        self.assertTrue(np.all(d!=(1-d))) # Full states distinguish complementary regions.

if __name__=='__main__':unittest.main()
