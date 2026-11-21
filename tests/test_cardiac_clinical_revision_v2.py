import json,unittest
from pathlib import Path
import numpy as np
from medical_ai.cardiac.clinical_revision_models import fit_revision,independent_audit,parent_instruments
from medical_ai.cardiac.stage_g_model import Encoder,predict

class ClinicalRevisionTests(unittest.TestCase):
    def config(self):
        c=json.loads(Path('config/cardiac/clinical_neural_revision.v2.json').read_text());c.update(epochs=2,batch_size=32)
        c['audit'].update(n_estimators=8,min_samples_leaf=5)
        return c
    def data(self):
        rng=np.random.default_rng(1);X=rng.normal(size=(160,18));X[rng.random(X.shape)<.1]=np.nan
        y=(np.nan_to_num(X[:,0])+.2*np.nan_to_num(X[:,12])>0).astype(int)
        masks=np.ones((2,18));masks[0,12:]=0
        return X,y,masks
    def test_zero_penalty_is_exact_control(self):
        X,y,m=self.data();a=fit_revision(X,y,m,3,self.config());b=fit_revision(X,y,m,3,self.config(),beta=0)
        np.testing.assert_array_equal(predict(a,X,m[0]),predict(b,X,m[0]))
    def test_hidden_value_and_availability_noninterference(self):
        X,y,m=self.data();e=Encoder().fit(X);alter=X.copy();alter[:,12:]=9999
        np.testing.assert_array_equal(e.encode(X,m[0]),e.encode(alter,m[0]))
        h,n=parent_instruments(X,X,e,m[0]);hh,nn=parent_instruments(alter,X,e,m[0]);np.testing.assert_array_equal(h,hh)
    def test_constant_audit_is_zero(self):
        X,y,m=self.data();p=np.full(len(X),.2)
        a=independent_audit(X,X,Encoder().fit(X),m[0],p,p,self.config())
        np.testing.assert_allclose(a['projection'],p,atol=1e-12)
        np.testing.assert_allclose(a['projection_squared_error'],0,atol=1e-24)
        np.testing.assert_array_equal(a['moments'],0)
    def test_audit_is_honest_and_has_no_outcome_argument(self):
        import inspect
        self.assertNotIn('y',inspect.signature(independent_audit).parameters)
        X,y,m=self.data();child=np.linspace(.05,.95,len(X));parent=np.full(len(X),.5)
        c=self.config();a=independent_audit(X,X,Encoder().fit(X),m[0],parent,child,c)
        changed=child.copy();changed[0]=.999
        b=independent_audit(X,X,Encoder().fit(X),m[0],parent,changed,c)
        # Own target never influences the held-out audit projection.
        self.assertEqual(a['projection'][0],b['projection'][0])
    def test_grid_and_population_specification(self):
        c=self.config();self.assertEqual(c['coherence_candidates'][0],0)
        self.assertEqual(len(set(c['training_seeds'])),5)
        final_seeds=[seed+1000+k for seed in c['training_seeds'] for k in range(5)]
        self.assertEqual(len(set(final_seeds)),25)

if __name__=='__main__':unittest.main()
