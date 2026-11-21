import unittest,json
from pathlib import Path
import numpy as np
from medical_ai.cardiac.clinical_revision_statistics import point_and_draws,seed_weights,summary_row,calibration_bins

class RevisionStatisticsTests(unittest.TestCase):
    def config(self):
        c=json.loads(Path('config/cardiac/clinical_neural_revision.v1.json').read_text());c['bootstrap_repeats']=20;return c
    def test_point_and_loss(self):
        y=np.tile([0,1],30);p=np.tile([.01,.9],30);point,draw=point_and_draws(y,p,self.config())
        self.assertEqual(point['auc'],1);self.assertEqual(point['diagnostic_loss'],0)
        np.testing.assert_array_equal(draw['diagnostic_loss'],0)
    def test_crossed_seed_weights(self):
        c=self.config();w=seed_weights(c);self.assertEqual(w.shape,(20,5));np.testing.assert_allclose(w.sum(1),1)
        r=summary_row(np.arange(5),np.tile(np.arange(5),(20,1)),w,{})
        self.assertEqual(r['mean'],2);self.assertEqual(r['patient_low'],2)
    def test_calibration_aggregation(self):
        y=np.tile([0,0,0,0,1],100);p=np.linspace(.01,.8,len(y));bins=calibration_bins(y,p)
        self.assertEqual(sum(r['n'] for r in bins),len(y));self.assertEqual(sum(r['positive'] for r in bins),int(y.sum()))
        self.assertTrue(all(r['n']>=20 and r['positive']>=5 and r['n']-r['positive']>=5 for r in bins))
if __name__=='__main__':unittest.main()
