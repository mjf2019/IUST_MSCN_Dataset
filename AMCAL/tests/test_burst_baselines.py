"""Scientific and causal checks for matched Burst baseline runners."""
from copy import deepcopy
import ast
import importlib.util
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_baselines as m
from baselines import dfe_burst as dfe


class BurstBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);m.core.device=torch.device('cpu')

    def setUp(self):
        m.core.set_seed(42)
        self.a=SimpleNamespace(seed=42,classes=['0','1'],budget_fraction=.25,
                               batch_size=4,af_lr=1e-5)
        rng=np.random.default_rng(5)
        self.x=rng.normal(size=(12,8)).astype(np.float32)
        self.y=np.arange(12)%2

    def test_gaussian_bound_matches_equations_and_has_gradients(self):
        mu=torch.tensor([[0.,1.],[2.,3.]],requires_grad=True)
        var=torch.tensor([[1.,2.],[2.,4.]],requires_grad=True)
        actual=dfe.GaussianCompression.pairwise_kl(mu,var)
        expected=.5*(torch.log(var[1]/var[0])+(var[0]+(mu[0]-mu[1]).square())/var[1]-1).sum()
        self.assertTrue(torch.allclose(actual[0,1],expected,atol=1e-6))
        self.assertTrue(torch.allclose(actual.diag(),torch.zeros(2),atol=1e-6))
        features=[torch.randn(4,3,2,2,requires_grad=True) for _ in range(6)]
        loss=dfe.GaussianCompression()(features);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(f.grad is not None and torch.isfinite(f.grad).all() for f in features))
        self.assertTrue(any(f.grad.abs().sum()>0 for f in features))
        self.assertLessEqual(float(loss.detach()),sum([.01,.02,.03,.04,.05,.05])*np.log(4)+1e-5)

    def test_triplet_is_literal_equation12(self):
        anchor=torch.tensor([[0.,0.]]);positive=torch.tensor([[3.,0.]]);negative=torch.tensor([[1.,0.]])
        self.assertAlmostEqual(float(dfe.constrained_triplet(anchor,positive,negative)),4.)

    def test_backbone_and_input_keep_all_100_features(self):
        transform=dfe.FlowImages().fit(np.arange(300,dtype=np.float32).reshape(3,100))
        before=transform.scaler.data_max_.copy()
        images=transform.transform(np.full((2,100),10000,dtype=np.float32))
        self.assertEqual(images.shape,(2,1,10,10))
        self.assertTrue(np.array_equal(before,transform.scaler.data_max_))
        model=dfe.DFEBackbone().eval()
        self.assertEqual(sum(isinstance(layer,torch.nn.Conv2d) and layer.kernel_size==(3,3)
                             for layer in model.modules()),13)
        self.assertEqual(sum(isinstance(layer,torch.nn.Conv2d) and layer.kernel_size==(3,3) and layer.stride==(2,2)
                             for layer in model.modules()),3)
        z,features=model(torch.tensor(images),return_features=True)
        self.assertEqual(z.shape,(2,256));self.assertEqual(len(features),6)

    def test_templates_are_separate_and_vote_is_majority_not_centroid(self):
        library=dfe.TemplateLibrary(self.x,self.y,per_class=3)
        x,y=library.arrays();self.assertEqual(len(y),6)
        previous=list(library.rows[0])
        library.add(np.ones(8,dtype=np.float32),0)
        self.assertEqual(len(library.rows[0]),3)
        self.assertTrue(np.array_equal(library.rows[0][0],previous[1]))
        prediction,_=dfe.knn_vote(np.array([[0.],[1.],[2.]],dtype=np.float32),np.array([0,1,1]),np.array([[0.]],dtype=np.float32),3)
        self.assertEqual(int(prediction[0]),1)

    def test_af_gradient_reversal_sign_and_past_only_update(self):
        af=m.af_module();x=torch.ones(2,requires_grad=True)
        af.GradientReversal.apply(x,1.).sum().backward()
        self.assertTrue(torch.equal(x.grad,-torch.ones(2)))
        model=m.af_network(8,2,torch.device('cpu'))
        saved=dict(state=m.cpu_state(model),source_x=self.x,source_y=self.y,k=1,
                   dann_source_x=self.x,dann_source_y=self.y)
        learner=m.Learner('af',saved,self.a)
        learner.update(self.x[0],0)
        self.assertEqual(learner.steps,0)  # Never use a future target to fill BatchNorm.
        learner.update(self.x[1],1)
        self.assertEqual(learner.steps,1)
        self.assertEqual(len(learner.paid_x),2)
        self.assertEqual(len(learner.support_y),len(self.y)+2)

    def test_dfe_stream_scores_before_feedback_and_never_trains_backbone(self):
        model=dfe.DFEBackbone();transform=dfe.FlowImages().fit(self.x)
        saved=dict(state=m.cpu_state(model),transform=transform,
                   library=dfe.TemplateLibrary(transform.transform(self.x),self.y,3),k=1)
        learner=m.Learner('dfe',saved,self.a);weights=m.cpu_state(learner.model)
        learner.update(self.x[0],1)
        for key,value in weights.items():self.assertTrue(torch.equal(value,learner.model.state_dict()[key]))
        domain=(torch.tensor(self.x).unsqueeze(1),self.y,np.arange(12))
        before,trace=m.evaluate_level(self.a,'dfe',saved,domain,domain[:2])
        queried={r['source_row'] for r in trace if r['queried']}
        changed=self.y.copy()
        for i in range(12):
            if i not in queried:changed[i]=1-changed[i]
        after,trace2=m.evaluate_level(self.a,'dfe',saved,(domain[0],changed,domain[2]),domain[:2])
        self.assertEqual(before['labels'],3)
        self.assertEqual([r['prediction'] for r in trace],[r['prediction'] for r in trace2])
        self.assertEqual([r['queried'] for r in trace],[r['queried'] for r in trace2])
        self.assertEqual(before['same_level']['unqueried']['n'],9)

    def test_ral_wrapper_really_queries_and_respects_global_cap(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        model=m.RAL([KNeighborsClassifier(1),DecisionTreeClassifier(random_state=42)],1.,1.,.01,budget=None)
        model.fit(self.x,self.y)
        domain=(torch.tensor(self.x).unsqueeze(1),self.y,np.arange(12))
        result,trace=m.evaluate_level(self.a,'ral',dict(model=model),domain,domain[:2])
        self.assertEqual(result['labels'],3)
        self.assertEqual(sum(row['queried'] for row in trace),3)
        self.assertEqual(result['same_level']['unqueried']['n'],9)

    @unittest.skipUnless(importlib.util.find_spec('river'),'River is optional except for ARF runs')
    def test_arf_real_forest_causal_budget_reset_and_observer_isolation(self):
        self.a.arf_trees=2;self.a.arf_train_passes=1
        with contextlib.redirect_stdout(io.StringIO()):saved=m.train_arf(self.a,(self.x,self.y))
        self.assertEqual(saved['trees'],2)
        self.assertEqual(saved['training_samples_per_pass'],12)
        domain=(torch.tensor(self.x).unsqueeze(1),self.y,np.arange(12))
        result,trace=m.evaluate_level(self.a,'arf',saved,domain,domain[:2])
        plain,trace2=m.evaluate_level(self.a,'arf',saved,domain,domain[:2],observer=False)
        self.assertEqual(trace,trace2)
        self.assertEqual(result['labels'],3)
        self.assertEqual(result['online_learn_one_calls'],3)
        self.assertEqual(result['same_level']['unqueried']['n'],9)
        queried={r['source_row'] for r in trace if r['queried']}
        changed=self.y.copy()
        for i in range(len(changed)):
            if i not in queried:changed[i]=1-changed[i]
        _,other=m.evaluate_level(self.a,'arf',saved,(domain[0],changed,domain[2]),domain[:2])
        self.assertEqual([r['prediction'] for r in trace],[r['prediction'] for r in other])
        self.assertEqual([r['queried'] for r in trace],[r['queried'] for r in other])
        learner0=m.Learner('arf',saved,self.a);learner1=m.Learner('arf',saved,self.a)
        self.assertIsNot(learner0.model,learner1.model)
        self.assertIsNot(learner0.model.models[0],learner1.model.models[0])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'arf.joblib';joblib.dump(saved,path)
            restored=joblib.load(path)
            self.assertTrue(np.array_equal(m.Learner('arf',restored,self.a).predict(self.x),learner0.predict(self.x)))

    def test_all_four_train_save_load_evaluate_integration(self):
        self.a.dfe_epochs=1;self.a.af_epochs=1;self.a.patience=1;self.a.templates=3
        self.a.dfe_betas=[.01,.02,.03,.04,.05,.05];self.a.dfe_lr=.001
        self.a.dfe_lambda=1.;self.a.dfe_alpha1=.5;self.a.dfe_alpha2=2.;self.a.dfe_eta=.5
        self.a.train_levels=[0,1,3];self.a.levels=[5];self.a.methods=['cnn','ral','af','dfe'];self.a.jobs=1
        domain=(torch.tensor(self.x).unsqueeze(1),self.y,np.arange(12))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.a.output=root/'baselines';self.a.source_run=root/'source';self.a.burst_dir=root/'data'
            models=self.a.source_run/'Models';models.mkdir(parents=True);self.a.burst_dir.mkdir()
            joblib.dump({},models/'protocol_preprocessing.joblib')
            cnn=m.core.CNN1D(8,2)
            torch.save(dict(model_state_dict=m.cpu_state(cnn)),models/'initial_cnn_model.pth')
            for level in (0,1,3,5):(self.a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv').write_text(str(level))
            with patch.object(m.shared,'read_domain',return_value=domain),contextlib.redirect_stdout(io.StringIO()) as output:
                m.train(self.a,{'feature_names':[str(i) for i in range(8)],
                               'train_ids':np.arange(12),'val_ids':np.arange(12,24),
                               'test_ids':np.arange(24,36)})
                m.evaluate(self.a,{})
            self.assertIn('Combined baselines',output.getvalue())
            self.assertEqual(len(list((self.a.output/'Results').glob('*_metrics.json'))),4)
            self.assertTrue(list((self.a.output/'Results').glob('*_same_level.csv')))


if __name__=='__main__':unittest.main()
