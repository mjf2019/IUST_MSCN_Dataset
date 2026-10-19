"""Equation, budget, drift-replacement, weighted-ARF and causal-stream checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import math
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.calmid_micfoal import CALMID, MicFoal, Decision, LabelWindows, features
from baselines.weighted_arf import WeightedARFClassifier, WeightedTree
from river import forest, tree


class RNG:
    def __init__(self, values):self.values=iter(values)
    def random(self):return next(self.values)
    def poisson(self, rate):return 0


class ActivePaperTests(unittest.TestCase):
    def fixture(self, kind, seed=42):
        rng=np.random.default_rng(21)
        x=rng.normal(size=(48,4));y=np.arange(48)%3
        model=CALMID(range(3),seed=seed,models=2,window=12) if kind=='calmid' else MicFoal(seed=seed,models=2,window=12)
        model.fit_source(x[:24],y[:24]);model.begin_stream(seed,.2)
        return model,x[24:],y[24:]

    def test_calmid_difficulty_and_log_weight_are_equations_1_3(self):
        model=CALMID(range(3),models=1,window=10)
        d=Decision(1,0,1,.6,.2,uncertain=True)
        self.assertAlmostEqual(model.difficulty(d,0),.8)
        self.assertAlmostEqual(model.difficulty(d,1),1.2*math.e)
        self.assertAlmostEqual(model.difficulty(d,2),1.2*math.exp(2))
        for y in [0]*6+[1]*3+[2]:model.windows.append(y,y,True)
        self.assertAlmostEqual(model.imbalance(0),1.8)
        self.assertAlmostEqual(model.imbalance(2),.3)
        model.pending=d;model.seen=1
        with patch.object(model,'_fit_weight') as fit:
            model.update(np.zeros(4),0,d)
        expected=math.log1p(.8+1/model.imbalance(0))
        self.assertAlmostEqual(model.last_weight,expected)
        self.assertAlmostEqual(fit.call_args.args[3],expected)

    def test_calmid_asymmetric_threshold_adjustments(self):
        model=CALMID(range(3),models=1)
        for y in [0]*8+[1]*2:model.windows.append(y,y,True)
        d=Decision(1,0,1,.6,.2,uncertain=True)
        model._threshold_update(d,0)
        self.assertAlmostEqual(model.thresholds[0,1],.5*.99**2)
        self.assertEqual(model.thresholds[1,0],.5)
        model.thresholds[0,1]=.49
        model._threshold_update(Decision(1,0,1,.9,.8,selective=True),1)
        self.assertEqual(model.thresholds[0,1],.5)

    def test_hybrid_order_and_native_ratio_gates_need_no_true_label(self):
        cal=CALMID(range(3),models=1)
        cal.probabilities=lambda x:{0:.6,1:.3,2:.1}
        cal.rng=RNG([0.])
        selected,d=cal.request(np.zeros(2))
        self.assertTrue(selected and d.random);self.assertFalse(d.uncertain)
        cal.update(np.zeros(2),0,d)
        # Random bypasses native ratio; global permission still blocks it.
        cal.rng=RNG([0.]);selected,d=cal.request(np.zeros(2),can_query=False)
        self.assertFalse(selected);self.assertEqual(cal.windows.rows[-1],(None,None,None))
        mic=MicFoal(models=1);mic._register(0);mic._register(1)
        mic.probabilities=lambda x:{0:.55,1:.45};mic.rng=RNG([0.])
        selected,d=mic.request(np.zeros(2))
        self.assertTrue(selected and d.random and d.uncertain)

    def test_windows_age_at_every_row_and_random_counts_are_unbiased_subset(self):
        w=LabelWindows(3)
        w.append(0,1,True);w.append(1,1,False);w.append()
        self.assertEqual(w.random,{0:1});self.assertEqual(w.true,{0:1,1:1})
        self.assertAlmostEqual(w.macro_f1(),1/3)
        w.append();self.assertEqual(w.random,{})
        w.append();self.assertIsNone(w.macro_f1())

    def test_micfoal_current_class_count_weight_and_empty_random_case(self):
        mic=MicFoal(models=1,window=5)
        for c in (0,1,2):mic._register(c)
        mic.windows.append(0,0,True);mic.windows.append(0,1,True)
        # Only one class in Wt, despite three globally known classes.
        d=Decision(1,1,0,.8,.6)
        weight,mu=mic.training_weight(0,d)
        self.assertEqual(mu,1)
        self.assertAlmostEqual(weight,1+math.log1p(2*(1+1)+1))
        weight,mu=mic.training_weight(2,d)
        self.assertEqual(mu,0)
        self.assertAlmostEqual(weight,1+math.log1p(4+2))
        mic.windows.append(1,1,False)
        d=Decision(1,1,0,.8,.6,uncertain=True)
        weight,mu=mic.training_weight(1,d)
        self.assertEqual(mu,0)
        self.assertAlmostEqual(weight,1+math.log1p(1+.9-.8+2))

    def test_micfoal_paper_step_minor_major_and_wrong_rules(self):
        def update(d,y,rows):
            mic=MicFoal(models=1,window=10)
            for c in (0,1,2):mic._register(c)
            for r in rows:mic.windows.append(r,r,True)
            mic.seen=1;mic.pending=d
            with patch.object(mic.model,'learn_one'):mic.update(np.zeros(4),y,d)
            return mic
        major=update(Decision(1,0,1,.8,.6,uncertain=True),0,[0,0,0,1])
        self.assertAlmostEqual(major.thresholds[0],.9*.9**2)
        minor=update(Decision(1,0,1,.8,.6,uncertain=True),0,[0,1,1,1,1,1])
        self.assertAlmostEqual(minor.thresholds[0],.9*.9)
        wrong=update(Decision(1,0,1,.95,.9,selective=True),1,[0,1])
        self.assertAlmostEqual(wrong.thresholds[0],.9*1.1)
        uncertain_wrong=update(Decision(1,0,1,.8,.6,uncertain=True),1,[0,1])
        self.assertAlmostEqual(uncertain_wrong.thresholds[0],.9)

    def test_micfoal_expands_only_after_paid_new_class(self):
        mic=MicFoal(models=1,window=10);mic._register(0);mic._register(1)
        mic.probabilities=lambda x:{0:.55,1:.45}
        selected,d=mic.request(np.zeros(4));self.assertTrue(selected)
        self.assertNotIn(9,mic.thresholds)
        mic.update(np.zeros(4),9,d)
        self.assertEqual(mic.thresholds[9],.9)

    def test_periodic_budget_uses_paid_window_and_is_bounded(self):
        mic=MicFoal(models=1,window=2)
        mic.windows.append(0,1,False);mic.windows.append(1,0,False)
        mic.seen=2;mic._periodic()
        self.assertAlmostEqual(mic.budget,.2*(1+.6/10))
        self.assertEqual(mic.last_period_f1,0)
        mic.windows=LabelWindows(2);mic.windows.append(0,0,True);mic.windows.append(1,1,True)
        before=mic.budget;mic.seen=4;mic._periodic()
        self.assertAlmostEqual(mic.budget,before*(1-.1/100))
        for _ in range(10000):mic.seen+=2;mic._periodic()
        self.assertGreaterEqual(mic.budget,.1)
        mic.adaptive_budget=False;before=mic.budget;mic._periodic();self.assertEqual(before,mic.budget)

    def test_calmid_replacement_is_worst_sorted_balanced_and_decayed(self):
        model=CALMID(range(3),models=3,window=30,random_ratio=.3)
        self.assertEqual(model.cache_size,3)
        for c in range(3):
            for t in (c+1,c+4):model.cache[c].append((t,np.array([t]),c,1.))
        model.clock=15
        for detector,rate in zip(model.detectors,(.1,.9,.2)):
            for _ in range(10):detector.update(rate)
        old=list(model.trees);calls=[]
        with patch.object(model,'_fit_weight',side_effect=lambda tree,x,y,w:calls.append((x[0],y,w))):
            self.assertEqual(model._replace_worst(),1)
        self.assertIsNot(old[1],model.trees[1]);self.assertIs(old[0],model.trees[0])
        self.assertEqual([row[0] for row in calls],[1,2,3,4,5,6])
        for t,y,w in calls:self.assertAlmostEqual(w,math.exp((t-15)/30)*math.log1p(2))
        self.assertEqual(model.detectors[1].estimation,0)

    def test_weighted_arf_unit_weight_matches_river_without_extra_poisson(self):
        ordinary=forest.ARFClassifier(n_models=3,seed=42)
        weighted=WeightedARFClassifier(n_models=3,seed=42)
        rng=np.random.default_rng(3)
        for i,row in enumerate(rng.normal(size=(30,4))):
            ordinary.learn_one(features(row),i%2);weighted.learn_one(features(row),i%2,w=1)
            self.assertEqual(ordinary.predict_proba_one(features(row)),weighted.predict_proba_one(features(row)))
        for a,b in zip(ordinary,weighted):
            self.assertEqual(a._train_weight_seen_by_model,b.tree._train_weight_seen_by_model)

    def test_weighted_arf_fractional_weights_backgrounds_and_pickle(self):
        ordinary=WeightedARFClassifier(n_models=3,seed=42)
        weighted=WeightedARFClassifier(n_models=3,seed=42)
        x={0:.2,1:.8}
        ordinary.learn_one(x,0,w=1);weighted.learn_one(x,0,w=2.25)
        for a,b in zip(ordinary,weighted):
            self.assertAlmostEqual(b.tree._train_weight_seen_by_model,2.25*a.tree._train_weight_seen_by_model)
        weighted._background[0]=weighted._new_base_model()
        restored=pickle.loads(pickle.dumps(weighted))
        self.assertIs(restored._background[0].weight_box,restored._sample_weight_box)
        self.assertIs(restored.data[0].weight_box,restored._sample_weight_box)
        before=restored._background[0].tree._train_weight_seen_by_model
        restored.learn_one(x,1,w=2.25)
        self.assertGreater(restored._background[0].tree._train_weight_seen_by_model,before)
        self.assertEqual(restored._sample_weight_box,[1.])

    def test_unqueried_truth_never_changes_actions_learning_or_observers(self):
        for method in ('calmid','micfoal'):
            original,x,y=self.fixture(method)
            def run(labels,observe=False):
                model=deepcopy(original);trace=[];paid=set()
                if observe:model.predict(x)
                for i,(row,label) in enumerate(zip(x,labels)):
                    prediction=model.predict_one(row)
                    selected,d=model.request(row,can_query=model.queried<4)
                    if selected:model.update(row,label,d);paid.add(i)
                    trace.append((prediction,selected,d))
                before=pickle.dumps(model)
                if observe:model.predict(x)
                self.assertEqual(before,pickle.dumps(model))
                return trace,paid,model
            first,paid,m0=run(y)
            changed=y.copy()
            for i in range(len(y)):
                if i not in paid:changed[i]=99
            second,paid2,m1=run(changed,True)
            self.assertEqual(first,second);self.assertEqual(paid,paid2)
            self.assertEqual(m0.statistics(),m1.statistics())
            self.assertEqual(list(m0.windows.rows),list(m1.windows.rows))
            self.assertEqual(pickle.dumps(m0),pickle.dumps(m1))
            self.assertLessEqual(len(paid),4)
            self.assertEqual(m0.seen,24)

    def test_real_wrapper_train_save_evaluate_both_methods(self):
        import torch
        import amcal_burst_baselines as m
        torch.set_num_threads(1);m.core.device=torch.device('cpu')
        rng=np.random.default_rng(4);x=rng.normal(size=(18,8)).astype('float32');y=np.arange(18)%3
        domain=(torch.tensor(x).unsqueeze(1),y,np.arange(18))
        a=SimpleNamespace(seed=42,classes=['0','1','2'],budget_fraction=.2,af_lr=1e-5,
            train_levels=[0,1,3],levels=[5],methods=['calmid','micfoal'],active_models=2,active_window=6)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a.output=root/'run';a.source_run=root/'source';a.burst_dir=root/'data'
            (a.source_run/'Models').mkdir(parents=True);a.burst_dir.mkdir()
            joblib.dump({},a.source_run/'Models/protocol_preprocessing.joblib')
            for level in (0,1,3,5):(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv').write_text(str(level))
            artifact=dict(feature_names=[str(i) for i in range(8)],train_ids=np.arange(18),
                          val_ids=np.arange(18,36),test_ids=np.arange(36,54))
            with patch.object(m.shared,'read_domain',return_value=domain),contextlib.redirect_stdout(io.StringIO()) as output:
                m.train(a,artifact);m.evaluate(a,artifact)
            self.assertTrue((a.output/'Models/calmid.joblib').is_file())
            self.assertTrue((a.output/'Models/micfoal.joblib').is_file())
            self.assertEqual(len(list((a.output/'Results').glob('*_metrics.json'))),2)
            self.assertIn('Online runtime',output.getvalue())
            self.assertIn('Native budget',output.getvalue())


if __name__=='__main__':unittest.main()
