"""User-run CNN-only fine-tuning, bundle invariants and forgetting accounting checks."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_post_selector_cnn13 as runner
from test_two_action_ddqn import Fixture


class PostSelectorCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.base.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def args(self):
        return SimpleNamespace(dataset='IAT',beta=26.,seed=42,classes=['0','1'],cnn_epochs=2,cnn_lr=.001,patience=3)

    def test_only_the_copied_cnn_is_trained(self):
        fixture = Fixture(); agent = fixture.learner()
        cnn_hash = runner.base.weight_hash(fixture.cnn.state_dict())
        context_hash = runner.base.weight_hash(fixture.template.state_dict())
        query_hash = runner.base.weight_hash(agent.policy.state_dict())
        model = deepcopy(fixture.cnn)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            report = runner.fit_cnn(model,fixture.domain,fixture.reference,self.args(),Path(temporary)/'Models',{})
        self.assertIn(report['selected_epoch'],[1,2])
        self.assertNotEqual(cnn_hash,runner.base.weight_hash(model.state_dict()))
        self.assertEqual(cnn_hash,runner.base.weight_hash(fixture.cnn.state_dict()))
        self.assertEqual(context_hash,runner.base.weight_hash(fixture.template.state_dict()))
        self.assertEqual(query_hash,runner.base.weight_hash(agent.policy.state_dict()))
        self.assertFalse(model.training); self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_fine_tuning_resumes_after_a_completed_epoch(self):
        fixture = Fixture(); model = deepcopy(fixture.cnn); calls = [0]
        actual_metrics = runner.raw_cnn_metrics
        def interrupted(*args,**kwargs):
            calls[0] += 1
            if calls[0] == 2: raise RuntimeError('Simulated validation interruption')
            return actual_metrics(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)/'Models'
            with patch.object(runner,'raw_cnn_metrics',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    runner.fit_cnn(model,fixture.domain,fixture.reference,self.args(),directory,{})
            resumed = deepcopy(fixture.cnn)
            report = runner.fit_cnn(resumed,fixture.domain,fixture.reference,self.args(),directory,{})
            self.assertEqual([row['epoch'] for row in report['epochs']],[1,2])

    def test_bundles_have_identical_context_and_query_but_different_cnn(self):
        fixture = Fixture(); agent = fixture.learner(); fixture.healthy = dict(config={})
        context_hash = runner.base.weight_hash(fixture.template.state_dict())
        query_hash = runner.base.weight_hash(agent.policy.state_dict())
        original_cnn_hash = runner.base.weight_hash(fixture.cnn.state_dict())
        tuned = deepcopy(fixture.cnn)
        with torch.no_grad(): next(tuned.parameters()).add_(.1)
        with tempfile.TemporaryDirectory() as temporary:
            current, adapted = Path(temporary)/'current.pth',Path(temporary)/'cnn13.pth'
            runner.save_bundle(current,'Current',fixture.cnn,fixture,agent,{})
            runner.save_bundle(adapted,'CNN13',tuned,fixture,agent,{})
            runner.load_variant(fixture,agent,adapted,{},context_hash,query_hash)
            self.assertEqual(runner.base.weight_hash(fixture.cnn.state_dict()),runner.base.weight_hash(tuned.state_dict()))
            runner.load_variant(fixture,agent,current,{},context_hash,query_hash)
            self.assertEqual(original_cnn_hash,runner.base.weight_hash(fixture.cnn.state_dict()))
            bad = torch.load(adapted,map_location='cpu',weights_only=False)
            next(iter(bad['context_state_dict'].values())).add_(1.)
            torch.save(bad,adapted)
            with self.assertRaisesRegex(ValueError,'Context or DDQN changed'):
                runner.load_variant(fixture,agent,adapted,{},context_hash,query_hash)

    def test_common_reference_accounts_for_cnn_damage_and_online_damage(self):
        fixture = Fixture(); fixture.a.dataset = 'IAT'; fixture.a.beta = 26.; agent = fixture.learner()
        common = dict(AC=80.,PR=80.,RC=80.,F1=80.)
        before,after = dict(AC=70.,PR=70.,RC=70.,F1=70.),dict(AC=65.,PR=65.,RC=65.,F1=65.)
        result = dict(online=dict(AC=75.,PR=74.,RC=73.,F1=72.),samples=32,labels=4,cap=6,labels_percent=12.5,
            source_before=before,source_after=after,original_domain_forgetting={k:5. for k in common},
            target_after=dict(AC=85.,PR=84.,RC=83.,F1=82.),td_updates=0)
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()), \
                patch.object(runner.base,'run_episode',return_value=(result,[dict(step=1)])):
            row = runner.compare_case(fixture,agent,'CNN13',5,fixture.domain,fixture.reference,common,
                Path(temporary)/'case',{})
        self.assertEqual(row['cnn_finetune_source_drop'],10.)
        self.assertEqual(row['forgetting'],5.)
        self.assertEqual(row['total_source_drop'],15.)
        self.assertEqual(row['total_source_drop_F1'],15.)

    def test_development_data_reads_only_train_and_val_one_and_three(self):
        fixture = Fixture(); fixture.a.burst_dir = Path('/unused'); fixture.artifact = {}
        pools = {(split,level):(torch.randn(80,1,8)+level,np.arange(80)%2,
            np.arange(80)+(1000 if split == 'val_ids' else 0))
            for split in ('train_ids','val_ids') for level in (1,3)}
        def read_domain(burst,artifact,level,split):
            self.assertIn(level,(1,3)); self.assertIn(split,('train_ids','val_ids'))
            return pools[(split,level)]
        with patch.object(runner.base.pilot.base.legacy,'read_domain',side_effect=read_domain) as reader:
            train,val,manifest = runner.development_data(fixture,32,32)
        self.assertEqual(reader.call_count,4); self.assertEqual(len(train[1]),64); self.assertEqual(len(val[1]),64)
        self.assertEqual(set(manifest['train_ids']),{'1','3'})
        self.assertFalse(set(map(int,train[2])).intersection(map(int,val[2])))
        overlapping = tuple(part[:32] for part in pools[('train_ids',1)])
        with patch.object(runner.base.pilot.base.legacy,'read_domain',return_value=overlapping):
            with self.assertRaisesRegex(ValueError,'IDs overlap'):
                runner.development_data(fixture,32,32)


if __name__ == '__main__': unittest.main()
