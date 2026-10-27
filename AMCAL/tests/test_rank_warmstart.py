from copy import deepcopy
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import math
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires existing PyTorch; nothing is installed')
class RankWarmstartTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_two_stage_orchestration_never_fits_val_or_selects_warmup_checkpoint(self):
        from enhancements.budget_actor import quota_max_step
        from enhancements.standardized_actor import StandardizedQueryMaximizer
        from enhancements.rank_warmstart import pairwise_rank_step
        source=Path(__file__).resolve().parents[1]/'amcal_budget_actor.py'
        functions=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)
            and n.name in ('train','normalizer','score_reference','record_rank_stage')]
        events=[]
        def write_json(path,value):path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output=Path(tmp),utility_early=True,rank_warm_start=True,seed=42,
                actor_lr=.01,budget_fraction=.1,steps_per_epoch=2,batch_size=12,epochs=2)
            states=torch.randn(12,2)
            target=torch.linspace(-2,2,12)
            val_states=states.clone().requires_grad_(True);val_targets=target.clone().requires_grad_(True)
            def collect(*args):
                a.ranking_groups=[0]*4+[1]*4+[3]*4;a.signal_scales=dict(gap=1.)
                for level in (0,1,3):write_json(a.output/f'collection_level{level}.json',dict(samples=40,cap=4))
                return states,target
            def warm(*args):events.append('warm');return pairwise_rank_step(*args)
            def maximum(*args):events.append('max');return quota_max_step(*args)
            ns=dict(torch=torch,np=np,deepcopy=deepcopy,math=math,json=json,VERSION=1,
                StandardizedQueryMaximizer=StandardizedQueryMaximizer,
                query_window_count=lambda n,c,f:int(n*f),collect=collect,
                collect_rank_validation=lambda *args:(val_states,val_targets,[0]*4+[1]*4+[3]*4),
                protocol=lambda *args:dict(rank_warm_start=True),quota_max_step=maximum,
                evaluate_levels=lambda *args:[dict(online=dict(AC=70.),original_domain_forgetting=dict(AC=0.))]*3,
                pilot=SimpleNamespace(write_json=write_json,digest=lambda path:'testhash',
                    base=SimpleNamespace(legacy=SimpleNamespace(core=SimpleNamespace(
                        device=torch.device('cpu'),set_seed=torch.manual_seed)))))
            exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
            with patch('enhancements.rank_warmstart.pairwise_rank_step',side_effect=warm),contextlib.redirect_stdout(io.StringIO()):
                ns['train'](a,None,None,None,None,'teacherhash')
            self.assertEqual(events,['warm']*4+['max']*4)
            self.assertIsNone(val_states.grad);self.assertIsNone(val_targets.grad)
            saved=torch.load(a.output/'Models'/'budget_actor.pth',weights_only=False)
            self.assertEqual(saved['selected_epoch'],1)
            stages=json.loads((a.output/'ranking_stages.json').read_text())['stages']
            self.assertEqual([s['stage'] for s in stages],
                ['initial','rank_warmup_1','rank_warmup_2','quota_max_1','quota_max_2'])

    def test_learns_within_level_order_and_keeps_value_and_input_fixed(self):
        from enhancements.minimax_query import QueryMaximizer
        from enhancements.rank_warmstart import pairwise_rank_step,rank_metrics
        torch.manual_seed(42);torch.set_num_threads(1)
        head=QueryMaximizer(2)
        states=torch.tensor([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.],[-2.,1.],[-1.,1.],[1.,1.],[2.,1.]],requires_grad=True)
        targets=torch.tensor([-2.,-1.,1.,2.,98.,99.,101.,102.],requires_grad=True)
        groups=[0]*4+[1]*4
        value=deepcopy(head.value.state_dict())
        optimizer=torch.optim.Adam(head.actor.parameters(),lr=.01)
        for _ in range(60):pairwise_rank_step(head,optimizer,states,targets,groups)
        report=rank_metrics(head,states,targets,groups,.5)
        self.assertGreater(report['spearman'],.99);self.assertAlmostEqual(report['target_regret'],0.)
        self.assertIsNone(states.grad);self.assertIsNone(targets.grad)
        for key,weight in value.items():torch.testing.assert_close(weight,head.value.state_dict()[key])

    def test_no_cross_level_comparisons_or_tied_pair_training(self):
        from enhancements.minimax_query import QueryMaximizer
        from enhancements.rank_warmstart import pairwise_rank_step
        head=QueryMaximizer(2);optimizer=torch.optim.Adam(head.actor.parameters())
        # Only cross-level differences exist, so a global-level shortcut is forbidden.
        with self.assertRaises(ValueError):
            pairwise_rank_step(head,optimizer,torch.zeros(4,2),torch.tensor([0.,0.,100.,100.]),[0,0,1,1])

    def test_following_max_phase_still_ascends_the_original_signed_objective(self):
        from enhancements.budget_actor import quota_max_step,quota_probabilities
        from enhancements.minimax_query import QueryMaximizer
        from enhancements.rank_warmstart import pairwise_rank_step
        torch.manual_seed(7);head=QueryMaximizer(1)
        states=torch.tensor([[-2.],[-1.],[1.],[2.]])
        targets=torch.tensor([-4.,-3.,-2.,-1.])
        warm=torch.optim.Adam(head.actor.parameters(),lr=.01)
        for _ in range(10):pairwise_rank_step(head,warm,states,targets,[0]*4)
        optimizer=torch.optim.SGD(head.actor.parameters(),lr=.01)
        def objective():
            with torch.no_grad():return float((quota_probabilities(head(states)[0],.5)*targets).mean())
        before=objective();quota_max_step(head,optimizer,states,targets,.5)
        self.assertGreaterEqual(objective(),before-1e-7)


if __name__=='__main__':unittest.main()
