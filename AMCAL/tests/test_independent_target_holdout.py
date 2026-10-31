"""User-run CPU contracts for independent target evaluation and replay isolation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancements.independent_target_holdout import split_targets,purge_replay,run_holdout_case,base
from test_two_action_ddqn import Fixture


class TargetHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); base.pilot.base.legacy.core.device=torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def test_joint_id_and_feature_groups_are_disjoint_across_levels(self):
        domains={level:(torch.randn(60,1,8)+level,np.arange(60)%3,np.arange(1000,1060))
            for level in (5,7)}
        domains[7][0][1]=domains[5][0][0]
        streams,audits,manifest=split_targets(domains,42)
        stream_ids={int(i) for d in streams.values() for i in d[2]}
        audit_ids={int(i) for d in audits.values() for i in d[2]}
        stream_keys=set().union(*(set(base.pilot.feature_keys(d[0])) for d in streams.values()))
        audit_keys=set().union(*(set(base.pilot.feature_keys(d[0])) for d in audits.values()))
        self.assertFalse(stream_ids & audit_ids); self.assertFalse(stream_keys & audit_keys)
        self.assertEqual(set(audits[5][2]),set(audits[7][2]))
        self.assertEqual(1000 in audit_ids,1001 in audit_ids)
        for level in domains: self.assertEqual(len(streams[level][1])+len(audits[level][1]),60)
        _,_,again=split_targets(domains,42)
        self.assertEqual(manifest,again)
        self.assertEqual(json.loads(json.dumps(manifest)),manifest)

    def test_source_replay_removes_both_flow_and_feature_overlap(self):
        x=torch.randn(10,1,8); anchors=(x,np.arange(10)%2,np.arange(10))
        audit=(torch.stack([x[0],x[1]+100]),np.array([0,1]),np.array([1000,1]))
        kept,report=purge_replay(anchors,{5:audit})
        self.assertEqual(len(kept[1]),8); self.assertEqual(report['removed_rows'],2)
        self.assertFalse({0,1} & set(map(int,kept[2])))
        self.assertFalse(set(base.pilot.feature_keys(kept[0])) & set(base.pilot.feature_keys(audit[0])))

    def test_final_target_field_is_independent_audit_not_seen_stream(self):
        f=Fixture(); agent=f.learner()
        audit=(torch.randn(8,1,8)+50,np.arange(8)%2,np.arange(9000,9008))
        metrics=lambda value:{key:float(value) for key in ('AC','PR','RC','F1')}
        original=deepcopy(f.template.state_dict())
        def endpoint(experiment,context,data):
            self.assertIs(data,audit)
            return metrics(10 if context is f.template else 30)
        def episode(observer,*args,**kwargs):
            observer.context()
            self.assertFalse(kwargs['training'])
            return dict(target_before=metrics(80),target_after=metrics(90),labels=6),[dict(source_row=0)]
        with patch.object(base,'endpoint',side_effect=endpoint),patch.object(base,'run_episode',side_effect=episode):
            result,trace=run_holdout_case(f,agent,f.domain,audit,f.reference,42,'LearnedBetaDDQN',5)
        self.assertEqual(result['target_before']['AC'],10)
        self.assertEqual(result['target_after']['AC'],30)
        self.assertEqual(result['adaptation_stream_seen_after']['AC'],90)
        self.assertEqual(result['target_holdout_gain']['AC'],20)
        self.assertEqual(result['labels_percent_of_all_target_rows'],15.)
        for key,value in f.template.state_dict().items(): torch.testing.assert_close(value,original[key],rtol=0,atol=0)

    def test_both_methods_never_put_audit_ids_in_adaptation_trace(self):
        f=Fixture(); agent=f.learner()
        audit=(torch.randn(8,1,8)+50,np.arange(8)%2,np.arange(9000,9008))
        for method in ('LearnedBetaDDQN','UncertaintyWindow'):
            result,trace=run_holdout_case(f,agent,f.domain,audit,f.reference,42,method,5)
            self.assertEqual(result['samples'],32); self.assertEqual(result['target_holdout_rows'],8)
            self.assertEqual(result['target_total_rows'],40)
            self.assertLessEqual(result['labels'],int(.2*32))
            self.assertFalse(set(audit[2]) & {r['source_row'] for r in trace})
            self.assertEqual(result['td_updates'],0)
        self.assertEqual(agent.updates,0); self.assertEqual(agent.steps,0)

    def test_leaking_audit_or_replay_is_rejected(self):
        f=Fixture(); agent=f.learner()
        audit=tuple(part[:8] for part in f.domain)
        with self.assertRaisesRegex(ValueError,'adaptive stream'):
            run_holdout_case(f,agent,f.domain,audit,f.reference,42,'LearnedBetaDDQN',5)
        audit=tuple(part[:8] for part in f.anchors)
        with self.assertRaisesRegex(ValueError,'source replay'):
            run_holdout_case(f,agent,f.domain,audit,f.reference,42,'LearnedBetaDDQN',5)


if __name__=='__main__': unittest.main()

