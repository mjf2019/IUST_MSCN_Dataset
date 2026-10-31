"""User-run CPU tests for TRAIN-meta gradients, replay relabeling and isolation."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancements.meta_beta_ddqn import MetaBetaDDQN,MetaRecorder,ComponentReward
import amcal_meta_beta_ddqn as experiment_runner
from test_two_action_ddqn import Fixture


class MetaBetaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        self.fixture=Fixture()
        self.agent=MetaBetaDDQN(self.fixture.spec,self.fixture.actor_saved['state'],'cpu',42,
            batch_size=4,learn_every=1,target_every=2)
        self.states=[torch.randn(22)*.01 for _ in range(12)]

    def meta(self):
        self.agent.set_meta_rows([(s,.2,.02 if i%2 else -.01) for i,s in enumerate(self.states)])

    def test_calibration_uses_training_magnitudes_without_manual_beta(self):
        self.meta()
        gain=np.sqrt(np.mean([(.02 if i%2 else -.01)**2 for i in range(12)]))
        self.assertAlmostEqual(self.agent.beta_value(),1./gain,places=3)
        self.assertTrue(self.agent.calibrated)
        with self.assertRaisesRegex(ValueError,'unidentifiable'):
            other=MetaBetaDDQN(self.fixture.spec,self.fixture.actor_saved['state'],'cpu',batch_size=4)
            other.set_meta_rows([(s,1.,0.) for s in self.states])

    def test_raw_reward_is_recomputed_at_current_weight(self):
        agent=self.agent; agent.learn_every=100
        agent.remember(self.states[0],1,ComponentReward(.2,.03,10.),self.states[1],True,False)
        self.assertEqual(agent.memory[0][2:4],(.2,.03))
        row=agent.memory[0]
        agent.memory.extend([row,row,row])
        batch=agent.batch()
        a=agent.targets(batch,torch.tensor(2.))
        b=agent.targets(batch,torch.tensor(5.))
        torch.testing.assert_close(b-a,torch.full_like(a,.09))

    def test_meta_step_changes_only_beta_and_never_context_or_q_weights(self):
        self.meta(); agent=self.agent
        # Keep residuals in Huber's quadratic region to make the hypergradient observable.
        with torch.no_grad(): agent.log_beta.zero_()
        for i in range(4):
            agent.memory.append((agent._state(self.states[i]),1,.1,.01,
                agent._state(self.states[i+1]),True,False))
        agent.meta_rows=[(s,.1,.02) for s in self.states]
        policy=deepcopy(agent.policy.state_dict()); target=deepcopy(agent.target.state_dict())
        context=deepcopy(self.fixture.template.state_dict()); cnn=deepcopy(self.fixture.cnn.state_dict())
        before=agent.beta_value()
        report=agent.meta_step(agent.batch())
        self.assertTrue(math.isfinite(report['hypergradient']))
        self.assertNotEqual(report['hypergradient'],0.)
        self.assertNotEqual(before,agent.beta_value())
        for model,state in [(agent.policy,policy),(agent.target,target),
                (self.fixture.template,context),(self.fixture.cnn,cnn)]:
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,state[key],rtol=0,atol=0)

    def test_ddqn_updates_and_resume_roundtrip(self):
        self.meta(); agent=self.agent
        for i in range(8):
            agent.remember(self.states[i],1,ComponentReward(.1,.001,agent.beta_value()),
                self.states[i+1],True,False)
        self.assertGreater(agent.updates,0); self.assertGreater(agent.meta_updates,0)
        saved=agent.snapshot()
        other=MetaBetaDDQN(self.fixture.spec,self.fixture.actor_saved['state'],'cpu',42,
            batch_size=4,learn_every=1,target_every=2)
        other.restore(saved)
        self.assertEqual(agent.beta_value(),other.beta_value())
        self.assertEqual(agent.updates,other.updates); self.assertEqual(agent.steps,other.steps)
        for a,b in zip(agent.batch(),other.batch()): torch.testing.assert_close(a,b,rtol=0,atol=0)
        agent.learn(); other.learn()
        self.assertAlmostEqual(agent.beta_value(),other.beta_value(),places=7)
        for k,v in agent.policy.state_dict().items(): torch.testing.assert_close(v,other.policy.state_dict()[k])

    def test_budget_mask_and_meta_collection_do_not_train_live_agent(self):
        self.meta(); agent=self.agent; recorder=MetaRecorder(agent)
        before=deepcopy(agent.snapshot())
        for state in self.states:
            action,_=recorder.choose(state,False,1.,np.random.default_rng(42))
            self.assertEqual(action,0)
        recorder.remember(self.states[0],1,ComponentReward(.1,.02,1.),self.states[1],True,False)
        self.assertEqual(len(recorder.rows),1)
        self.assertEqual(recorder.updates,0); self.assertEqual(recorder.steps,0)
        self.assertEqual(agent.steps,before['steps']); self.assertEqual(agent.updates,before['updates'])
        self.assertEqual(agent.beta_value(),float(before['log_beta'].exp()))
        for key,value in agent.policy.state_dict().items(): torch.testing.assert_close(value,before['policy'][key])

    def test_missing_paid_components_and_nonzero_skip_are_rejected(self):
        agent=self.agent
        with self.assertRaisesRegex(ValueError,'raw components'):
            agent.remember(self.states[0],1,1.,self.states[1],False,True)
        with self.assertRaisesRegex(ValueError,'Skip reward'):
            agent.remember(self.states[0],0,1.,self.states[1],False,True)

    def test_meta_streams_and_probes_are_train_only_and_disjoint(self):
        f=self.fixture; f.a.burst_dir=Path('/unused'); f.artifact={}; f.train_anchors=f.anchors
        pools={level:(torch.randn(200,1,8)+20+level,np.arange(200)%2,
            np.arange(200)+1000*(level+1)) for level in (0,1,3)}
        inner=[(torch.randn(8,1,8)+50+level,np.arange(8)%2,
            np.arange(8)+20000+100*level) for level in (0,1,3)]
        def read(path,artifact,level,split):
            self.assertEqual(split,'train_ids')
            return pools[level]
        with patch.object(experiment_runner.base.pilot.base.legacy,'read_domain',side_effect=read) as reader:
            streams,probes,anchors,manifest=experiment_runner.reserve_meta(f,inner,32,8)
        self.assertEqual(reader.call_count,3); self.assertEqual(len(manifest),6)
        reserved=[*f.train.values(),f.reference,*inner]
        ids={int(i) for d in reserved for i in d[2]}
        keys=set().union(*(set(experiment_runner.base.pilot.feature_keys(d[0])) for d in reserved))
        for d in [*streams.values(),*probes.values()]:
            new_ids=set(map(int,d[2])); new_keys=set(experiment_runner.base.pilot.feature_keys(d[0]))
            self.assertFalse(ids & new_ids); self.assertFalse(keys & new_keys)
            ids.update(new_ids); keys.update(new_keys)
        meta_keys=set().union(*(set(experiment_runner.base.pilot.feature_keys(d[0]))
            for d in [*streams.values(),*probes.values()]))
        self.assertFalse(meta_keys & set(experiment_runner.base.pilot.feature_keys(anchors[0])))


if __name__=='__main__': unittest.main()

