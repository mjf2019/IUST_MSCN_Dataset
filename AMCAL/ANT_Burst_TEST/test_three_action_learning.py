"""Optional PyTorch integration tests for warmup and validation isolation."""
import copy
import importlib.util
import random
import unittest
from argparse import Namespace


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class LearningChecks(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        import amcal_burst_three_action as routing
        self.np, self.torch, self.routing = np, torch, routing
        self.args = Namespace(seed=42, gamma=.98, batch_size=128, online_warmup=6,
                              dqn_lr=.001, context_lr=.001, budget_fraction=.2,
                              stream_order="input", eval_epsilon=1.,
                              selection_cost=.2, gain_weight=1.)
        self.device = torch.device("cpu")

    def test_online_warmup_and_small_batch(self):
        t, r = self.torch, self.routing
        agent = r.Agent(7, self.args, self.device)
        agent.warmup = self.args.online_warmup
        state = t.ones(1, 7)
        for _ in range(5):
            agent.remember(state, 0, 1., state, True, False)
        self.assertIsNone(agent.optimize())
        agent.remember(state, 2, .5, state, True, False)
        value = agent.optimize()
        self.assertTrue(self.np.isfinite(value))
        self.assertEqual(agent.steps, 1)
        self.assertEqual(agent.batch, 128)

    def test_validation_isolation_and_caps(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        t, np, r = self.torch, self.np, self.routing
        frame = pd.DataFrame({"x0":np.arange(40)/40, "x1":np.arange(40)%3,
                              "label":[0]*20+[1]*20})
        bundle = dict(scaler=StandardScaler().fit(frame.iloc[:,:-1]),
                      encoder=LabelEncoder().fit(frame.iloc[:,-1]),
                      columns=list(frame.columns), num_classes=2, context_loss_mode="aligned")
        cnn=t.nn.Sequential(t.nn.Flatten(), t.nn.Linear(2,2)).eval().requires_grad_(False)
        context=t.nn.Linear(2,2).eval()
        agent=r.Agent(7,self.args,self.device)
        state=t.ones(1,7)
        agent.remember(state,0,1.,state,True,False)
        models=[cnn,context,agent.q,agent.target]
        before=[copy.deepcopy(model.state_dict()) for model in models]
        optimizer_before=copy.deepcopy(agent.optimizer.state_dict())
        py_rng=random.getstate();np_rng=np.random.get_state();torch_rng=t.get_rng_state().clone()
        summary=r.validation(frame,bundle,cnn,context,agent,self.device,self.args)
        self.assertEqual(summary["label_cap"],8)
        self.assertLessEqual(summary["labels_used"],8)
        self.assertEqual(summary["scoring"],"before feedback")
        for model,snapshot in zip(models,before):
            for key,value in model.state_dict().items():
                self.assertTrue(t.equal(value,snapshot[key]))
        self.assertEqual(len(agent.buffer),1)
        self.assertEqual(agent.steps,0)
        self.assertEqual(agent.optimizer.state_dict(),optimizer_before)
        self.assertEqual(random.getstate(),py_rng)
        after_np=np.random.get_state()
        self.assertEqual(after_np[0],np_rng[0]);self.assertTrue(np.array_equal(after_np[1],np_rng[1]))
        self.assertEqual(after_np[2:],np_rng[2:])
        self.assertTrue(t.equal(torch_rng,t.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
