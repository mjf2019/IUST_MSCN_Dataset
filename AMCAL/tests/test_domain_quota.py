from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires existing PyTorch; nothing is installed')
class DomainQuotaTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_each_level_satisfies_its_own_quota_and_gradient_check(self):
        from enhancements.budget_actor import level_quota_objective
        z=torch.tensor([-1.,-.3,.4,-.2,.5,1.2,-1.5,0.,.8],dtype=torch.double,requires_grad=True)
        scores=torch.linspace(-2,2,9,dtype=torch.double)
        groups=[0]*3+[1]*3+[3]*3;quotas={0:.25,1:.5,3:.75}
        _,p=level_quota_objective(z,scores,groups,quotas)
        for i,level in enumerate((0,1,3)):
            self.assertAlmostEqual(float(p.detach()[3*i:3*i+3].mean()),quotas[level],places=10)
        self.assertTrue(torch.autograd.gradcheck(lambda value:level_quota_objective(value,scores,groups,quotas)[0],(z,)))

    def test_per_level_target_offsets_cannot_drive_query_gradients(self):
        from enhancements.budget_actor import level_quota_objective
        z=torch.tensor([-.8,.1,.6,-.3,.4,.9],dtype=torch.double,requires_grad=True)
        scores=torch.tensor([-3.,-2.,-1.,.1,.3,.5],dtype=torch.double)
        groups=[0]*3+[1]*3;quotas={0:.25,1:.5}
        objective,_=level_quota_objective(z,scores,groups,quotas)
        first=torch.autograd.grad(objective,z)[0]
        shifted=scores+torch.tensor([1000.]*3+[-500.]*3,dtype=torch.double)
        objective,_=level_quota_objective(z,shifted,groups,quotas)
        second=torch.autograd.grad(objective,z)[0]
        torch.testing.assert_close(first,second,atol=1e-12,rtol=1e-10)

    def test_duplicating_a_level_does_not_increase_its_objective_weight(self):
        from enhancements.budget_actor import level_quota_objective
        z=torch.tensor([-1.,1.,-.5,.5],dtype=torch.double)
        s=torch.tensor([-3.,-1.,2.,4.],dtype=torch.double)
        first,_=level_quota_objective(z,s,[0,0,1,1],{0:.5,1:.5})
        index=torch.tensor([0,1,0,1,2,3])
        second,_=level_quota_objective(z[index],s[index],[0]*4+[1]*2,{0:.5,1:.5})
        torch.testing.assert_close(first,second)

    def test_sampler_is_seeded_unique_and_contains_every_level(self):
        from enhancements.budget_actor import balanced_level_indices
        groups=torch.tensor([0]*60+[1]*80+[3]*100)
        first=balanced_level_indices(groups,128,torch.Generator().manual_seed(42))
        second=balanced_level_indices(groups,128,torch.Generator().manual_seed(42))
        torch.testing.assert_close(first,second)
        self.assertEqual(len(first.unique()),len(first))
        self.assertEqual([int((groups[first]==level).sum()) for level in (0,1,3)],[42,42,42])
        with self.assertRaises(ValueError):balanced_level_indices(groups,5,torch.Generator())
        with self.assertRaises(ValueError):balanced_level_indices([0,1,1],6,torch.Generator())

    def test_max_step_keeps_inputs_and_value_head_fixed_and_improves_objective(self):
        from enhancements.budget_actor import level_quota_max_step,level_quota_objective
        from enhancements.minimax_query import QueryMaximizer
        torch.manual_seed(7);head=QueryMaximizer(1)
        states=torch.tensor([[-2.],[-1.],[1.],[2.]],requires_grad=True)
        scores=torch.tensor([-4.,-3.,-2.,-1.],requires_grad=True)
        groups=[0,0,1,1];quotas={0:.5,1:.5}
        before_value=deepcopy(head.value.state_dict())
        def objective():
            with torch.no_grad():return float(level_quota_objective(head(states)[0],scores,groups,quotas)[0])
        before=objective()
        level_quota_max_step(head,torch.optim.SGD(head.actor.parameters(),lr=.05),states,scores,groups,quotas)
        self.assertGreater(objective(),before)
        self.assertIsNone(states.grad);self.assertIsNone(scores.grad)
        for key,value in before_value.items():torch.testing.assert_close(value,head.value.state_dict()[key])

    def test_missing_quota_and_singleton_level_fail(self):
        from enhancements.budget_actor import level_quota_objective
        with self.assertRaises(ValueError):
            level_quota_objective(torch.zeros(4),torch.ones(4),[0,0,1,1],{0:.5})
        with self.assertRaises(ValueError):
            level_quota_objective(torch.zeros(3),torch.ones(3),[0,1,1],{0:.5,1:.5})


if __name__=='__main__':unittest.main()
