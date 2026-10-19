"""Exercise production score assembly/action mapping without a Torch runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'amcal_burst_structured_bank_protocol.py'
TREE=ast.parse(SOURCE.read_text())


class Rewards(np.ndarray):
    def max(self, axis=None, keepdim=False):
        return SimpleNamespace(values=np.max(np.asarray(self),axis=axis,keepdims=keepdim))


def assemble(probabilities, update, decoupled=False):
    cls=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='StructuredQ')
    forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assignment=next(n for n in forward.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='query_reward' for t in n.targets))
    env={'self':SimpleNamespace(budget_aware=False,decoupled_query=decoupled),
         'reward':np.asarray(2*np.asarray(probabilities)-1).view(Rewards),
         'future':np.array([[7.]]),'update':np.asarray(update),
         'torch':SimpleNamespace(cat=lambda parts,dim:np.concatenate(parts,axis=dim))}
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(SOURCE),'exec'),env)
    returned=next(n for n in forward.body if isinstance(n,ast.Return))
    return eval(compile(ast.Expression(returned.value),str(SOURCE),'eval'),env)


class DecoupledQueryTests(unittest.TestCase):
    def test_useful_update_not_penalized_for_weak_current_context(self):
        p=[[.99,.2,.3,.4]];u=[[.05,.02,-.03]]
        old=assemble(p,u);new=assemble(p,u,True)
        self.assertLess(old[0,4],old[0,0])
        self.assertGreater(new[0,4],new[0,0])
        np.testing.assert_allclose(new[:,:4],old[:,:4])

    def test_query_advantage_is_update_value(self):
        u=np.array([[.05,-.02,.1]])
        q=assemble([[.2,.8,.4,.6]],u,True)
        np.testing.assert_allclose(q[:,4:]-q[:,:4].max(axis=1,keepdims=True),u)

    def test_negative_update_does_not_force_spending(self):
        q=assemble([[.99,.2,.3,.4]],[[-.02,-.03,-.04]],True)
        self.assertLess(q[:,4:].max(),q[:,:4].max())

    def test_default_preserves_coupled_score(self):
        p=np.array([[.8,.2,.3,.4]]);u=np.array([[.01,.02,.03]])
        np.testing.assert_allclose(assemble(p,u),np.concatenate([7+2*p-1,7+2*p[:,1:]-1+u],axis=1))

    def test_prediction_route_independent_of_all_three_update_slots(self):
        node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='action_route')
        env={'ROUTES':4,'SLOTS':3,'base':SimpleNamespace(route_index=lambda action,slots:action if action<4 else action-3)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),env)
        values=np.array([4.,1.,2.,3.,9.,8.,7.])
        for action in range(4,7):
            self.assertEqual(env['action_route'](action,values,False,True),0)
            self.assertEqual(env['action_route'](action,values,False),action-3)
        for action in range(4):self.assertEqual(env['action_route'](action,values,False,True),action)


if __name__=='__main__':unittest.main()
