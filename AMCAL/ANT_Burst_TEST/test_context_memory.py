"""Context memory tests: standard-library checks + optional real PyTorch checks."""
import random
import unittest
from amcal_burst_context_memory import ContextMemory, make_context_memory, with_context_rehearsal

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None


class MemoryTests(unittest.TestCase):
    def test_fifo_class_caps_and_balance(self):
        m = ContextMemory(32,42)
        for c in range(6):
            for i in range(50):
                m.add((c,i),c,100*c+i)
        self.assertEqual(len(m),192)
        self.assertTrue(all(len(b)==32 for b in m.buckets.values()))
        self.assertEqual(m.buckets[0][0][0],(0,18))
        batch=m.sample_items(16)
        counts=[sum(item[1]==c for item in batch) for c in range(6)]
        self.assertLessEqual(max(counts)-min(counts),1)
        self.assertEqual(len({item[2] for item in batch}),16)

    def test_rng_and_reproducibility(self):
        a,b=ContextMemory(3,0),ContextMemory(3,0)
        for c in range(3):
            for i in range(3):
                a.add((c,i),c);b.add((c,i),c)
        before=random.getstate()
        self.assertEqual(a.sample_items(8),b.sample_items(8))
        self.assertEqual(before,random.getstate())

    def test_disabled_and_empty_paths_are_exact_identity(self):
        loss=object()
        self.assertIsNone(make_context_memory("off"))
        for m in (None,ContextMemory(2,42)):
            combined,info=with_context_rehearsal(loss,m,None,None)
            self.assertIs(combined,loss)
            self.assertEqual(info["replay_samples"],0)
        m=ContextMemory(2);m.add("unused",0)
        combined,info=with_context_rehearsal(loss,m,None,None,weight=0)
        self.assertIs(combined,loss)
        self.assertEqual(m.replayed_samples,0)

    def test_small_memory_has_no_repeated_draws(self):
        m=ContextMemory(2)
        m.add("a",0,1);m.add("b",1,2)
        self.assertEqual(len(m.sample_items(16)),2)


@unittest.skipIf(torch is None,"PyTorch unavailable")
class TorchMemoryTests(unittest.TestCase):
    def test_snapshot_loss_gradient_and_frozen_classifier(self):
        torch.manual_seed(42)
        context=nn.Conv1d(1,1,1)
        cnn=nn.Sequential(nn.Flatten(),nn.Linear(4,2)).eval()
        for p in cnn.parameters():p.requires_grad_(False)
        initial={n:p.detach().clone() for n,p in cnn.named_parameters()}
        m=ContextMemory(2,42)
        x=torch.randn(2,1,4); y=torch.tensor([0,1])
        snapshot=x.clone()
        m.add_batch(x,y,[1,2]); x.add_(100)
        torch.testing.assert_close(m.buckets[0][0][0],snapshot[0])
        fresh=torch.randn(1,1,4)
        current=nn.functional.cross_entropy(cnn(context(fresh)),torch.tensor([1]))
        combined,info=with_context_rehearsal(current,m,context,cnn,16,1)
        self.assertEqual(info["replay_samples"],2)
        self.assertGreater(combined.item(),current.item())
        combined.backward()
        self.assertGreater(context.weight.grad.abs().sum().item(),0)
        for n,p in cnn.named_parameters():
            torch.testing.assert_close(p,initial[n],rtol=0,atol=0)
            self.assertIsNone(p.grad)
        self.assertEqual(len(m),2)  # Sampling/loss never inserts the new query.


if __name__=="__main__":
    unittest.main()
