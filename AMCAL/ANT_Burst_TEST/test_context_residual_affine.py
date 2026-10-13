"""CPU checks for residual Context. Run from repository root."""
import unittest
import torch
import torch.nn as nn
import amcal_burst_legacy as runner


class ResidualContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.previous = runner.CONTEXT_ARCHITECTURE
        runner.CONTEXT_ARCHITECTURE = "residual_affine_v1"
        self.model = runner.ContextAwareNetwork(8, num_layers=1, dim_feedforward=64).cpu()

    def tearDown(self):
        runner.CONTEXT_ARCHITECTURE = self.previous

    def test_initial_identity_in_train_and_eval(self):
        for batch in (1, 4):
            x = torch.randn(batch, 1, 8)
            for training in (True, False):
                self.model.train(training)
                torch.testing.assert_close(self.model(x), x, rtol=0, atol=0)

    def test_direct_scale_shift_and_output_range(self):
        self.model.eval()
        with torch.no_grad():
            self.model.modulation_head[-1].bias.copy_(torch.tensor([0.4, -0.3]))
        x = torch.linspace(-3, 3, 8).reshape(1, 1, 8)
        expected = x * (1 + torch.tanh(torch.tensor(0.4))) + torch.tanh(torch.tensor(-0.3))
        torch.testing.assert_close(self.model(x), expected)
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                             for m in self.model.modules()))

    def test_gradient_through_frozen_classifier(self):
        self.model.train()
        classifier = nn.Sequential(nn.Flatten(), nn.Linear(8, 3)).eval()
        for p in classifier.parameters():
            p.requires_grad_(False)
        before = {n: p.detach().clone() for n, p in classifier.named_parameters()}
        x, y = torch.randn(4, 1, 8), torch.tensor([0, 1, 2, 0])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss = nn.CrossEntropyLoss()(classifier(self.model(x)), y)
        loss.backward()
        self.assertGreater(self.model.modulation_head[-1].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(classifier(self.model(x)), y).backward()
        # Zero head initialization blocks upstream gradients only on the first step.
        self.assertGreater(self.model.feature_projection.weight.grad.abs().sum().item(), 0)
        for n, p in classifier.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
