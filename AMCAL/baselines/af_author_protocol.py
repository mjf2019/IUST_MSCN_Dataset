"""AF single-source author structure, ported to PyTorch for Burst inputs.

Reference: SmartHomePrivacyProject/AdaptiveFingerprinting at
f9d33289f24447fe5edd738c0394d269d4648fb1 (AF/model.py,
one_classifier.py, test.py and tools/utility.py).
This is a documented port, not a bit-identical Keras reproduction. Burst
IAT/PS features and AMCAL partitions differ from packet-direction WF traces.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.neighbors import KNeighborsClassifier

AUTHOR_COMMIT = 'f9d33289f24447fe5edd738c0394d269d4648fb1'
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class AFParameters:
    source_per_class: int = 25
    target_pool_per_class: int = 20
    iterations: int = 1000
    batch_size: int = 64
    embedding_dim: int = 512
    learning_rate: float = 1e-5
    class_weight: float = 4.
    domain_weight: float = 4.
    tuning_epochs: int = 30
    tuning_patience: int = 10
    tuning_lr: float = .001


def per_class_indices(y, classes, count, rng):
    """Fail on undersized classes; never silently duplicate support examples."""
    ids = []
    for label in classes:
        available = np.flatnonzero(np.asarray(y) == label)
        if len(available) < count:
            raise ValueError(f'Class {label}: need {count} rows, found {len(available)}.')
        ids.extend(rng.permutation(available)[:count])
    return np.asarray(ids, dtype=np.int64)


class SamePool(nn.Module):
    """Keras SAME max pooling, including asymmetric padding at short widths."""
    def __init__(self, kernel, stride):
        super().__init__()
        self.kernel, self.stride = kernel, stride

    def forward(self, x):
        length = x.shape[-1]
        pad = max((math.ceil(length / self.stride)-1)*self.stride+self.kernel-length, 0)
        return F.max_pool1d(F.pad(x, (pad//2, pad-pad//2), value=-float('inf')),
                            self.kernel, self.stride)


class KerasBatchNorm(nn.Module):
    """Keras moments also support singleton batches in small N-shot tasks."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer('running_mean', torch.zeros(width))
        self.register_buffer('running_var', torch.ones(width))

    def forward(self, x):
        axes = (0, 2) if x.ndim == 3 else (0,)
        shape = (1, -1, 1) if x.ndim == 3 else (1, -1)
        if self.training:
            mean = x.mean(axes)
            var = x.var(axes, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), .01)
                self.running_var.lerp_(var.detach(), .01)
        else:
            mean, var = self.running_mean, self.running_var
        return ((x-mean.reshape(shape)) / torch.sqrt(var.reshape(shape)+.001)
                * self.weight.reshape(shape) + self.bias.reshape(shape))


def bn(width):
    return KerasBatchNorm(width)


class Extractor(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        blocks = []
        channels = 1
        for index, width in enumerate((32, 64, 128, 256)):
            blocks.extend([nn.Conv1d(channels, width, 8, padding='same'), nn.ELU(),
                           nn.Conv1d(width, width, 8, padding='same'), nn.ELU(),
                           bn(width), SamePool(8, 4)])
            if index < 3:
                blocks.append(nn.Dropout(.1))
            channels = width
        self.blocks = nn.Sequential(*blocks)
        self.features = nn.Linear(256, embedding_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.features(self.blocks(x).mean(-1))


class Head(nn.Module):
    """Author default *_depth=1 convolutional classification/domain head."""
    def __init__(self, outputs):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 128, 4, padding='same'), bn(128),
                                  nn.Softsign(), SamePool(4, 1), nn.Dropout(.4))
        self.dense = nn.Sequential(nn.Linear(128, 512), bn(512), nn.Softsign(),
                                   nn.Dropout(.4), nn.Linear(512, outputs), bn(outputs))

    def forward(self, z):
        return self.dense(self.conv(z.unsqueeze(1)).mean(-1))


class AFNetwork(nn.Module):
    def __init__(self, classes, parameters):
        super().__init__()
        self.extractor = Extractor(parameters.embedding_dim)
        self.classifier = Head(classes)
        self.discriminator = Head(1)
        # Keras Conv/Dense defaults are Glorot uniform and zero bias.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


def tensor(x, device, integer=False):
    return torch.as_tensor(x, dtype=torch.long if integer else torch.float32, device=device)


def author_adversarial_step(model, combined_optimizer, domain_optimizer, sx, sy, tx, p):
    """Two optimizers and flipped labels, as in author one_classifier.train.

    The author restores discriminator weights AND BN buffers after the combined
    step, then restores all other weights/buffers after the discriminator step.
    We exclude those parameters from the corresponding optimizer and restore
    BN state explicitly. No GRL surrogate or target class labels are used.
    """
    device = next(model.parameters()).device
    xs, xt = tensor(sx, device), tensor(tx, device)
    n, m = len(xs), len(xt)
    if min(n, m) < 2:
        raise ValueError('Adversarial minibatches need at least two samples per domain.')
    x = torch.cat((xs, xt))
    domain = torch.cat((torch.zeros(n, device=device), torch.ones(m, device=device)))
    model.train()
    discriminator_state = {k: v.detach().clone() for k, v in model.discriminator.state_dict().items()}
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)
    combined_optimizer.zero_grad()
    z = model.extractor(x)
    logits = model.classifier(z)
    # Keras weights mask target classification but averages over the full batch.
    cls = F.cross_entropy(logits[:n], tensor(sy, device, True), reduction='sum') / (n+m)
    adv = F.binary_cross_entropy_with_logits(model.discriminator(z).flatten(), domain)
    loss = p.class_weight*cls + p.domain_weight*adv
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite AF adversarial loss.')
    loss.backward()
    combined_optimizer.step()
    model.discriminator.load_state_dict(discriminator_state)
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(True)
    # The extractor's BN statistics must not be updated in this second pass.
    extractor_state = {k: v.detach().clone() for k, v in model.extractor.named_buffers()}
    with torch.no_grad():
        z = model.extractor(x)
    for k, value in model.extractor.named_buffers():
        value.copy_(extractor_state[k])
    domain_optimizer.zero_grad()
    domain_loss = F.binary_cross_entropy_with_logits(model.discriminator(z).flatten(), 1-domain)
    if not torch.isfinite(domain_loss):
        raise RuntimeError('Nonfinite AF discriminator loss.')
    domain_loss.backward()
    domain_optimizer.step()
    return float(loss.detach()), float(domain_loss.detach())


def embed(extractor, x, device, batch_size=256):
    extractor.eval()
    with torch.no_grad():
        return np.concatenate([extractor(tensor(x[i:i+batch_size], device)).cpu().numpy()
                               for i in range(0, len(x), batch_size)])


def target_knn(support_z, support_y, shots):
    """Author utility.kNN_train: target-only, cosine, distance weights, k=N."""
    model = KNeighborsClassifier(n_neighbors=shots, weights='distance', metric='cosine', algorithm='brute')
    return model.fit(support_z, support_y)


def fit(source_x, source_y, target_x, target_y, classes, seed, p=AFParameters(), device='cpu'):
    """Fit without a test/validation argument; target labels are used in tuning only."""
    if min(len(source_x), len(target_x)) < 2:
        raise ValueError('AF requires at least two examples in each domain.')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    model = AFNetwork(len(classes), p).to(device)
    combined = torch.optim.Adam(list(model.extractor.parameters())+list(model.classifier.parameters()),
                                lr=p.learning_rate, eps=1e-7)
    domain = torch.optim.Adam(model.discriminator.parameters(), lr=p.learning_rate, eps=1e-7)
    logs = []
    for iteration in range(p.iterations):
        # Official generator draws without replacement. Tiny Burst N-shot tasks
        # may have fewer than 64 total examples: use a smaller domain batch,
        # explicitly recorded, rather than duplicating examples or leaking data.
        size = min(p.batch_size, len(source_x), len(target_x))
        si = rng.choice(len(source_x), size, replace=False)
        ti = rng.choice(len(target_x), size, replace=False)
        losses = author_adversarial_step(model, combined, domain,
                                        source_x[si], source_y[si], target_x[ti], p)
        if iteration == 0 or (iteration+1) % 100 == 0 or iteration+1 == p.iterations:
            logs.append(dict(iteration=iteration+1, combined=losses[0], discriminator=losses[1]))
            print(f'AF adaptation {iteration+1}/{p.iterations} | loss {losses[0]:.4f}', flush=True)
    # Author test.tuning_model uses a NEW linear softmax head on the embedding,
    # Adam defaults, 30 epochs, batch64, stopping on TRAIN accuracy, patience10.
    # test.copy_weights copies feature_model.layers[:-3]: all convolution/BN
    # weights, but NOT the final FeaturesVec Dense. It is freshly initialized
    # before supervised tuning. Both Keras models then share that extractor.
    nn.init.xavier_uniform_(model.extractor.features.weight)
    nn.init.zeros_(model.extractor.features.bias)
    tune = nn.Linear(p.embedding_dim, len(classes)).to(device)
    nn.init.xavier_uniform_(tune.weight)
    nn.init.zeros_(tune.bias)
    optimizer = torch.optim.Adam(list(model.extractor.parameters())+list(tune.parameters()),
                                 lr=p.tuning_lr, eps=1e-7)
    best, stale, tuning = -1., 0, []
    for epoch in range(p.tuning_epochs):
        model.extractor.train()
        order = rng.permutation(len(target_y))
        correct = 0
        for start in range(0, len(order), p.batch_size):
            ids = order[start:start+p.batch_size]
            logits = tune(model.extractor(tensor(target_x[ids], device)))
            loss = F.cross_entropy(logits, tensor(target_y[ids], device, True))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite AF tuning loss.')
            correct += int((logits.argmax(1) == tensor(target_y[ids], device, True)).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = correct / len(target_y)
        tuning.append(dict(epoch=epoch+1, training_accuracy=accuracy))
        if accuracy > best:
            best, stale = accuracy, 0
        else:
            stale += 1
            if stale >= p.tuning_patience:
                break
    model.eval()
    return model, dict(parameters=asdict(p), adaptation=logs, tuning=tuning,
                       actual_adversarial_batch_size=min(p.batch_size, len(source_x), len(target_x)),
                       author_commit=AUTHOR_COMMIT, protocol_version=PROTOCOL_VERSION)
