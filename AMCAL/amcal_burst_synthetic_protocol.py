"""Clean-only CNN; three-action RL on resampled synthetic shifts, never Burst attacks.

Train with this entry point; evaluate with amcal_burst_official_protocol.py.
Synthetic shifts model numeric distribution changes, not validated packet attacks.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import amcal_burst_official_protocol as official

core = official.core


def shift_magnitudes(raw, rng, strength, noise_ratio, limits):
    """Preserve direction and exact padding zeros; no labels or model gradients."""
    magnitude = np.abs(raw)
    active = magnitude > 0
    # A coherent feature-wise scale change plus smaller row-wise variation.
    gain = np.exp(rng.uniform(-strength, strength, size=(1, raw.shape[1])))
    noise = rng.normal(0, strength * noise_ratio, size=raw.shape)
    shifted = magnitude * gain * np.exp(np.clip(noise, -2, 2))
    shifted = np.minimum(shifted, limits)
    shifted = np.where(active, np.maximum(shifted, np.finfo(np.float32).tiny), 0)
    return (np.sign(raw) * shifted).astype(np.float32)


class SyntheticDomains:
    def __init__(self, features, raw, scaler, limits, seed, max_strength, noise_ratio, resample):
        self.features = features.detach()
        self.scaler, self.limits = scaler, limits
        self.raw = np.asarray(raw, dtype=np.float64).copy()
        self.seed, self.max_strength, self.noise_ratio = seed, max_strength, noise_ratio
        self.resample, self.counter, self.cache = resample, 0, {}

    def __getitem__(self, key):
        if key == 0:
            return self.features
        if key not in (1, 2):
            raise ValueError('Synthetic domain IDs are 0 (clean), 1 (mild), 2 (strong).')
        if not self.resample and key in self.cache:
            return self.cache[key]
        serial = self.counter if self.resample else 0
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(key), serial]))
        self.counter += 1
        lo, hi = ((0.05, 0.4) if key == 1 else (0.4, 1.0))
        strength = float(rng.uniform(lo, hi) * self.max_strength)
        raw = shift_magnitudes(self.raw, rng, strength, self.noise_ratio, self.limits)
        scaled = self.scaler.transform(raw).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError('Synthetic shift produced nonfinite features.')
        value = torch.as_tensor(scaled, device=self.features.device).reshape_as(self.features)
        if not self.resample:
            self.cache[key] = value
        print(f'  Synthetic {"train" if self.resample else "validation"} domain {key}: strength={strength:.4f}')
        return value


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--synthetic-max-strength', type=float, default=0.7)
    parser.add_argument('--synthetic-noise-ratio', type=float, default=0.25)
    extra, remaining = parser.parse_known_args()
    if 'train' not in remaining:
        parser.error('Training only; use amcal_burst_official_protocol.py evaluate afterwards.')
    if '--official-split-dir' not in remaining:
        parser.error('--official-split-dir is required.')
    if '--original-split-dir' in remaining or '--train-perturbations' in remaining:
        parser.error('Use official clean data; omit --original-split-dir and --train-perturbations.')
    if not 0 < extra.synthetic_max_strength <= 2 or not 0 <= extra.synthetic_noise_ratio <= 1:
        parser.error('Strength must be in (0,2], noise ratio in [0,1].')

    def train_synthetic(dataset_path, base_epoch=100, base_patience=15,
                        adver_epoch=100, adver_patience=15, pretrain_epochs=20):
        train, val, _test, full, nclasses, width, counts = core.load_data(dataset_path=dataset_path)
        artifact = joblib.load('Models/protocol_preprocessing.joblib')
        scaler = artifact['scaler']
        # Bounds are fitted exclusively to unperturbed training rows.
        source = pd.read_csv(Path(dataset_path) / core.SOURCE_FILENAME)
        raw_train = source.iloc[artifact['train_ids'], :-1].to_numpy(dtype=np.float64)
        limits = np.max(np.abs(raw_train), axis=0, keepdims=True)
        manifest = dict(protocol='synthetic magnitude shifts v1; clean-only frozen CNN',
                        real_attack_levels_used=[], synthetic_domain_ids=[0, 1, 2],
                        max_strength=extra.synthetic_max_strength, noise_ratio=extra.synthetic_noise_ratio,
                        bounds='per-feature maximum absolute clean training value',
                        preserves='sign and zero padding; labels unchanged by construction, semantic validity not certified',
                        validation='independent fixed synthetic shifts; clean validation IDs only',
                        heldout_scoring_during_training=False, seed=core.RUN_SEED)
        Path('synthetic_training.json').write_text(json.dumps(manifest, indent=2))
        calls = 0

        def build_domains(cnn, features, labels):
            nonlocal calls
            is_train = calls == 0
            calls += 1
            split = 'train_ids' if is_train else 'val_ids'
            if calls > 2:
                raise ValueError('Unexpected extra domain construction.')
            if len(features) != len(artifact[split]):
                raise ValueError('Synthetic domain rows differ from official partition.')
            report = Path('burst_training_domains.json')
            data = json.loads(report.read_text()) if report.exists() else {}
            data[split] = dict(rows=len(features), protocol=manifest['protocol'], domains='synthetic IDs 0/1/2, not Burst levels')
            report.write_text(json.dumps(data, indent=2))
            return SyntheticDomains(features, source.iloc[artifact[split], :-1].to_numpy(dtype=np.float64), scaler, limits, core.RUN_SEED + (710001 if is_train else 820001),
                                    extra.synthetic_max_strength, extra.synthetic_noise_ratio, is_train)

        core.build_training_domains = build_domains
        print('CNN: clean training only. Context pretraining: clean training only. RL: clean and resampled synthetic shifts.')
        cnn = core.CNN1D(width, nclasses).to(core.device)
        cnn = core.train_initial_cnn(cnn, train, val, counts, base_epoch, base_patience)
        torch.save(dict(preprocessing_sha256=core.file_sha256('Models/protocol_preprocessing.joblib'),
                        protocol_version=3, model_state_dict=cnn.state_dict(), input_length=width,
                        num_classes=nclasses), 'Models/initial_cnn_model.pth')
        agent = core.DQNSelectorAgent(width + 2*nclasses + 2 + (4 if core.ROUTE_HISTORY_ENABLED else 0), core.device)
        context = core.ContextAwareNetwork(width).to(core.device)
        core.train_adversarial_system(cnn, agent, context, full, val, counts, width, nclasses,
                                      adver_epoch, adver_patience, pretrain_epochs)
        path = Path('Models/adversarial_best_dqn.pth')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint['training_perturbations'] = []
        checkpoint['training_shift_protocol'] = manifest['protocol']
        checkpoint['synthetic_training'] = manifest
        torch.save(checkpoint, path)
        print('Synthetic-shift training complete; no real attack CSV or heldout test was used.')

    core.train_main = train_synthetic
    sys.argv = [sys.argv[0]] + remaining + ['--train-perturbations', '0', '1', '2']
    official.main()


if __name__ == '__main__':
    main()
