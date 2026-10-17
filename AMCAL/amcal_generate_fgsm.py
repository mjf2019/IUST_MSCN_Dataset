"""Generate feature-space FGSM on the exact saved held-out test partition."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import amcal_shared_protocol as protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint-output', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1 or any(not np.isfinite(e) or e < 0 for e in args.epsilons):
        parser.error('Batch size must be positive and epsilons finite and nonnegative.')
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        parser.error('Choose an empty output directory; existing attacks are never overwritten.')
    models = Path(args.checkpoint_output) / 'Models'
    artifact_path = models / 'protocol_preprocessing.joblib'
    artifact = joblib.load(artifact_path)
    if artifact.get('protocol_version') != 3:
        raise ValueError('Expected protocol v3 preprocessing.')
    if protocol.file_sha256(args.source) != artifact['source_sha256']:
        raise ValueError('Source hash differs from training source.')
    source = pd.read_csv(args.source)
    if list(source.columns) != artifact['feature_names'] + [artifact['label_name']]:
        raise ValueError('Source schema differs from training schema.')
    ids = np.asarray(artifact['test_ids'], dtype=np.int64)
    if set(ids) & (set(artifact['train_ids']) | set(artifact['val_ids'])):
        raise ValueError('Saved partitions overlap.')
    frame = source.iloc[ids].copy()
    scaled = artifact['scaler'].transform(frame.iloc[:, :-1].to_numpy())
    labels = artifact['label_encoder'].transform(frame.iloc[:, -1].to_numpy())
    saved = torch.load(models / 'initial_cnn_model.pth', map_location=protocol.device, weights_only=True)
    if saved.get('preprocessing_sha256') != protocol.file_sha256(artifact_path):
        raise ValueError('CNN/preprocessing mismatch.')
    cnn = protocol.CNN1D(scaled.shape[1], len(artifact['label_encoder'].classes_)).to(protocol.device)
    cnn.load_state_dict(saved['model_state_dict'])
    cnn.eval()
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)
    signs = []
    for start in range(0, len(scaled), args.batch_size):
        x = torch.tensor(scaled[start:start+args.batch_size], dtype=torch.float32,
                         device=protocol.device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(labels[start:start+args.batch_size], dtype=torch.long, device=protocol.device)
        loss = F.cross_entropy(cnn(x), y)
        grad, = torch.autograd.grad(loss, x)
        signs.append(grad.sign().squeeze(1).detach().cpu().numpy())
    direction = np.concatenate(signs)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'source_row': ids}).to_csv(destination / 'source_rows.csv', index=False)
    for epsilon in args.epsilons:
        perturbed = scaled + epsilon * direction
        raw = artifact['scaler'].inverse_transform(perturbed)
        attack = pd.DataFrame(raw, columns=artifact['feature_names'])
        attack[artifact['label_name']] = frame.iloc[:, -1].to_numpy()
        path = destination / f'FGSM_eps_{epsilon:g}.csv'
        attack.to_csv(path, index=False)
        print('Saved:', path, 'rows:', len(attack))
    manifest = dict(source_sha256=artifact['source_sha256'],
        preprocessing_sha256=protocol.file_sha256(artifact_path),
        cnn_sha256=protocol.file_sha256(models / 'initial_cnn_model.pth'),
        rows=len(ids), partition='saved held-out test IDs only',
        method='untargeted FGSM against frozen base CNN, true-label cross entropy',
        epsilon_space='training StandardScaler feature units; L-infinity',
        clipping='none; feature-space robustness diagnostic, not packet-realizability claim',
        original_files_modified=False, epsilons=args.epsilons)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Row map:', destination / 'source_rows.csv')


if __name__ == '__main__':
    main()
