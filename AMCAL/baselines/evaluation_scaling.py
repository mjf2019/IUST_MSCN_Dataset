"""Explicit notebook-style preprocessing ablation; no checkpoint mutations."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .adversarial_domains import digest, domain_name


def notebook_scaler(root, artifact, levels):
    # The old notebook fits ONCE on the first requested positive attack level,
    # then calls transform for every subsequent level. Never fit per level.
    positive=[level for level in levels if level>0]
    if not positive:raise ValueError('Notebook test scaler requires an attack level greater than zero.')
    level=positive[0]
    path=Path(root)/domain_name(level)
    frame=pd.read_csv(path)
    expected=artifact['feature_names']+[artifact['label_name']]
    if list(frame.columns)!=expected:raise ValueError('Notebook scaler feature schema mismatch.')
    features=frame.iloc[:,:-1]
    if not np.isfinite(features.to_numpy(dtype=float)).all():raise ValueError('Nonfinite notebook scaler fit input.')
    fitted=StandardScaler().fit(features)
    info=dict(mode='notebook',fit_level=level,fit_rows=len(frame),fit_csv_sha256=digest(path),
              scope='attack TEST inputs only; clean/reference/TRAIN preprocessing unchanged',
              rule='fit once on complete first requested positive attack file; transform subsequent files')
    return {**artifact,'scaler':fitted},info
