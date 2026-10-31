"""Explicit classifier architecture selection; existing defaults are preserved."""


def expected_cnn_architecture(dataset, light_cnn=False):
    if dataset not in ('IAT', 'PS'):
        raise ValueError('Unknown Burst dataset.')
    return 'light_ps_cnn_v1' if dataset == 'PS' or light_cnn else 'legacy'


def validate_cnn_variant(dataset, architecture, light_cnn=False):
    expected = expected_cnn_architecture(dataset, light_cnn)
    if architecture != expected:
        raise ValueError(f'Expected CNN {expected} for {dataset}; checkpoint contains {architecture}. '
                         'Use matching source/teacher models; architecture conversion is not supported.')
    return expected
