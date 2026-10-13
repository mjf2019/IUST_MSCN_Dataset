# Residual affine Context experiment

Select --context-architecture residual_affine_v1 for fresh training. This is a new architecture experiment, not exact reproduction of the manuscript encoder.

Each feature is a token (shared 1-to-64 embedding plus learned column identity). A four-layer Transformer with 8 heads, GELU and pre-norm produces two scalars per feature through a LayerNorm/Linear head. Dropout is zero and there is no BatchNorm. All model inputs/outputs keep shape [B,1,F].

The output is x * (1 + tanh(scale_logit)) + tanh(shift_logit). Scaling lies in (0,2); shifts lie in (-1,1) standardized-feature units. It can attenuate/amplify features and correct offsets. No ReLU or LayerNorm is applied to the output feature vector. The output head starts with zero weights/bias so train/eval start exactly at the identity mapping. At the first backward pass, gradients reach the head; after a head update they also reach the Transformer.

Existing feature_tokens_v1 and legacy architectures/checkpoints remain supported. Evaluation automatically loads the recorded architecture; explicitly requesting a different architecture is rejected in learned evaluation. The random control loads the same recorded Context architecture. No weights/replay are converted.

Loss, RL state/reward, sample ordering, budget, base classifier and optimizer schedules are unchanged. Use a new output directory and retrain CNN/Context/selector for this experiment. Online Context LR is independently set via --online-context-lr.

Local checks: syntax and actual forward-function arithmetic through NumPy tensor doubles, including identity and affine scale/shift. PyTorch is unavailable locally. The accompanying test_context_residual_affine.py runs CPU PyTorch tests for exact initial identity in train/eval and batches 1/4, direct modulation arithmetic, absence of BatchNorm, gradient propagation through a frozen classifier, and unchanged classifier weights.
