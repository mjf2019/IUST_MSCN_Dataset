# Feature tokens and manuscript selector state

New training uses feature_tokens_v1: each scalar column becomes a 64-dimensional token using a shared linear projection plus a learned column identity. Four Transformer encoder layers (8 heads, feedforward 1024) produce one sigmoid weight per feature. The existing stack(raw features, weights) and convolutional encoder remain unchanged. The manuscript does not specify token embedding width or identity encoding; these are explicit implementation choices.

The selector state is [raw x, 1 - softmax(CNN(Context(x)))], using the full class-probability vector both before selection and after a Context update. Dimension is F+C. Action-0 next state remains its original state. State observations use eval mode and no_grad without changing BatchNorm buffers or requiring labels.

Checkpoint metadata records context_architecture and selector_state_version on all three models. Evaluation infers legacy single-token/scalar-state behavior for untagged older checkpoints and rejects mixed architecture/state metadata. The random online ablation also loads the matching Context architecture.

Fresh training in a new output folder is required to obtain the corrected models. Existing weights and replay are not converted. Reward, selection cost, gamma, Context objective, pretraining and optimizer rates are unchanged; this is not a claim that all remaining implementation details match the paper.

Validation: Python syntax; numerical state-vector and dimension checks with NumPy tensor doubles; legacy state behavior; four state call sites; tokenization source checks. PyTorch is unavailable in the execution environment, so no real Transformer forward/backward or model training was run.
