# Architecture

## Data path

1. `audit.py` and `discovery.py` discover robot, camera, action, task, and tactile fields.
2. `windows.py` aligns the latest pre-anchor image, anchor robot state, actual future 300 ms
   action sequence, and asynchronous real tactile samples. No tactile interpolation is used.
3. `vision.py` and `smolvla_condition.py` freeze SmolVLA and cache the multimodal condition
   tokens used by its action-generation path.
4. `tactile_history.py` caches the real valid left/right samples from the previous 200 ms.
5. `model.py` defines the token-level Transformer Envelope Expert.
6. `transformer_train.py` trains left/right Gaussian heads using episode-disjoint splits.
7. `contact_recalibration.py`, `stable_contact_experiment.py`, and `bilateral_average.py`
   perform calibration and stable-contact evaluation without updating model weights.
8. `inference.py` provides offline and Online RL adapters.

## Main model

The default expert receives frozen SmolVLA condition tokens, one robot-state token, nine
executed-action tokens, and optional left/right causal tactile-history tokens. Two learned query
tokens produce independent left/right `mu`, `log_var`, and auxiliary contact logits.

Training uses weighted Gaussian NLL plus SmoothL1 on `mu`. Calibration episodes estimate
one-sided conformal coefficients. Test episodes are held out from training and calibration.

## Online safety boundary

The learned envelope is a soft Online RL cost and is gated only by causal tactile history.
Raw-force hard stop/retract logic must remain independent of this model and active at all times.

