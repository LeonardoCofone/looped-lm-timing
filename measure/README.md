# measure/

Scripts that load a trained model and write what the paper measures. Each script for our models runs over every trained
run in `--runs` (or those whose name contains `--only`), skips the runs it has already measured unless `--redo 1`, and
writes one file per run into `--out` (default `results/models/<run>/`). Some scripts measure only the runs they apply to
(for example `r >= 4` for `plane_dynamics.py`). They take their sequences from the same pool of 64 validation sequences (`robustness.py`
draws a new pool, `downstream.py` reads the task file) and start from the same initial state (seed 7), so measurements of different
scripts refer to the same trajectory. `--first k` takes the sequences from position `k` of the pool instead of 0; with
a separate `--out`, it repeats a measurement on disjoint sequences (Appendix A, sequence batches). `field_alignment.py` and the `public_*` scripts take their models explicitly.

```
python -m measure.lyapunov --runs runs --data data
```

| file | writes | what it measures |
|---|---|---|
| `common.py` | | Shared pieces: argument parser, validation sequences, Jacobian-vector products by central differences, the finite-time Lyapunov spectrum, the Hutchinson estimate of non-normality, the energy and accumulated phase of the steps in their leading plane. |
| `curves.py` | `curves.json`, `curves_tokens.npz` | Loss, accuracy, entropy, state norm and step norm at every loop up to `max(8r, 64)` on 32 sequences (bfloat16 on a GPU). The median step per loop gives the contraction rate `q`. |
| `loss_tokens.py` | `loss.json`, `loss_tokens.npz` | Loss of every token at loops `r` and `r + 1` on the 64 sequences, in float16, for the standard errors over sequences. |
| `lyapunov.py` | `lyapunov.json` | Local log-stretch of 16 tangent directions at every loop up to `max(2r, r + 8)`; its mean over directions is `lambda_k`, whose minimum is the brake. |
| `step_geometry.py` | `step_geometry.json` | Gram matrix of the steps up to `4r`, energy of the two leading modes and accumulated phase of the step in their plane, before and after `r`. |
| `state_distance.py` | `state_distance.json` | Distances between all states up to `8r` (at most 128 loops) and the loss read with and without the coda. |
| `loop_decoder.py` | `loop_decoder.json`, `loop_decoder_k4.json` | A ridge decoder of the loop index from the normalised state, fitted on loops 1 to `r` (or 1 to `--fitK`) on half of the sequences and read up to `4r` on the other half. |
| `temporal_edit.py` | `temporal_edit.json` | Moves the state at loop `r/2` along the decoder direction or along random directions of the same size, and records where the loss lands. |
| `non_normality.py` | `non_normality.json`, `non_normality_coarse.json` | Departure from normality `||J J^T - J^T J|| / ||J^T J||` of the Jacobian of one loop, at every loop (`--dense 1`) or at five loops. |
| `plane_dynamics.py` | `plane_dynamics.json`, `plane_dynamics_tokens.npz` | Trajectories to `max(16r, 128)` on 8 sequences; the 2x2 Jacobian in the plane of the motion, in full and with the attention output, the MLP output, the attention pattern or the normalisation held at their values; the update rescaled from the start or after `r`; the same paths frozen after `r`; the decay and the attention heads of the core. |
| `time_scale.py` | `time_scale.json` | Loss after every loop up to `max(32, 2r)` with every update scaled by `c` from 0.5 to 3, on 16 sequences of 1024 tokens taken every 10^6 tokens of the validation set (the sequences of the registered test, not the pool of 64). |
| `update_gauge.py` | `update_gauge.json` | On the same 16 sequences, central differences at `c = 1` of the loss at `r` along two paths: the update scaled by `c` (`dpsi`, the `Psi_r'(1)` of Proposition 1) and the weights scaled so that the linear part of the update scales exactly by `c` (decay `a -> 1 - c(1 - a)`, injection and output projections of the core times `c`; `G`). `N = G - dpsi` is the share of the nonlinear branches; `psi2` is the curvature in `c`. |
| `ensembles.py` | `results/november/ensembles.json` | The prediction at `r` against those at `r + j` of the same model (`--self`) and against the prediction at `r` of another model (`--pairs a\|b`): losses, KL, the ratio `R` and the gain `G` of the best mix of the logits. Same 16 sequences. Reproduces the loop entries of the registered file exactly; the pairs with GPTs need the GPT checkpoints, which are not in this repository. |
| `predictions.py` | `predictions.json` | On the same 16 sequences: loss after every loop, KL between the predictions of consecutive loops in both directions, KL from the prediction at `r` to every loop, and, for the first 16 loops, KL between every pair and the calibration error over 15 bins (these two on predictions stored in float16). |
| `scaled_update.py` | `scaled_update.json` | `r'` loops with the update scaled by `r/r'` for `r' = r, 2r, 4r, 8r`, Richardson extrapolation, and the split of the step into injection and core. |
| `doubled_update.py` | `doubled_update.json` | The update doubled after the first `k` trained loops (fixed-depth models with `r >= 8`). |
| `field_alignment.py` | `results/field_alignment/` | Aligns the states of two models with an affine map and compares their rescaled steps at the same rescaled time. |
| `prepare_downstream.py` | `data/downstream_tasks.jsonl` | Samples 1000 examples each of LAMBADA, HellaSwag, ARC-Easy and PIQA. |
| `downstream.py` | `downstream.json` | Zero-shot accuracy of every loop, for the trained loop and the update scaled by 1/2. |
| `ritz.py` | `ritz.json` | Ritz values of the Jacobian of one loop at loops `r/2`, `r`, `2r` and `4r` from 48 Arnoldi steps on 2 sequences of 256 tokens, with the residual of each value. |
| `step_gain.py` | `step_gain.json` | After `r`, in float64 on 2 sequences of 256 tokens: the true gain of the step `|d_{k+1}|/|d_k|`, the full Jacobian applied to the step and its restriction to the plane of the steps in `[r, 4r)`, at loops `r`, `3r/2`, `2r`, `3r` (with a precision check: the Jacobian-vector product with a step of 1e-3 |d_k| against 1e-4 |d_k|, median relative difference at most 1e-3) and at every loop from `r` to `3r - 1` (fixed-depth models trained through every loop with `r >= 8`; the results include nine of them). |
| `robustness.py` | `robustness.json` | The brake with 16 and 32 directions and the step geometry on new validation sequences. |
| `public_random_depth.py` | `results/public/<model>/public_dynamics.json` | The same measures on the released Parcae and Huginn models (random depth). Needs their weights and, for Parcae, its source repository. |
| `public_ouro.py` | `results/public/<model>/ouro_dynamics.json` | The same measures on Ouro 1.4B and 2.6B. Needs `transformers` 4.5x (tested with 4.56.2). |
| `public_ouro_scaled.py` | `results/public/<model>/ouro_scaled.json` | Ouro with its update scaled by `4/r'`. |

`lyapunov.py`, `non_normality.py`, `ritz.py`, `robustness.py` and the plane Jacobians of `plane_dynamics.py` run in float64. Only
`curves.py` (like training) enables TF32 matrix products on a GPU; the other scripts keep full float32, as for the
numbers of the paper. On one GPU the full set of measurements of one model takes 10 to 20 minutes; `plane_dynamics.py`
and `downstream.py` are the slowest.
