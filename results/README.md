# results/

The measurements behind the paper, as written by `measure/`. Nothing here is edited by hand.

## models/

One folder per trained model, named as in `loopedlm/config.py` (for example `fixed_r8_b8_s0`, `poisson_r8_b4_muon_s1`,
`fixed_r16_b16_D1500M_s0`). 51 models make up the model table of the paper; the others are the contraction
experiments (the pilot at 0.1B tokens, `_D100M`, and the recipes with the update damped in training, `_us`) and the
scale check (three models of 472M parameters trained on 3B tokens, `_d1792`).

| file | written by | content |
|---|---|---|
| `run_info.json` | `loopedlm.train` | Resolved configuration, number of steps and parameters. |
| `train_log.jsonl`, `eval_log.jsonl` | `loopedlm.train` | Every 20 steps: training loss, learning rate, gradient norm, time, mean depth and the range of the decay. Every 500 steps: validation loss at depths 1, 2, 4, 6, 8, 12, 16, 24 and 32. |
| `curves.json` | `measure.curves` | `per_k[k]`: loss and step statistics after `k` loops; `step_norm_median[k]`: median of `\|s_k - s_(k-1)\|` over the tokens of the first 8 sequences, which is the step `d_(k-1)` of the paper (so `q` uses entries `r` and `2`). |
| `loss.json`, `loss_tokens.npz` | `measure.loss_tokens` | Loss of every token at loops `r` and `r + 1`, in float16, on the 64 validation sequences; 33 models, used for the standard errors in Appendix A. |
| `time_scale.json` | `measure.time_scale` | `table[c][k - 1]`: loss after `k` loops with every update scaled by `c`; 14 models, used in Section 8. |
| `predictions.json` | `measure.predictions` | `loss[k]`, `kl_next[k]` = KL(p_k \|\| p_k+1), `kl_prev[k]` = KL(p_k \|\| p_k-1), `kl_from_r[k]` = KL(p_r \|\| p_k), and for 8 models `kl_pairs[j][k]` = KL(p_j \|\| p_k) and `ece[k]` for loops 1 to 16 (index `k` is loop `k`, entry 0 unused); 14 models, used in Appendix E. Each list stops where the registered runs stopped (`kl_next` at `max(32, 2r + 4)`, `kl_from_r` at `3r + 2`); `measure.predictions` writes longer lists that agree on the overlap. |
| `update_gauge.json` | `measure.update_gauge` | `steps[i]` for the steps `h` = 0.05 and 0.02 of the central differences in `c`: `psi` = loss at `r` with the update scaled by `1 - h`, 1, `1 + h`; `path` = loss at `r` with the weights scaled along `theta(c)` at `1 - h`, `1 + h`; `dpsi` = `Psi_r'(1)`, `G`, `N` = `G - dpsi`, `psi2` = `Psi_r''(1)`; 10 models (the 7 fixed models of T-GAUGE, two Poisson and one with the loss at every loop), used in Section 8 and Appendix E. |
| `final_depth_stats.json` | training code | Loss and step statistics after every loop on the end-of-training evaluation, written by the original training code (not by `loopedlm.train`); used by the test on the slope at `r`. |
| `curves_tokens.npz` | `measure.curves` | Loss and step of every token (8 sequences); kept only for the 15 models of the per-token tests. |
| `lyapunov.json` | `measure.lyapunov` | `lyap[k][i]`: log-stretch of direction `i` at loop `k + 1`. |
| `step_geometry.json` | `measure.step_geometry` | `G`: Gram matrix of the steps; `pre`, `post`: mode energy and accumulated phase before and after `r`; `loss_coda`. |
| `state_distance.json` | `measure.state_distance` | `D[i][j]`: mean distance between the states after `i + 1` and `j + 1` loops, divided by the norm at `r`. |
| `loop_decoder.json`, `loop_decoder_k4.json` | `measure.loop_decoder` | `r2`, decoded loop index `khat_mean` per loop, relative step per loop. |
| `temporal_edit.json` | `measure.temporal_edit` | Loss after edits of the state at loop `r/2`. |
| `non_normality.json`, `non_normality_coarse.json` | `measure.non_normality` | `nu[k]` and the per-probe terms used for bootstrap intervals. |
| `plane_dynamics.json`, `plane_dynamics_tokens.npz` | `measure.plane_dynamics` | `A`: trajectory to `16r`; `B`: Jacobian in the plane at selected loops (per-token 2x2 matrices in the npz); `C`: interventions; `D`: decay and attention heads. |
| `scaled_update.json` | `measure.scaled_update` | Loss of `r' = r, 2r, 4r, 8r` loops with the update scaled by `r/r'`, distances between their states, Richardson extrapolation, split of the step into injection and core. |
| `doubled_update.json` | `measure.doubled_update` | Loss when the update is doubled after `k` trained loops. |
| `downstream.json` | `measure.downstream` | Accuracy per task and loop, trained and with the update scaled by 1/2. |
| `ritz.json` | `measure.ritz` | `k[loop]`: Ritz values `[real, imag]` sorted by modulus and their residuals `res`; first grid only. |
| `step_gain.json` | `measure.step_gain` | `points[k<loop>]`: quantiles of `a` (planar restriction), `b` (full Jacobian along the step), `c` (true gain), precision checks, share of the step in the plane; `all_k`: medians at every loop from `r` to `3r - 1`; `geo_mean_r_3r`; `summary`. Nine fixed-depth models. |
| `robustness.json` | `measure.robustness` | Log-stretch with 16 and 32 directions and step geometry on new sequences. |

Indices: in `curves.json`, `per_k[k]` is the state after `k` loops (`k = 0` is the initial state). In the other files
lists usually start at loop 1, so entry `k - 1` is loop `k`.

## public/

`Ouro-1.4B`, `Ouro-2.6B` (`ouro_dynamics.json`, `ouro_scaled.json`), `parcae-140m`, `parcae-370m`, `parcae-770m`,
`parcae-1.3b` and `huginn-0125` (`public_dynamics.json`), measured on the same validation text retokenised for each
model.

## november/

Outputs of the registered tests of 1 November 2026 that `measure/` does not reproduce, read by
`analysis/tests_november.py`. They were measured with the same checkpoints and the same 16 sequences of 1024 tokens
(32 of 512 for `telescoping.json`, 8 of 512 for Ouro), by code written for the registration and not included here, except `ensembles.json`, whose loop entries
`measure/ensembles.py` reproduces.
The controls use GPTs of 12 and 16 layers at width 640, trained on the same data, which are not among the models of
this repository. The keys are short names of the models (`keys.json`: `r8s0` is `fixed_r8_b8_s0`, `pb32` is
`poisson_r8_b32_s0`, `g12` and `g16` the GPTs).

| file | content |
|---|---|
| `ensembles.json` | `self_<model>[j]`: the prediction at `r` against the one at `r + j` of the same model; `<a>\|<b>`: the predictions at `r` of two models. `la`, `lq`: losses; `kl`; `R` = (lq - la) / kl; `mix`: loss of the mixed logits for weights 0.05 to 0.5; `G`: gain of the best mix. |
| `span.json` | Same pairs, with the final features: `R2` of a ridge map from `h_r` (fitted on 64 other sequences), `res_lin` and `res_xi`, the parts of the residual term carried by the linear map and by what it misses, and `id_check`, the error of the identity. For the GPTs, `skip-<n>`: the state after layer `L - n` passed through the last two layers and the readout; `lens-<n>`: the same state read out directly. |
| `telescoping.json` | For loops and GPTs, `stat`: mean and ratio (absolute sum over tokens over sum of absolute values) of the derivative of the loss in the scale of each loop step (`Qp<i>`) or GPT branch (`s<j>`; `c<j>` with the gradient taken at the output, the direct term); `Psi`: loss with every update or branch scaled by `c`; `cross_<a>\|<b>`: the same quantity between two models. |
| `gpt_substeps.json` | For the GPTs, `loss[n]` with every branch split into `n` substeps; for the loops, `flow`: loss with the update scaled by `1/8` at times 0.5 to 3 in units of `r`. |
| `shorter_longer.json` | For 8 fixed models, 16 batches of 4 sequences of 256 tokens: `L` (loss at `r`), `Gd` (loss after `r/2` loops with the update doubled, minus `L`), `Gu` (after `2r` loops with it halved, minus `L`). |
| `jacobian_negative.json` | Ritz values of the one-loop Jacobian (Arnoldi, 80 Jacobian-vector products in float64) at loops `r/2`, `r - 1`, `r`, for 3 sequences of 256 tokens: `s<seq>_k<loop>` with `iso` (isolated negative real values), `neg_real`, `nconv`, `top`. |
| `ouro_1.4b.json`, `ouro_2.6b.json`, `ouro_cross.json` | Ouro: `table[c][k - 1]`, loss after `k` loops with the update scaled by `c`; `pairs["4\|k"]`: ratio and gain of the prediction at loop 4 against loop `k`; between the two models in `ouro_cross.json`; `check_vs_forward`: largest gap to the official forward. |

## field_alignment/

`field_alignment_base.json` (depths 8, 12, 16) and `field_alignment_grid.json` (depths 6 to 24, two seeds): the
alignment of the rescaled steps of two models, from `measure.field_alignment`.
