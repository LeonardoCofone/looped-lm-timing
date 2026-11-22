# figures/

| file | content |
|---|---|
| `make_figures.py` | Draws every figure of the paper from `results/` into `out/` and prints `OK` for each one whose panels, labels and legends fit inside the figure without overlapping. |
| `out/fig_intro.pdf` | Figure 1: loss past `r = 8` by recipe, and the update scaled by 1/2 (Section 1). |
| `out/fig_profile.pdf` | Figure 2: step before `r` for `r = 6..24`, raw and multiplied by `r` (Section 3). |
| `out/fig_brake.pdf` | Figure 3: typical log-stretch `lambda_k` in `k/r` for `r = 6..24`, and with three times the data (Section 4). |
| `out/fig_wave.pdf` | Figure 4: loss and distance from `s_r` past `r`, and the accumulated phase of the step (Section 5). |
| `out/fig_recipe.pdf` | Figure 5: brake and loss past `r` by recipe at `r = 8`, and the public models (Section 7). |
| `out/fig_temporal.pdf` | Figure 6: decoded loop index (Appendix F). |
| `out/fig_normality.pdf` | Figure 7: departure from normality along the loop (Appendix G). |

```
python figures/make_figures.py
```
