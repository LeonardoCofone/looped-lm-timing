# tests/

| file | content |
|---|---|
| `phase_estimator.py` | Runs the phase estimator of `measure/common.py` on a synthetic damped spiral (300 tokens, 64 dimensions, one turn in 13 steps) and prints the phase after one turn against `2 pi`, the check quoted in Appendix A. Seconds. |
| `time_scale_identity.py` | Checks the identity of Proposition 1 on a small random loop in double precision, against finite differences, at steps of 1, 0.3 and 0.05. Seconds. |
| `gauge_identity.py` | Checks on a tiny random loop in double precision that `G - Psi'(1)` from `measure/update_gauge.py` equals the branch term computed directly with forward-mode derivatives, at three step sizes. Seconds. |
| `smoke.py` | Trains two tiny models (fixed and random depth) on random tokens on the CPU, runs nine measurement scripts on them, checks that every output file is written and that `--first` reads other sequences. About ten minutes. |

```
python tests/smoke.py
python -m tests.phase_estimator
python -m tests.time_scale_identity
python -m tests.gauge_identity
```

The smoke test checks that the pipeline runs end to end; it says nothing about the numbers. The numbers of the paper
are checked by running the scripts in `analysis/` on `results/`.
