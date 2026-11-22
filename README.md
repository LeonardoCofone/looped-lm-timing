# Where the Loss Is Applied Shapes the Timing of Looped Language Models

Code, measurements and the test registry for the paper *Where the Loss Is Applied Shapes the Timing of Looped Language Models*.

A looped language model applies one block of layers many times, so it can spend more computation at test time by
running more loops. Models trained with the loss after a fixed number of loops `r` get worse past `r`, while models
trained at a random number of loops do not. In our controls the difference follows which loops carry the loss:

- with the loss only at loop `r` and the gradient through every loop, the model learns a trajectory timed to end at
  `r`, where the loop contracts most strongly;
- run past `r`, the state keeps moving on a slowly shrinking spiral, so the loss rises and, for `r = 6` to `12`, falls
  again;
- scaling every update by `r/r'` moves the end of this trajectory from loop `r` to loop `r'`, up to `2r`, without
  retraining.

With a random depth or with the loss at every loop, none of this happens. These recipes also contract more strongly,
which may contribute.

## Reproduce the paper from the included results

No GPU and no checkpoints are needed. Every number, table, figure and registered test of the paper is computed from
`results/`:

```
pip install -r requirements.txt
python -m analysis.model_table            # the table of the 51 looped models
python -m analysis.recipe_table           # the table of recipes at r = 8
python -m analysis.tests_controls         # registered tests on the controls
python figures/make_figures.py            # the figures, in figures/out/
```

The proofs of the paper are checked in Lean 4, in `lean/`:

```
cd lean
lake exe cache get
lake build
```

`registry/tests.csv` lists the 127 registered tests with the date their threshold was written, the outcome, the
verdict and the script that recomputes it. Run any script in `analysis/` to see its tests.

## Retrain and remeasure

Training and measurement need a GPU (0.5B tokens per model of 114M parameters, 3B for the 472M models).

```
python -m loopedlm.data --out data                          # FineWeb-Edu, BPE tokenizer of 32,768 tokens
python -m loopedlm.train --preset base --sampling fixed --mean_rec 8 --mean_bptt 8 --seed 0
python -m measure.curves --runs runs --data data            # then any other script in measure/
```

`configs/` lists the command of every model in the paper. Each measurement script writes one JSON file per model into
`results/models/<run>/`, the files that `analysis/` reads. A run already measured there is skipped: pass `--redo 1` to
replace the included measurements, or `--out` to write elsewhere.

## What is not included

Checkpoints (about 0.5 GB per model of 114M parameters) and the per-token arrays of most models are not in the
repository. Every checkpoint can be retrained with its command in `configs/`.

## Layout

See [STRUCTURE.md](STRUCTURE.md). Every folder has a `README.md` that describes each of its files.

## License

CC BY-NC-SA 4.0, see [LICENSE](LICENSE).

## Citation

```
@article{losstiming2026,
  title  = {Where the Loss Is Applied Shapes the Timing of Looped Language Models},
  author = {Anonymous},
  year   = {2026}
}
```
