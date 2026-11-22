# lean/

The proofs of Appendix C, checked in Lean 4 (`v4.36.0`) with Mathlib (`v4.36.0`). Every statement of the paper has a
theorem here, except the bound of Henrici (1962) in the second part of Remark 1, which is cited. No file contains
`sorry`, `admit` or a new `axiom`.

| file | content |
|---|---|
| `LoopProofs/StepRecursion.lean` | The step recursion with the averaged Jacobian, and Remark 2 (Appendix C). |
| `LoopProofs/NormalStep.lean` | Remark 1, first part: with normal Jacobians of spectral radius at most one, the step never grows. |
| `LoopProofs/Return.lean` | Remark 3 and the energy identity. |
| `LoopProofs/Confinement.lean` | Proposition 1. |
| `LoopProofs/BeyondSupport.lean` | Corollary 2, and the cost of an excursion stated after Proposition 1 (Section 8). |
| `LoopProofs/Euler.lean` | Proposition 3 (a), (b) and (c), with the Taylor step. |
| `LoopProofs/SpiralMode.lean` | Proposition 4 for one mode. |
| `LoopProofs/SpiralOperator.lean` | Proposition 4, the spiral equation summed over the eigenmodes. |
| `LoopProofs/LinearLoop.lean` | Proposition 4 (i) and (ii), and the gain as a function of the state (Remark 2). |
| `LoopProofs/HalfTurn.lean` | Proposition 5. |
| `LoopProofs/ScaledUpdate.lean` | The eigenvalue of the slowed update (Appendix A). |
| `Axioms.lean` | Prints the axioms each main theorem depends on: only `propext`, `Classical.choice` and `Quot.sound`. |

```
lake exe cache get
lake build
lake env lean Axioms.lean
```

The formal statements differ from the text in four places. Remark 1 is stated for operators on a complex Hilbert
space, where normality and the spectral radius are defined. Proposition 4 assumes finitely many orthonormal
eigenvectors, as the spectral theorem gives in finite dimension. Proposition 5 places the fixed point at 1 in the
complex plane; the general case multiplies the loss by the squared norm of the fixed point. Proposition 3 asks the
derivative of the solution to be Lipschitz, which follows from `F` being twice continuously differentiable, and assumes
the bounds on `F` and `G_r` only at the states the loop visits. The first-order argument for the angle between steps,
given in the proof of Proposition 3 outside its statement, is not formalized.
