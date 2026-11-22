import Mathlib

open Complex

namespace LoopProofs

theorem geometric_step_sum (μ : ℂ) (hμ : μ ≠ 1) (j : ℕ) :
    ∑ i ∈ Finset.range j, μ ^ i = (1 - μ ^ j) / (1 - μ) := by
  have h1 : (1 : ℂ) - μ ≠ 0 := sub_ne_zero.mpr (Ne.symm hμ)
  have h2 : μ - 1 ≠ 0 := sub_ne_zero.mpr hμ
  rw [geom_sum_eq hμ, div_eq_div_iff h2 h1]
  ring

theorem mode_modulus (ρ θ : ℝ) (j : ℕ) :
    Complex.normSq (1 - ((ρ : ℂ) * (Complex.cos θ + Complex.sin θ * I)) ^ j) =
      1 - 2 * ρ ^ j * Real.cos (j * θ) + ρ ^ (2 * j) := by
  rw [mul_pow, Complex.cos_add_sin_mul_I_pow]
  have hc : Complex.cos (↑j * ↑θ) = ↑(Real.cos (↑j * θ)) := by push_cast; rfl
  have hs : Complex.sin (↑j * ↑θ) = ↑(Real.sin (↑j * θ)) := by push_cast; rfl
  rw [hc, hs, Complex.normSq_apply]
  simp only [sub_re, sub_im, one_re, one_im, mul_re, mul_im, add_re, add_im, ofReal_re,
    ofReal_im, I_re, I_im, ← ofReal_pow]
  have hsc := Real.sin_sq_add_cos_sq (↑j * θ)
  rw [pow_mul']
  linear_combination (ρ ^ j) ^ 2 * hsc

end LoopProofs
