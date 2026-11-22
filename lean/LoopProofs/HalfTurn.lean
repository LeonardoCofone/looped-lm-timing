import Mathlib

open Complex

namespace LoopProofs

theorem half_turn_loss (α ρ θ : ℝ) (k : ℕ) :
    Complex.normSq ((1 : ℂ) - ((ρ : ℂ) * (Complex.cos θ + Complex.sin θ * I)) ^ k - α) =
      (α - 1) ^ 2 + 2 * (α - 1) * ρ ^ k * Real.cos (k * θ) + ρ ^ (2 * k) := by
  rw [mul_pow, Complex.cos_add_sin_mul_I_pow]
  have hc : Complex.cos (↑k * ↑θ) = ↑(Real.cos (↑k * θ)) := by push_cast; rfl
  have hs : Complex.sin (↑k * ↑θ) = ↑(Real.sin (↑k * θ)) := by push_cast; rfl
  rw [hc, hs, Complex.normSq_apply]
  simp only [sub_re, sub_im, one_re, one_im, mul_re, mul_im, add_re, add_im, ofReal_re,
    ofReal_im, I_re, I_im, ← ofReal_pow]
  have hsc := Real.sin_sq_add_cos_sq (↑k * θ)
  rw [pow_mul']
  linear_combination (ρ ^ k) ^ 2 * hsc

theorem closed_form (μ z : ℂ) (s : ℕ → ℂ) (h0 : s 0 = 0)
    (hs : ∀ k, s (k + 1) = z + μ * (s k - z)) (k : ℕ) : s k = z * (1 - μ ^ k) := by
  induction k with
  | zero => simp [h0]
  | succ n ih =>
    rw [hs n, ih, pow_succ]
    ring

theorem loss_upper_bound (α ρ c : ℝ) (k : ℕ) (hα : 1 < α) (hρ : 0 ≤ ρ) (hc : c ≤ 1) :
    (α - 1) ^ 2 + 2 * (α - 1) * ρ ^ k * c + ρ ^ (2 * k) ≤ (α - 1 + ρ ^ k) ^ 2 := by
  have hpk : 0 ≤ ρ ^ k := pow_nonneg hρ k
  have h2 : ρ ^ (2 * k) = (ρ ^ k) ^ 2 := by rw [pow_mul']
  rw [h2]
  nlinarith [mul_nonneg (mul_nonneg (sub_nonneg.mpr hα.le) hpk) (by linarith : (0 : ℝ) ≤ 1 - c)]

theorem loss_lower_bound (α ρ c : ℝ) (k : ℕ) (hα : 1 < α) (hρ : 0 ≤ ρ) (hc : -1 ≤ c) :
    (α - 1 - ρ ^ k) ^ 2 ≤ (α - 1) ^ 2 + 2 * (α - 1) * ρ ^ k * c + ρ ^ (2 * k) := by
  have hpk : 0 ≤ ρ ^ k := pow_nonneg hρ k
  have h2 : ρ ^ (2 * k) = (ρ ^ k) ^ 2 := by rw [pow_mul']
  rw [h2]
  nlinarith [mul_nonneg (mul_nonneg (sub_nonneg.mpr hα.le) hpk) (by linarith : (0 : ℝ) ≤ c + 1)]

theorem loss_minimal_iff (α ρ c : ℝ) (k : ℕ) (hα : 1 < α) (hρ : 0 < ρ) :
    (α - 1) ^ 2 + 2 * (α - 1) * ρ ^ k * c + ρ ^ (2 * k) = (α - 1 - ρ ^ k) ^ 2 ↔ c = -1 := by
  have hpk : 0 < ρ ^ k := pow_pos hρ k
  have h2 : ρ ^ (2 * k) = (ρ ^ k) ^ 2 := by rw [pow_mul']
  rw [h2]
  have ha : 0 < 2 * (α - 1) * ρ ^ k := by positivity
  constructor
  · intro h
    have : 2 * (α - 1) * ρ ^ k * (c + 1) = 0 := by linarith
    rcases mul_eq_zero.mp this with h0 | h0
    · linarith
    · linarith
  · intro h
    rw [h]
    ring

theorem smallest_half_turn (r : ℕ) (hr : 0 < r) (θ : ℝ) (hθ : 0 < θ)
    (h : Real.cos (r * θ) = -1) : Real.pi / r ≤ θ := by
  have hr' : (0 : ℝ) < r := Nat.cast_pos.mpr hr
  by_contra hlt
  rw [not_le] at hlt
  have h1 : r * θ < Real.pi := by
    have := (lt_div_iff₀ hr').mp hlt
    linarith
  have h0 : 0 ≤ r * θ := by positivity
  have := Real.cos_lt_cos_of_nonneg_of_le_pi h0 le_rfl h1
  rw [Real.cos_pi] at this
  linarith

theorem step_increases_with_angle (ρ θ₁ θ₂ : ℝ) (hρ : 0 < ρ) (h1 : 0 ≤ θ₁) (h2 : θ₂ ≤ Real.pi)
    (h12 : θ₁ < θ₂) :
    1 - 2 * ρ * Real.cos θ₁ + ρ ^ 2 < 1 - 2 * ρ * Real.cos θ₂ + ρ ^ 2 := by
  have := Real.cos_lt_cos_of_nonneg_of_le_pi h1 h2 h12
  nlinarith

theorem cos_at_multiples_of_r (r : ℕ) (hr : 0 < r) (m : ℕ) :
    Real.cos ((2 * m + 1) * r * (Real.pi / r)) = -1 ∧
      Real.cos (2 * m * r * (Real.pi / r)) = 1 := by
  have hr' : (r : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hr.ne'
  have e1 : (2 * (m : ℝ) + 1) * r * (Real.pi / r) = Real.pi + (m : ℕ) * (2 * Real.pi) := by
    field_simp
    ring
  have e2 : 2 * (m : ℝ) * r * (Real.pi / r) = (m : ℕ) * (2 * Real.pi) := by
    field_simp
  constructor
  · rw [e1, Real.cos_add_nat_mul_two_pi, Real.cos_pi]
  · rw [e2, Real.cos_nat_mul_two_pi]

theorem period_two_r (r : ℕ) (hr : 0 < r) (k : ℕ) :
    Real.cos (((k + 2 * r : ℕ) : ℝ) * (Real.pi / r)) = Real.cos ((k : ℝ) * (Real.pi / r)) := by
  have hr' : (r : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hr.ne'
  have e : ((k + 2 * r : ℕ) : ℝ) * (Real.pi / r) = (k : ℝ) * (Real.pi / r) + 2 * Real.pi := by
    push_cast
    field_simp
  rw [e, Real.cos_add_two_pi]

end LoopProofs
