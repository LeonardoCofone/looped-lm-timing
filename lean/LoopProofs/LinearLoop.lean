import Mathlib

namespace LoopProofs

theorem real_mode_nondecreasing (ρ : ℝ) (h0 : 0 ≤ ρ) (h1 : ρ < 1) (j : ℕ) :
    (1 - ρ ^ j) ^ 2 ≤ (1 - ρ ^ (j + 1)) ^ 2 := by
  have a : ρ ^ (j + 1) ≤ ρ ^ j := pow_le_pow_of_le_one h0 h1.le (Nat.le_succ j)
  have b : ρ ^ j ≤ 1 := pow_le_one₀ h0 h1.le
  have c : 0 ≤ ρ ^ (j + 1) := pow_nonneg h0 _
  nlinarith

theorem real_spectrum_no_return {ι : Type*} (s : Finset ι) (c ρ : ι → ℝ)
    (hc : ∀ l ∈ s, 0 ≤ c l) (h0 : ∀ l ∈ s, 0 ≤ ρ l) (h1 : ∀ l ∈ s, ρ l < 1) (j : ℕ) :
    ∑ l ∈ s, c l * (1 - ρ l ^ j) ^ 2 ≤ ∑ l ∈ s, c l * (1 - ρ l ^ (j + 1)) ^ 2 :=
  Finset.sum_le_sum fun l hl =>
    mul_le_mul_of_nonneg_left (real_mode_nondecreasing (ρ l) (h0 l hl) (h1 l hl) j) (hc l hl)

theorem complex_mode_decomposition (ρ θ : ℝ) (j : ℕ) :
    1 - 2 * ρ ^ j * Real.cos (j * θ) + ρ ^ (2 * j) =
      (1 - ρ ^ j) ^ 2 + 2 * ρ ^ j * (1 - Real.cos (j * θ)) := by
  rw [pow_mul']
  ring

theorem complex_mode_envelopes (ρ θ : ℝ) (h0 : 0 ≤ ρ) (j : ℕ) :
    (1 - ρ ^ j) ^ 2 ≤ 1 - 2 * ρ ^ j * Real.cos (j * θ) + ρ ^ (2 * j) ∧
      1 - 2 * ρ ^ j * Real.cos (j * θ) + ρ ^ (2 * j) ≤ (1 + ρ ^ j) ^ 2 := by
  rw [complex_mode_decomposition]
  have hp : 0 ≤ ρ ^ j := pow_nonneg h0 j
  have hc1 := Real.cos_le_one (j * θ)
  have hc2 := Real.neg_one_le_cos (j * θ)
  constructor <;> nlinarith

theorem lower_envelope_iff (ρ θ : ℝ) (h0 : 0 < ρ) (j : ℕ) :
    1 - 2 * ρ ^ j * Real.cos (j * θ) + ρ ^ (2 * j) = (1 - ρ ^ j) ^ 2 ↔
      Real.cos (j * θ) = 1 := by
  rw [complex_mode_decomposition]
  have hp : 0 < ρ ^ j := pow_pos h0 j
  constructor
  · intro h
    have : 2 * ρ ^ j * (1 - Real.cos (j * θ)) = 0 := by linarith
    rcases mul_eq_zero.mp this with h' | h'
    · linarith
    · linarith
  · intro h
    rw [h]
    ring

theorem mode_period (θ : ℝ) (hθ : θ ≠ 0) (x : ℝ) :
    Real.cos ((x + 2 * Real.pi / θ) * θ) = Real.cos (x * θ) := by
  have e : (x + 2 * Real.pi / θ) * θ = x * θ + 2 * Real.pi := by
    field_simp
  rw [e, Real.cos_add_two_pi]

theorem gain_is_function_of_state {E : Type*} [NormedAddCommGroup E] (Φ : E → E)
    (s : ℕ → E) (hs : ∀ k, s (k + 1) = Φ (s k)) (k : ℕ) :
    ‖s (k + 2) - s (k + 1)‖ / ‖s (k + 1) - s k‖ =
      ‖Φ (Φ (s k)) - Φ (s k)‖ / ‖Φ (s k) - s k‖ := by
  rw [hs (k + 1), hs k]

end LoopProofs
