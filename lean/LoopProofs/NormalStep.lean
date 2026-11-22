import Mathlib

open scoped NNReal ENNReal

namespace LoopProofs

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H]

theorem norm_le_one (J : H →L[ℂ] H) [IsStarNormal J] (hρ : spectralRadius ℂ J ≤ 1) :
    ‖J‖ ≤ 1 := by
  rw [IsStarNormal.spectralRadius_eq_nnNorm J] at hρ
  have h : ‖J‖₊ ≤ 1 := by exact_mod_cast hρ
  exact_mod_cast h

theorem step_does_not_grow (J : H →L[ℂ] H) [IsStarNormal J] (hρ : spectralRadius ℂ J ≤ 1)
    (d : H) : ‖J d‖ ≤ ‖d‖ :=
  calc ‖J d‖ ≤ ‖J‖ * ‖d‖ := J.le_opNorm d
    _ ≤ 1 * ‖d‖ := mul_le_mul_of_nonneg_right (norm_le_one J hρ) (norm_nonneg d)
    _ = ‖d‖ := one_mul _

theorem steps_do_not_grow (J : ℕ → H →L[ℂ] H) (hn : ∀ k, IsStarNormal (J k))
    (hρ : ∀ k, spectralRadius ℂ (J k) ≤ 1) (d : ℕ → H)
    (hd : ∀ k, d (k + 1) = J k (d k)) (k : ℕ) : ‖d k‖ ≤ ‖d 0‖ := by
  induction k with
  | zero => exact le_rfl
  | succ n ih =>
    have := hn n
    rw [hd n]
    exact le_trans (step_does_not_grow (J n) (hρ n) (d n)) ih

end LoopProofs
