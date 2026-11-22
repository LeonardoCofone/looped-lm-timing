import Mathlib

open scoped RealInnerProductSpace

namespace LoopProofs

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem energy_identity (S d : E) :
    ‖S + d‖ ^ 2 - ‖S‖ ^ 2 = 2 * ⟪S, d⟫ + ‖d‖ ^ 2 := by
  rw [norm_add_sq_real]
  ring

theorem return_iff (S d : E) :
    ‖S + d‖ < ‖S‖ ↔ ⟪S, d⟫ < -(1 / 2) * ‖d‖ ^ 2 := by
  rw [← sq_lt_sq₀ (norm_nonneg _) (norm_nonneg _), norm_add_sq_real]
  constructor <;> intro h <;> linarith

theorem no_return_without_obtuse_steps (d : ℕ → E)
    (h : ∀ i j, i < j → 0 ≤ ⟪d i, d j⟫) (j : ℕ) :
    ‖∑ i ∈ Finset.range j, d i‖ ≤ ‖∑ i ∈ Finset.range (j + 1), d i‖ := by
  have hin : 0 ≤ ⟪∑ i ∈ Finset.range j, d i, d j⟫ := by
    rw [sum_inner]
    exact Finset.sum_nonneg fun i hi => h i j (Finset.mem_range.mp hi)
  rw [Finset.sum_range_succ]
  have hsq : ‖∑ i ∈ Finset.range j, d i‖ ^ 2 ≤ ‖∑ i ∈ Finset.range j, d i + d j‖ ^ 2 := by
    rw [norm_add_sq_real]
    nlinarith [sq_nonneg ‖d j‖]
  exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hsq

end LoopProofs
