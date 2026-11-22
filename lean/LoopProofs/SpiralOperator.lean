import Mathlib

open scoped ComplexConjugate

namespace LoopProofs

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]

theorem pow_apply_eigenvector (A : H →L[ℂ] H) (v : H) (μ : ℂ) (h : A v = μ • v) (i : ℕ) :
    (A ^ i) v = μ ^ i • v := by
  induction i with
  | zero => simp
  | succ n ih =>
    rw [pow_succ']
    show A ((A ^ n) v) = _
    rw [ih, map_smul, h, smul_smul, pow_succ]

theorem displacement_in_modes {ι : Type*} (s : Finset ι) (A : H →L[ℂ] H) (U : ι → H) (μ w : ι → ℂ)
    (hA : ∀ l ∈ s, A (U l) = μ l • U l) (j : ℕ) :
    ∑ i ∈ Finset.range j, (A ^ i) (∑ l ∈ s, w l • U l) =
      ∑ l ∈ s, (w l * ∑ i ∈ Finset.range j, μ l ^ i) • U l := by
  have h1 : ∀ i, (A ^ i) (∑ l ∈ s, w l • U l) = ∑ l ∈ s, (w l * μ l ^ i) • U l := by
    intro i
    rw [map_sum]
    refine Finset.sum_congr rfl fun l hl => ?_
    rw [map_smul, pow_apply_eigenvector A (U l) (μ l) (hA l hl) i, smul_smul]
  simp_rw [h1]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [← Finset.sum_smul, Finset.mul_sum]

theorem norm_sq_orthonormal_sum {ι : Type*} (s : Finset ι) (U : ι → H) (hU : Orthonormal ℂ U)
    (c : ι → ℂ) :
    ‖∑ l ∈ s, c l • U l‖ ^ 2 = ∑ l ∈ s, Complex.normSq (c l) := by
  rw [@norm_sq_eq_re_inner ℂ, hU.inner_sum c c s, map_sum]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [← Complex.normSq_eq_conj_mul_self]
  exact Complex.ofReal_re _

theorem spiral_equation {ι : Type*} (s : Finset ι) (A : H →L[ℂ] H) (U : ι → H)
    (hU : Orthonormal ℂ U) (μ w : ι → ℂ) (hA : ∀ l ∈ s, A (U l) = μ l • U l)
    (h1 : ∀ l ∈ s, μ l ≠ 1) (j : ℕ) :
    ‖∑ i ∈ Finset.range j, (A ^ i) (∑ l ∈ s, w l • U l)‖ ^ 2 =
      ∑ l ∈ s, Complex.normSq (w l) * Complex.normSq (1 - μ l ^ j) /
        Complex.normSq (1 - μ l) := by
  rw [displacement_in_modes s A U μ w hA j, norm_sq_orthonormal_sum s U hU]
  refine Finset.sum_congr rfl fun l hl => ?_
  have hne : (1 : ℂ) - μ l ≠ 0 := sub_ne_zero.mpr (Ne.symm (h1 l hl))
  have hg : ∑ i ∈ Finset.range j, μ l ^ i = (1 - μ l ^ j) / (1 - μ l) := by
    rw [geom_sum_eq (h1 l hl), div_eq_div_iff (sub_ne_zero.mpr (h1 l hl)) hne]
    ring
  rw [hg, Complex.normSq_mul, Complex.normSq_div]
  ring

end LoopProofs
