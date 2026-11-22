import Mathlib

namespace LoopProofs

variable {ι E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

theorem eps_nonneg (s : Finset ι) (p ℓ : ι → ℝ) (ℓs : ℝ)
    (hp : ∀ k ∈ s, 0 ≤ p k) (hℓ : ∀ k ∈ s, ℓs ≤ ℓ k) :
    0 ≤ ∑ k ∈ s, p k * (ℓ k - ℓs) :=
  Finset.sum_nonneg fun k hk => mul_nonneg (hp k hk) (sub_nonneg.mpr (hℓ k hk))

theorem confinement_sq (s : Finset ι) (p ℓ : ι → ℝ) (ℓs κ ε : ℝ) (P : E →L[ℝ] F)
    (x : ι → E) (xh : E)
    (hp : ∀ k ∈ s, 0 ≤ p k) (hℓ : ∀ k ∈ s, ℓs ≤ ℓ k)
    (hε : ∑ k ∈ s, p k * (ℓ k - ℓs) = ε) (hκ : 0 < κ)
    (k : ι) (hk : k ∈ s) (hpk : 0 < p k)
    (hq : κ / 2 * ‖P (x k - xh)‖ ^ 2 ≤ ℓ k - ℓs) :
    ‖P (x k - xh)‖ ^ 2 ≤ 2 * ε / (κ * p k) := by
  have h1 : p k * (ℓ k - ℓs) ≤ ε := by
    rw [← hε]
    exact Finset.single_le_sum
      (fun i hi => mul_nonneg (hp i hi) (sub_nonneg.mpr (hℓ i hi))) hk
  have h2 : p k * (κ / 2 * ‖P (x k - xh)‖ ^ 2) ≤ ε :=
    le_trans (mul_le_mul_of_nonneg_left hq hpk.le) h1
  rw [le_div_iff₀ (mul_pos hκ hpk)]
  nlinarith

theorem confinement (s : Finset ι) (p ℓ : ι → ℝ) (ℓs κ ε : ℝ) (P : E →L[ℝ] F)
    (x : ι → E) (xh : E)
    (hp : ∀ k ∈ s, 0 ≤ p k) (hℓ : ∀ k ∈ s, ℓs ≤ ℓ k)
    (hε : ∑ k ∈ s, p k * (ℓ k - ℓs) = ε) (hκ : 0 < κ)
    (k : ι) (hk : k ∈ s) (hpk : 0 < p k)
    (hq : κ / 2 * ‖P (x k - xh)‖ ^ 2 ≤ ℓ k - ℓs) :
    ‖P (x k - xh)‖ ≤ Real.sqrt (2 * ε / κ) * (1 / Real.sqrt (p k)) := by
  have h := confinement_sq s p ℓ ℓs κ ε P x xh hp hℓ hε hκ k hk hpk hq
  have hε0 : 0 ≤ ε := hε ▸ eps_nonneg s p ℓ ℓs hp hℓ
  have hsq := Real.abs_le_sqrt h
  rw [abs_of_nonneg (norm_nonneg _)] at hsq
  have heq : 2 * ε / (κ * p k) = 2 * ε / κ / p k := by rw [div_div]
  rw [heq, Real.sqrt_div' _ hpk.le] at hsq
  calc ‖P (x k - xh)‖ ≤ Real.sqrt (2 * ε / κ) / Real.sqrt (p k) := hsq
    _ = Real.sqrt (2 * ε / κ) * (1 / Real.sqrt (p k)) := by ring

theorem confinement_pairs (s : Finset ι) (p ℓ : ι → ℝ) (ℓs κ ε : ℝ) (P : E →L[ℝ] F)
    (x : ι → E) (xh : E)
    (hp : ∀ k ∈ s, 0 ≤ p k) (hℓ : ∀ k ∈ s, ℓs ≤ ℓ k)
    (hε : ∑ k ∈ s, p k * (ℓ k - ℓs) = ε) (hκ : 0 < κ)
    (j k : ι) (hj : j ∈ s) (hk : k ∈ s) (hpj : 0 < p j) (hpk : 0 < p k)
    (hqj : κ / 2 * ‖P (x j - xh)‖ ^ 2 ≤ ℓ j - ℓs)
    (hqk : κ / 2 * ‖P (x k - xh)‖ ^ 2 ≤ ℓ k - ℓs) :
    ‖P (x j - x k)‖ ≤
      Real.sqrt (2 * ε / κ) * (1 / Real.sqrt (p j) + 1 / Real.sqrt (p k)) := by
  have aj := confinement s p ℓ ℓs κ ε P x xh hp hℓ hε hκ j hj hpj hqj
  have ak := confinement s p ℓ ℓs κ ε P x xh hp hℓ hε hκ k hk hpk hqk
  have hsplit : P (x j - x k) = P (x j - xh) - P (x k - xh) := by
    rw [← map_sub]
    congr 1
    abel
  rw [hsplit]
  calc ‖P (x j - xh) - P (x k - xh)‖ ≤ ‖P (x j - xh)‖ + ‖P (x k - xh)‖ := norm_sub_le _ _
    _ ≤ Real.sqrt (2 * ε / κ) * (1 / Real.sqrt (p j)) +
        Real.sqrt (2 * ε / κ) * (1 / Real.sqrt (p k)) := add_le_add aj ak
    _ = Real.sqrt (2 * ε / κ) * (1 / Real.sqrt (p j) + 1 / Real.sqrt (p k)) := by ring

end LoopProofs
