import Mathlib
import LoopProofs.Confinement

namespace LoopProofs

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

omit [NormedSpace ℝ E] in
theorem steps_geometric (x : ℕ → E) (R : ℕ) (q : ℝ) (hq0 : 0 ≤ q)
    (hq : ∀ k, R ≤ k → ‖x (k + 2) - x (k + 1)‖ ≤ q * ‖x (k + 1) - x k‖) (i : ℕ) :
    ‖x (R + i + 1) - x (R + i)‖ ≤ q ^ i * ‖x (R + 1) - x R‖ := by
  induction i with
  | zero => simp
  | succ n ih =>
    have h := hq (R + n) (Nat.le_add_right R n)
    have e1 : R + (n + 1) + 1 = R + n + 2 := by ring
    have e2 : R + (n + 1) = R + n + 1 := by ring
    rw [e1, e2]
    calc ‖x (R + n + 2) - x (R + n + 1)‖ ≤ q * ‖x (R + n + 1) - x (R + n)‖ := h
      _ ≤ q * (q ^ n * ‖x (R + 1) - x R‖) := mul_le_mul_of_nonneg_left ih hq0
      _ = q ^ (n + 1) * ‖x (R + 1) - x R‖ := by ring

theorem beyond_support_bound (P : E →L[ℝ] F) (hP : ∀ v, ‖P v‖ ≤ ‖v‖) (x : ℕ → E) (xh : E) (R : ℕ)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hq : ∀ k, R ≤ k → ‖x (k + 2) - x (k + 1)‖ ≤ q * ‖x (k + 1) - x k‖) (j : ℕ) :
    ‖P (x (R + j) - xh)‖ ≤ ‖P (x R - xh)‖ + ‖x (R + 1) - x R‖ / (1 - q) := by
  have tele : x (R + j) - xh =
      (x R - xh) + ∑ i ∈ Finset.range j, (x (R + i + 1) - x (R + i)) := by
    have := Finset.sum_range_sub (fun i => x (R + i)) j
    rw [show (∑ i ∈ Finset.range j, (x (R + i + 1) - x (R + i))) =
        ∑ i ∈ Finset.range j, (x (R + (i + 1)) - x (R + i)) by rfl, this]
    simp only [Nat.add_zero]
    abel
  have hsum : ‖∑ i ∈ Finset.range j, (x (R + i + 1) - x (R + i))‖ ≤
      ‖x (R + 1) - x R‖ / (1 - q) := by
    calc ‖∑ i ∈ Finset.range j, (x (R + i + 1) - x (R + i))‖
        ≤ ∑ i ∈ Finset.range j, ‖x (R + i + 1) - x (R + i)‖ := norm_sum_le _ _
      _ ≤ ∑ i ∈ Finset.range j, q ^ i * ‖x (R + 1) - x R‖ :=
          Finset.sum_le_sum fun i _ => steps_geometric x R q hq0 hq i
      _ = (∑ i ∈ Finset.range j, q ^ i) * ‖x (R + 1) - x R‖ := by rw [Finset.sum_mul]
      _ ≤ (1 / (1 - q)) * ‖x (R + 1) - x R‖ := by
          apply mul_le_mul_of_nonneg_right _ (norm_nonneg _)
          have h := geom_sum_Ico_le_of_lt_one (m := 0) (n := j) hq0 hq1
          rw [← Finset.range_eq_Ico, pow_zero] at h
          exact h
      _ = ‖x (R + 1) - x R‖ / (1 - q) := by ring
  rw [tele, map_add]
  calc ‖P (x R - xh) + P (∑ i ∈ Finset.range j, (x (R + i + 1) - x (R + i)))‖
      ≤ ‖P (x R - xh)‖ + ‖P (∑ i ∈ Finset.range j, (x (R + i + 1) - x (R + i)))‖ :=
        norm_add_le _ _
    _ ≤ ‖P (x R - xh)‖ + ‖∑ i ∈ Finset.range j, (x (R + i + 1) - x (R + i))‖ :=
        add_le_add le_rfl (hP _)
    _ ≤ ‖P (x R - xh)‖ + ‖x (R + 1) - x R‖ / (1 - q) := add_le_add le_rfl hsum

theorem excursion_cost {ι : Type*} (s : Finset ι) (p ℓ : ι → ℝ) (ℓs κ ε pmin a : ℝ)
    (P : E →L[ℝ] F) (x : ι → E) (xh : E)
    (hp : ∀ k ∈ s, 0 ≤ p k) (hℓ : ∀ k ∈ s, ℓs ≤ ℓ k)
    (hε : ∑ k ∈ s, p k * (ℓ k - ℓs) = ε) (hκ : 0 < κ) (hpmin : 0 < pmin)
    (j k : ι) (hj : j ∈ s) (hk : k ∈ s) (hpj : pmin ≤ p j) (hpk : pmin ≤ p k)
    (hqj : κ / 2 * ‖P (x j - xh)‖ ^ 2 ≤ ℓ j - ℓs)
    (hqk : κ / 2 * ‖P (x k - xh)‖ ^ 2 ≤ ℓ k - ℓs)
    (ha : 0 ≤ a) (hexc : a ≤ ‖P (x j - x k)‖) :
    κ * pmin * a ^ 2 / 8 ≤ ε := by
  have uj := confinement_sq s p ℓ ℓs κ ε P x xh hp hℓ hε hκ j hj (lt_of_lt_of_le hpmin hpj) hqj
  have uk := confinement_sq s p ℓ ℓs κ ε P x xh hp hℓ hε hκ k hk (lt_of_lt_of_le hpmin hpk) hqk
  have hε0 : 0 ≤ ε := hε ▸ eps_nonneg s p ℓ ℓs hp hℓ
  have bj : κ * pmin * ‖P (x j - xh)‖ ^ 2 ≤ 2 * ε := by
    have h := (le_div_iff₀ (mul_pos hκ (lt_of_lt_of_le hpmin hpj))).mp uj
    nlinarith [sq_nonneg ‖P (x j - xh)‖, mul_le_mul_of_nonneg_left hpj hκ.le]
  have bk : κ * pmin * ‖P (x k - xh)‖ ^ 2 ≤ 2 * ε := by
    have h := (le_div_iff₀ (mul_pos hκ (lt_of_lt_of_le hpmin hpk))).mp uk
    nlinarith [sq_nonneg ‖P (x k - xh)‖, mul_le_mul_of_nonneg_left hpk hκ.le]
  have tri : ‖P (x j - x k)‖ ≤ ‖P (x j - xh)‖ + ‖P (x k - xh)‖ := by
    have e : P (x j - x k) = P (x j - xh) - P (x k - xh) := by
      rw [← map_sub]
      congr 1
      abel
    rw [e]
    exact norm_sub_le _ _
  have hsum : a ≤ ‖P (x j - xh)‖ + ‖P (x k - xh)‖ := le_trans hexc tri
  have hsq : a ^ 2 ≤ 2 * (‖P (x j - xh)‖ ^ 2 + ‖P (x k - xh)‖ ^ 2) := by
    nlinarith [sq_nonneg (‖P (x j - xh)‖ - ‖P (x k - xh)‖), norm_nonneg (P (x j - xh)),
      norm_nonneg (P (x k - xh))]
  have hkp : 0 < κ * pmin := mul_pos hκ hpmin
  nlinarith [mul_le_mul_of_nonneg_left hsq hkp.le]

theorem past_last_weighted_loop {ι : Type*} (s : Finset ι) (p ℓ : ι → ℝ) (ℓs κ ε : ℝ) (P : E →L[ℝ] F)
    (hP : ∀ v, ‖P v‖ ≤ ‖v‖) (x : ℕ → E) (xh : E) (σ : ι → ℕ)
    (hp : ∀ k ∈ s, 0 ≤ p k) (hℓ : ∀ k ∈ s, ℓs ≤ ℓ k)
    (hε : ∑ k ∈ s, p k * (ℓ k - ℓs) = ε) (hκ : 0 < κ)
    (K : ι) (hK : K ∈ s) (hpK : 0 < p K)
    (hqK : κ / 2 * ‖P (x (σ K) - xh)‖ ^ 2 ≤ ℓ K - ℓs)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hq : ∀ k, σ K ≤ k → ‖x (k + 2) - x (k + 1)‖ ≤ q * ‖x (k + 1) - x k‖) (j : ℕ) :
    ‖P (x (σ K + j) - xh)‖ ≤
      Real.sqrt (2 * ε / κ) * (1 / Real.sqrt (p K)) + ‖x (σ K + 1) - x (σ K)‖ / (1 - q) := by
  have h1 := beyond_support_bound P hP x xh (σ K) q hq0 hq1 hq j
  have h2 := confinement s p ℓ ℓs κ ε P (fun k => x (σ k)) xh hp hℓ hε hκ K hK hpK hqK
  linarith

end LoopProofs
