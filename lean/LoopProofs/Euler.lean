import Mathlib

namespace LoopProofs

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem taylor_step (x x' : ℝ → E) (M : ℝ) (hM : 0 ≤ M)
    (hd : ∀ t, HasDerivAt x (x' t) t) (hl : ∀ t u, ‖x' t - x' u‖ ≤ M * |t - u|)
    (t h : ℝ) (hh : 0 ≤ h) :
    ‖x (t + h) - x t - h • x' t‖ ≤ M * h ^ 2 := by
  let g : ℝ → E := fun τ => x (t + τ) - τ • x' t
  have hg : ∀ τ, HasDerivAt g (x' (t + τ) - x' t) τ := by
    intro τ
    have h1 : HasDerivAt (fun τ => x (t + τ)) (x' (t + τ)) τ :=
      (hd (t + τ)).comp_const_add t τ
    have h2 : HasDerivAt (fun τ : ℝ => τ • x' t) ((1 : ℝ) • x' t) τ :=
      (hasDerivAt_id τ).smul_const (x' t)
    have h3 := h1.sub h2
    rw [one_smul] at h3
    exact h3
  have key := norm_image_sub_le_of_norm_deriv_le_segment' (f := g) (a := 0) (b := h)
    (C := M * h) (fun τ _ => (hg τ).hasDerivWithinAt) (fun τ hτ => by
      calc ‖x' (t + τ) - x' t‖ ≤ M * |t + τ - t| := hl _ _
        _ = M * τ := by rw [add_sub_cancel_left, abs_of_nonneg hτ.1]
        _ ≤ M * h := mul_le_mul_of_nonneg_left hτ.2.le hM) h ⟨hh, le_rfl⟩
  have e : g h - g 0 = x (t + h) - x t - h • x' t := by
    simp only [g, add_zero, zero_smul, sub_zero]
    abel
  rw [e] at key
  calc ‖x (t + h) - x t - h • x' t‖ ≤ M * h * (h - 0) := key
    _ = M * h ^ 2 := by ring

theorem one_step_error (F G : E → E) (x x' : ℝ → E) (sk sk1 : E) (t h L MG M2 : ℝ)
    (hh : 0 ≤ h) (hM2 : 0 ≤ M2)
    (hd : ∀ t, HasDerivAt x (x' t) t) (hode : ∀ t, x' t = F (x t))
    (hl : ∀ t u, ‖x' t - x' u‖ ≤ M2 * |t - u|)
    (hs : sk1 = sk + h • F sk + h ^ 2 • G sk) (hG : ‖G sk‖ ≤ MG)
    (hF : ‖F sk - F (x t)‖ ≤ L * ‖sk - x t‖) :
    ‖sk1 - x (t + h)‖ ≤ (1 + h * L) * ‖sk - x t‖ + h ^ 2 * (MG + M2) := by
  have tay := taylor_step x x' M2 hM2 hd hl t h hh
  rw [hode t] at tay
  have e : sk1 - x (t + h) = (sk - x t) + h • (F sk - F (x t)) + h ^ 2 • G sk
      - (x (t + h) - x t - h • F (x t)) := by
    rw [hs, smul_sub]
    abel
  rw [e]
  have hh2 : 0 ≤ h ^ 2 := sq_nonneg h
  calc ‖(sk - x t) + h • (F sk - F (x t)) + h ^ 2 • G sk - (x (t + h) - x t - h • F (x t))‖
      ≤ ‖sk - x t‖ + ‖h • (F sk - F (x t))‖ + ‖h ^ 2 • G sk‖
        + ‖x (t + h) - x t - h • F (x t)‖ := by
        refine le_trans (norm_sub_le _ _) ?_
        gcongr
        refine le_trans (norm_add_le _ _) ?_
        gcongr
        exact norm_add_le _ _
    _ ≤ ‖sk - x t‖ + h * (L * ‖sk - x t‖) + h ^ 2 * MG + M2 * h ^ 2 := by
        rw [norm_smul, norm_smul, Real.norm_of_nonneg hh, Real.norm_of_nonneg hh2]
        gcongr
    _ = (1 + h * L) * ‖sk - x t‖ + h ^ 2 * (MG + M2) := by ring

theorem discrete_gronwall (η : ℕ → ℝ) (a b : ℝ) (ha : 0 < a) (N : ℕ)
    (hrec : ∀ k, k < N → η (k + 1) ≤ (1 + a) * η k + b) :
    ∀ k, k ≤ N → η k ≤ (1 + a) ^ k * η 0 + b * ((1 + a) ^ k - 1) / a := by
  intro k
  induction k with
  | zero => intro _; simp
  | succ n ih =>
    intro hn
    have hpos : 0 < 1 + a := by linarith
    have ihn := ih (Nat.le_of_succ_le hn)
    calc η (n + 1) ≤ (1 + a) * η n + b := hrec n hn
      _ ≤ (1 + a) * ((1 + a) ^ n * η 0 + b * ((1 + a) ^ n - 1) / a) + b := by
        nlinarith [mul_le_mul_of_nonneg_left ihn hpos.le]
      _ = (1 + a) ^ (n + 1) * η 0 + b * ((1 + a) ^ (n + 1) - 1) / a := by
        field_simp
        ring

theorem gronwall_exp_bound (η : ℕ → ℝ) (n N : ℕ) (hn : 0 < n) (hN : N ≤ n) (L β c0 : ℝ) (hL : 0 < L)
    (hβ : 0 ≤ β) (hc0 : 0 ≤ c0)
    (hrec : ∀ k, k < N → η (k + 1) ≤ (1 + L / n) * η k + (1 / n) * β) (h0 : η 0 ≤ c0) :
    ∀ k, k ≤ N → η k ≤ Real.exp L * c0 + β * (Real.exp L - 1) / L := by
  intro k hk
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have ha : 0 < L / n := div_pos hL hnpos
  have hg := discrete_gronwall η (L / n) ((1 / n) * β) ha N hrec k hk
  have hp : (1 + L / n) ^ k ≤ Real.exp L := by
    have h1 : 1 + L / n ≤ Real.exp (L / n) := by
      have := Real.add_one_le_exp (L / n)
      linarith
    have h0' : 0 ≤ 1 + L / n := by positivity
    calc (1 + L / n) ^ k ≤ Real.exp (L / n) ^ k := pow_le_pow_left₀ h0' h1 k
      _ = Real.exp (k * (L / n)) := by rw [← Real.exp_nat_mul]
      _ ≤ Real.exp L := by
        apply Real.exp_le_exp.mpr
        have hkn : (k : ℝ) ≤ n := by exact_mod_cast le_trans hk hN
        rw [mul_div_assoc', div_le_iff₀ hnpos]
        nlinarith
  have hp1 : 1 ≤ (1 + L / n) ^ k := one_le_pow₀ (by linarith)
  have hη0 : (1 + L / n) ^ k * η 0 ≤ Real.exp L * c0 :=
    calc (1 + L / n) ^ k * η 0 ≤ (1 + L / n) ^ k * c0 :=
          mul_le_mul_of_nonneg_left h0 (by linarith)
      _ ≤ Real.exp L * c0 := mul_le_mul_of_nonneg_right hp hc0
  have hsec : (1 / n) * β * ((1 + L / n) ^ k - 1) / (L / n) ≤ β * (Real.exp L - 1) / L := by
    have e : (1 / (n : ℝ)) * β * ((1 + L / n) ^ k - 1) / (L / n) =
        β * ((1 + L / n) ^ k - 1) / L := by
      field_simp
    rw [e]
    apply div_le_div_of_nonneg_right _ hL.le
    exact mul_le_mul_of_nonneg_left (by linarith) hβ
  linarith

theorem euler_a (F G : E → E) (x x' : ℝ → E) (s : ℕ → E) (r N : ℕ) (hr : 0 < r)
    (hN : N ≤ r) (t0 L MG M2 c : ℝ) (hL : 0 < L) (hMG : 0 ≤ MG) (hM2 : 0 ≤ M2) (hc : 0 ≤ c)
    (hd : ∀ t, HasDerivAt x (x' t) t) (hode : ∀ t, x' t = F (x t))
    (hl : ∀ t u, ‖x' t - x' u‖ ≤ M2 * |t - u|)
    (hs : ∀ k, k < N → s (k + 1) = s k + (1 / (r : ℝ)) • F (s k) + (1 / (r : ℝ)) ^ 2 • G (s k))
    (hG : ∀ k, k < N → ‖G (s k)‖ ≤ MG)
    (hF : ∀ k, k < N →
      ‖F (s k) - F (x (t0 + k * (1 / r)))‖ ≤ L * ‖s k - x (t0 + k * (1 / r))‖)
    (h0 : ‖s 0 - x t0‖ ≤ c / r) :
    ∀ k, k ≤ N → ‖s k - x (t0 + k * (1 / r))‖ ≤
      (1 / r) * (c * Real.exp L + (MG + M2) * (Real.exp L - 1) / L) := by
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  have hrec : ∀ k, k < N → ‖s (k + 1) - x (t0 + ((k + 1 : ℕ) : ℝ) * (1 / r))‖ ≤
      (1 + L / r) * ‖s k - x (t0 + k * (1 / r))‖ + (1 / r) * ((1 / r) * (MG + M2)) := by
    intro k hk
    have et : t0 + ((k + 1 : ℕ) : ℝ) * (1 / r) = (t0 + k * (1 / r)) + 1 / r := by
      push_cast
      ring
    rw [et]
    have := one_step_error F G x x' (s k) (s (k + 1)) (t0 + k * (1 / r)) (1 / r) L MG M2
      (by positivity) hM2 hd hode hl (hs k hk) (hG k hk) (hF k hk)
    calc _ ≤ (1 + 1 / r * L) * ‖s k - x (t0 + k * (1 / r))‖ + (1 / r) ^ 2 * (MG + M2) := this
      _ = (1 + L / r) * ‖s k - x (t0 + k * (1 / r))‖ + (1 / r) * ((1 / r) * (MG + M2)) := by
        ring
  have h0' : ‖s 0 - x (t0 + ((0 : ℕ) : ℝ) * (1 / r))‖ ≤ c / r := by simpa using h0
  have hc0 : 0 ≤ c / r := div_nonneg hc hrpos.le
  have hβ : 0 ≤ (1 / (r : ℝ)) * (MG + M2) := by positivity
  intro k hk
  have := gronwall_exp_bound (fun k => ‖s k - x (t0 + (k : ℝ) * (1 / r))‖) r N hr hN L
    ((1 / r) * (MG + M2)) (c / r) hL hβ hc0 hrec h0' k hk
  calc ‖s k - x (t0 + k * (1 / r))‖
      ≤ Real.exp L * (c / r) + (1 / r) * (MG + M2) * (Real.exp L - 1) / L := this
    _ = (1 / r) * (c * Real.exp L + (MG + M2) * (Real.exp L - 1) / L) := by ring

theorem euler_b (F G : E → E) (sk sk1 xk : E) (r : ℕ) (hr : 0 < r) (L MG : ℝ)
    (hs : sk1 = sk + (1 / (r : ℝ)) • F sk + (1 / (r : ℝ)) ^ 2 • G sk) (hG : ‖G sk‖ ≤ MG)
    (hF : ‖F sk - F xk‖ ≤ L * ‖sk - xk‖) :
    ‖(r : ℝ) • (sk1 - sk) - F xk‖ ≤ L * ‖sk - xk‖ + MG / r := by
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  have e : (r : ℝ) • (sk1 - sk) - F xk = (F sk - F xk) + (1 / (r : ℝ)) • G sk := by
    rw [hs]
    have h1 : (r : ℝ) • ((1 / (r : ℝ)) • F sk) = F sk := by
      rw [smul_smul, mul_one_div_cancel hrpos.ne', one_smul]
    have h2 : (r : ℝ) • ((1 / (r : ℝ)) ^ 2 • G sk) = (1 / (r : ℝ)) • G sk := by
      rw [smul_smul]
      congr 1
      field_simp
    rw [show sk + (1 / (r : ℝ)) • F sk + (1 / (r : ℝ)) ^ 2 • G sk - sk =
        (1 / (r : ℝ)) • F sk + (1 / (r : ℝ)) ^ 2 • G sk by abel, smul_add, h1, h2]
    abel
  rw [e]
  calc ‖(F sk - F xk) + (1 / (r : ℝ)) • G sk‖ ≤ ‖F sk - F xk‖ + ‖(1 / (r : ℝ)) • G sk‖ :=
        norm_add_le _ _
    _ ≤ L * ‖sk - xk‖ + (1 / r) * MG := by
        rw [norm_smul, Real.norm_of_nonneg (by positivity)]
        gcongr
    _ = L * ‖sk - xk‖ + MG / r := by ring

theorem euler_c (F G : E → E) (x x' : ℝ → E) (s : ℕ → E) (r r' : ℕ) (hr : 0 < r)
    (hrr : r ≤ r') (t0 L MG M2 c : ℝ) (hL : 0 < L) (hMG : 0 ≤ MG) (hM2 : 0 ≤ M2)
    (hc : 0 ≤ c)
    (hd : ∀ t, HasDerivAt x (x' t) t) (hode : ∀ t, x' t = F (x t))
    (hl : ∀ t u, ‖x' t - x' u‖ ≤ M2 * |t - u|)
    (hs : ∀ k, k < r' → s (k + 1) = s k + ((r : ℝ) / r') •
      ((1 / (r : ℝ)) • F (s k) + (1 / (r : ℝ)) ^ 2 • G (s k)))
    (hG : ∀ k, k < r' → ‖G (s k)‖ ≤ MG)
    (hF : ∀ k, k < r' →
      ‖F (s k) - F (x (t0 + k * (1 / r')))‖ ≤ L * ‖s k - x (t0 + k * (1 / r'))‖)
    (h0 : ‖s 0 - x t0‖ ≤ c / r) :
    ∀ k, k ≤ r' → ‖s k - x (t0 + k * (1 / r'))‖ ≤
      (1 / r) * (c * Real.exp L + (MG + M2) * (Real.exp L - 1) / L) := by
  have hr' : 0 < r' := lt_of_lt_of_le hr hrr
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  have hr'pos : (0 : ℝ) < r' := by exact_mod_cast hr'
  have hrr' : (r : ℝ) ≤ r' := by exact_mod_cast hrr
  have hrec : ∀ k, k < r' → ‖s (k + 1) - x (t0 + ((k + 1 : ℕ) : ℝ) * (1 / r'))‖ ≤
      (1 + L / r') * ‖s k - x (t0 + k * (1 / r'))‖ +
        (1 / r') * ((1 / r') * ((r' / r) * MG + M2)) := by
    intro k hk
    have et : t0 + ((k + 1 : ℕ) : ℝ) * (1 / r') = (t0 + k * (1 / r')) + 1 / r' := by
      push_cast
      ring
    rw [et]
    have hs' : s (k + 1) = s k + (1 / (r' : ℝ)) • F (s k) +
        (1 / (r' : ℝ)) ^ 2 • (((r' : ℝ) / r) • G (s k)) := by
      have c1 : (r : ℝ) / r' * (1 / r) = 1 / r' := by field_simp
      have c2 : (r : ℝ) / r' * (1 / r) ^ 2 = (1 / r') ^ 2 * (r' / r) := by
        field_simp
      rw [hs k hk, smul_add, smul_smul, smul_smul, c1, c2, ← smul_smul, add_assoc]
    have hG' : ‖((r' : ℝ) / r) • G (s k)‖ ≤ (r' / r) * MG := by
      rw [norm_smul, Real.norm_of_nonneg (by positivity)]
      exact mul_le_mul_of_nonneg_left (hG k hk) (by positivity)
    have := one_step_error F (fun v => ((r' : ℝ) / r) • G v) x x' (s k) (s (k + 1))
      (t0 + k * (1 / r')) (1 / r') L ((r' / r) * MG) M2 (by positivity) hM2 hd hode hl hs' hG'
      (hF k hk)
    calc _ ≤ (1 + 1 / r' * L) * ‖s k - x (t0 + k * (1 / r'))‖ +
          (1 / r') ^ 2 * ((r' / r) * MG + M2) := this
      _ = _ := by ring
  have h0' : ‖s 0 - x (t0 + ((0 : ℕ) : ℝ) * (1 / r'))‖ ≤ c / r := by simpa using h0
  have hc0 : 0 ≤ c / r := div_nonneg hc hrpos.le
  have hβ : 0 ≤ (1 / (r' : ℝ)) * ((r' / r) * MG + M2) := by positivity
  intro k hk
  have hk' := gronwall_exp_bound (fun k => ‖s k - x (t0 + (k : ℝ) * (1 / r'))‖) r' r' hr' le_rfl L
    ((1 / r') * ((r' / r) * MG + M2)) (c / r) hL hβ hc0 hrec h0' k hk
  have hβle : (1 / (r' : ℝ)) * ((r' / r) * MG + M2) ≤ (1 / r) * (MG + M2) := by
    have e : (1 / (r' : ℝ)) * ((r' / r) * MG + M2) = MG / r + M2 / r' := by
      field_simp
    rw [e]
    have : M2 / (r' : ℝ) ≤ M2 / r := div_le_div_of_nonneg_left hM2 hrpos hrr'
    have e2 : (1 / (r : ℝ)) * (MG + M2) = MG / r + M2 / r := by ring
    rw [e2]
    linarith
  have hexp : 0 ≤ (Real.exp L - 1) / L := by
    apply div_nonneg _ hL.le
    have := Real.add_one_le_exp L
    linarith
  calc ‖s k - x (t0 + k * (1 / r'))‖
      ≤ Real.exp L * (c / r) + (1 / r') * ((r' / r) * MG + M2) * (Real.exp L - 1) / L := hk'
    _ ≤ Real.exp L * (c / r) + (1 / r) * (MG + M2) * (Real.exp L - 1) / L := by
        have := mul_le_mul_of_nonneg_right hβle hexp
        rw [mul_div_assoc, mul_div_assoc]
        linarith
    _ = (1 / r) * (c * Real.exp L + (MG + M2) * (Real.exp L - 1) / L) := by ring

end LoopProofs
