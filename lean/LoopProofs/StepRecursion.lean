import Mathlib

namespace LoopProofs

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

theorem step_recursion (Φ : E → E) (J : E → E →L[ℝ] E) (hΦ : ∀ y, HasFDerivAt Φ (J y) y)
    (hJ : Continuous J) (s d : E) :
    Φ (s + d) - Φ s = (∫ t in (0 : ℝ)..1, J (s + t • d)) d := by
  have hcont : Continuous fun t : ℝ => J (s + t • d) :=
    hJ.comp (continuous_const.add (continuous_id.smul continuous_const))
  have hg : ∀ t ∈ Set.uIcc (0 : ℝ) 1,
      HasDerivAt (fun t : ℝ => Φ (s + t • d)) (J (s + t • d) d) t := by
    intro t _
    have h1 : HasDerivAt (fun t : ℝ => s + t • d) d t := by
      have := ((hasDerivAt_id t).smul_const d).const_add s
      simpa using this
    exact (hΦ _).comp_hasDerivAt t h1
  have hint : IntervalIntegrable (fun t : ℝ => J (s + t • d) d) MeasureTheory.volume 0 1 :=
    (hcont.clm_apply continuous_const).intervalIntegrable 0 1
  have h := intervalIntegral.integral_eq_sub_of_hasDerivAt hg hint
  rw [ContinuousLinearMap.intervalIntegral_apply (hcont.intervalIntegrable 0 1) d, h]
  simp

theorem different_gains_different_states {S : Type*} (Γ : S → ℝ) (a b : S) (h : Γ a ≠ Γ b) : a ≠ b := by
  intro hab
  exact h (congrArg Γ hab)

end LoopProofs
