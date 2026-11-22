import Mathlib

namespace LoopProofs

theorem scaled_update_eigenvalue {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
    (J : V →ₗ[K] V) (v : V) (μ γ : K) (h : J v = μ • v) :
    v + γ • (J v - v) = (1 + γ * (μ - 1)) • v := by
  rw [h, add_smul, one_smul, mul_smul, sub_smul, one_smul]

end LoopProofs
