"""Checks the identity G = Psi'(1) + N of measure/update_gauge.py on a tiny random loop in double precision: G scales
the weights along theta(c), Psi'(1) scales the update, N moves the state along the branch shift E(s) of the core."""
import torch

from loopedlm import model as lm
from loopedlm.config import resolve
from loopedlm.model import LoopedLM
from measure.update_gauge import derivatives

# RMSNorm computes in float32 for the bf16 training; here it stays in float64, so that only the step h limits the check
lm.RMSNorm.forward = lambda self, x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
cfg = resolve("base", {"d": 32, "heads": 2, "ffn": 64, "vocab": 50, "T": 16, "mean_rec": 5})
m = LoopedLM(cfg).double().eval().requires_grad_(False)
with torch.no_grad():
    for b in m.core:
        b.attn.proj.weight.normal_(0, 0.3)
        b.mlp.proj.weight.normal_(0, 0.3)
x = torch.randint(0, 50, (2, 16))
y = torch.randint(0, 50, (2, 16))
dev = torch.device("cpu")
for h in (1e-2, 1e-3, 1e-4):
    d = derivatives(m, x, y, 5, 2, dev, h, direct=True)
    print("h %.0e  G %+.8f  Psi' %+.8f  N (G - Psi') %+.8f  N (direct) %+.8f  gap %.1e" % (
        h, d["G"], d["dpsi"], d["N"], d["N_direct"], abs(d["N"] - d["N_direct"])))
