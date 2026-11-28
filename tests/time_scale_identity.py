import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
N = 12
W1, W2, B = torch.randn(N, 32) / 3, torch.randn(32, N) / 3, torch.randn(N) / 2
TARGET = torch.randn(N)


def field(s):
    return torch.tanh(s @ W1) @ W2 + B - 0.3 * s


def loss(s):
    return ((s - TARGET) ** 2).sum() / 2 + 0.3 * torch.cos(s).sum()


def trajectory(h, r, c=1.0):
    s, out = torch.zeros(N), [torch.zeros(N)]
    for _ in range(r):
        s = s + c * h * field(s)
        out.append(s)
    return out


def loss_after(s, m, h):
    for _ in range(m):
        s = s + h * field(s)
    return loss(s)


def grad_after(s, m, h):
    s = s.detach().requires_grad_(True)
    return torch.autograd.grad(loss_after(s, m, h), s)[0]


def check(h, r):
    S = trajectory(h, r)
    d = [S[k + 1] - S[k] for k in range(r)]
    ell = [float(loss(s)) for s in S]
    eps = 1e-6
    finite_diff = (float(loss(trajectory(h, r, 1 + eps)[r])) - float(loss(trajectory(h, r, 1 - eps)[r]))) / (2 * eps)
    nodes, weights = np.polynomial.legendre.leggauss(40)
    nodes, weights = (nodes + 1) / 2, weights / 2
    rho = []
    for k in range(r):
        g = grad_after(S[k + 1], r - k - 1, h)
        rho.append(sum(w * float(((g - grad_after(S[k] + t * d[k], r - k - 1, h)) * d[k]).sum())
                       for t, w in zip(nodes, weights)))
    right = r * (ell[r] - ell[r - 1]) + sum(rho)
    print("step %.2f, %d loops: Psi'(1) by finite differences %.8f, r (l_r - l_(r-1)) + sum rho %.8f" % (h, r, finite_diff, right))


if __name__ == "__main__":
    for h, r in ((1.0, 6), (0.3, 8), (0.05, 40)):
        check(h, r)
