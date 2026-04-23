"""LIF SNN with surrogate-gradient training (BPTT through spike nonlinearity).

Forward pass: exact LIF dynamics with hard spike at threshold.
Backward pass: surrogate derivative σ'(V - thresh) (fast-sigmoid-like).

This trains reliably on MNIST at moderate size (~500 hidden neurons), unlike
pure STDP. Keeps the SNN temporal/sparse structure intact so γ still makes
sense as a topology-driven quantity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """Spike function with fast-sigmoid surrogate gradient."""
    @staticmethod
    def forward(ctx, v_minus_thresh):
        ctx.save_for_backward(v_minus_thresh)
        return (v_minus_thresh > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # fast-sigmoid surrogate: 1 / (1 + |x|/slope)^2
        slope = 5.0
        surr = 1.0 / (1.0 + slope * x.abs()) ** 2
        return grad_output * surr


spike_fn = FastSigmoidSurrogate.apply


class LIFSG(nn.Module):
    """One LIF-surrogate-gradient layer. No recurrence inside the layer."""
    def __init__(self, n_in: int, n_out: int, tau_mem: float = 20.0,
                 v_thresh: float = 1.0, density: float = 1.0,
                 w_init_scale: float = 0.5):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.v_thresh = v_thresh
        self.decay = float(torch.tensor(-1.0 / tau_mem).exp().item())
        self.W = nn.Parameter(torch.randn(n_out, n_in) * w_init_scale / (n_in ** 0.5))
        if density < 1.0:
            mask = (torch.rand_like(self.W) < density).float()
            self.register_buffer('mask', mask)
            with torch.no_grad():
                self.W.data *= mask
        self.density = density

    def masked_W(self):
        if self.density < 1.0:
            return self.W * self.mask
        return self.W

    def forward(self, input_spikes_seq: torch.Tensor) -> torch.Tensor:
        """input_spikes_seq: (B, T, n_in). Returns (B, T, n_out)."""
        B, T, _ = input_spikes_seq.shape
        W = self.masked_W()
        v = torch.zeros(B, self.n_out, device=input_spikes_seq.device)
        out = []
        for t in range(T):
            I = input_spikes_seq[:, t, :] @ W.t()
            v = v * self.decay + I
            s = spike_fn(v - self.v_thresh)
            # Hard reset: detach to avoid grad through reset (common choice)
            v = v * (1 - s.detach())
            out.append(s)
        return torch.stack(out, dim=1)


class SGSNN(nn.Module):
    """3-layer feedforward SNN trained with surrogate gradient."""
    def __init__(self, n_in: int, n_h1: int, n_h2: int, n_out: int,
                 density: float = 1.0, **kwargs):
        super().__init__()
        self.l1 = LIFSG(n_in, n_h1, density=density, **kwargs)
        self.l2 = LIFSG(n_h1, n_h2, density=density, **kwargs)
        self.readout = nn.Linear(n_h2, n_out)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, T, n_in). Returns logits (B, n_out)."""
        s1 = self.l1(x_seq)              # (B, T, n_h1)
        s2 = self.l2(s1)                 # (B, T, n_h2)
        # Readout on cumulative spikes (rate code)
        cum = s2.sum(dim=1) / s2.shape[1]
        return self.readout(cum)


def poisson_spike_train(x: torch.Tensor, T: int, rate_scale: float = 0.5) -> torch.Tensor:
    B, D = x.shape
    p = x.unsqueeze(1).expand(B, T, D) * rate_scale
    return (torch.rand_like(p) < p).float()


if __name__ == '__main__':
    torch.manual_seed(0)
    m = SGSNN(784, 100, 50, 10, density=1.0)
    x = torch.rand(8, 784)
    st = poisson_spike_train(x, T=30)
    logits = m(st)
    y = torch.randint(0, 10, (8,))
    loss = F.cross_entropy(logits, y)
    loss.backward()
    gnorm = sum(p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
    print(f'logits: {tuple(logits.shape)}, loss: {loss.item():.4f}, grad norm: {gnorm:.4f}')
