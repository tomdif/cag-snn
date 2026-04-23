"""Minimal LIF SNN with pair-based STDP, in pure PyTorch (no Norse/BindsNET dependency).

Structure:
  input (784) → hidden₁ (N1) → hidden₂ (N2) → readout (10)

LIF dynamics (discrete time, dt=1 ms):
  V(t+1) = V(t) * exp(-dt/tau_mem) + I(t)     when not in refractory
  spike  = (V > threshold)                      (hard)
  V      = 0 after spike; hold for refractory period

STDP:
  pre/post traces updated per step with exp decay τ.
  On post spike: Δw = A_pre * pre_trace[presyn]   (LTP)
  On pre spike:  Δw = A_post * post_trace[postsyn] (LTD)
  Weights clamped to [0, w_max].
"""
import torch
import torch.nn as nn


class LIFLayer(nn.Module):
    """One-step LIF update for a population. No learning here; just dynamics."""
    def __init__(self, n_in: int, n_out: int, tau_mem: float = 20.0,
                 v_thresh: float = 1.0, refractory: int = 5,
                 w_init_scale: float = 0.3, w_max: float = 1.0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.tau_mem = tau_mem
        self.v_thresh = v_thresh
        self.refractory = refractory
        self.w_max = w_max
        # W[j, i]: weight from presyn i → postsyn j
        self.W = nn.Parameter(
            torch.rand(n_out, n_in) * w_init_scale, requires_grad=False)
        # State (set by reset)
        self.v = None
        self.ref = None
        self.pre_trace = None
        self.post_trace = None
        self.decay = torch.tensor(-1.0 / tau_mem).exp().item()

    def reset(self, batch_size: int = 1, device='cpu'):
        self.v = torch.zeros(batch_size, self.n_out, device=device)
        self.ref = torch.zeros(batch_size, self.n_out, device=device)
        self.pre_trace = torch.zeros(batch_size, self.n_in, device=device)
        self.post_trace = torch.zeros(batch_size, self.n_out, device=device)

    def step(self, input_spikes: torch.Tensor, tau_trace: float = 20.0):
        """input_spikes: (B, n_in) binary spikes. Returns (B, n_out)."""
        B = input_spikes.shape[0]
        device = input_spikes.device
        if self.v is None or self.v.shape[0] != B:
            self.reset(batch_size=B, device=device)
        I = input_spikes @ self.W.t()
        in_ref = self.ref > 0
        self.v = torch.where(in_ref, torch.zeros_like(self.v),
                              self.v * self.decay + I)
        out_spikes = (self.v >= self.v_thresh).float()
        self.v = torch.where(out_spikes.bool(), torch.zeros_like(self.v), self.v)
        self.ref = torch.where(out_spikes.bool(),
                                torch.full_like(self.ref, self.refractory),
                                torch.clamp(self.ref - 1.0, min=0.0))
        tdec = float(torch.tensor(-1.0 / tau_trace).exp().item())
        # Stash BEFORE-update traces for STDP (these are the "incoming" pre-trace
        # and post-trace when this step's spikes occur)
        self._pre_trace_pre = self.pre_trace.clone()
        self._post_trace_pre = self.post_trace.clone()
        # Now update traces
        self.pre_trace = self.pre_trace * tdec + input_spikes
        self.post_trace = self.post_trace * tdec + out_spikes
        # Cache current-step spikes for STDP
        self._last_pre = input_spikes
        self._last_post = out_spikes
        return out_spikes

    def stdp_update(self, A_plus: float = 0.01, A_minus: float = 0.012):
        """Correct pair-based STDP using this step's spikes and the PRE-update traces.
          LTP: when a post spike occurs, strengthen by A_plus × pre_trace
               (pre fired recently before post → causal → potentiation)
          LTD: when a pre spike occurs, weaken by A_minus × post_trace
               (post fired recently before pre → anti-causal → depression)
        """
        if not hasattr(self, '_last_pre'):
            return
        pre = self._last_pre.mean(dim=0)                   # (n_in,)
        post = self._last_post.mean(dim=0)                  # (n_out,)
        pre_tr = self._pre_trace_pre.mean(dim=0)           # (n_in,)
        post_tr = self._post_trace_pre.mean(dim=0)         # (n_out,)
        # LTP: Δw_{j,i} += A_plus * post[j] * pre_tr[i]
        ltp = A_plus * torch.outer(post, pre_tr)
        # LTD: Δw_{j,i} -= A_minus * post_tr[j] * pre[i]
        ltd = A_minus * torch.outer(post_tr, pre)
        self.W.data += ltp - ltd
        self.W.data.clamp_(0.0, self.w_max)


class SNN(nn.Module):
    """3-layer feedforward SNN: input → H1 → H2 → output."""
    def __init__(self, n_in: int, n_h1: int, n_h2: int, n_out: int,
                 density: float = 1.0, **layer_kwargs):
        super().__init__()
        self.l1 = LIFLayer(n_in, n_h1, **layer_kwargs)
        self.l2 = LIFLayer(n_h1, n_h2, **layer_kwargs)
        # Readout: linear classifier trained supervised on cumulative spike counts
        self.readout = nn.Linear(n_h2, n_out, bias=True)
        # Apply sparsity mask if density < 1
        if density < 1.0:
            with torch.no_grad():
                mask1 = (torch.rand_like(self.l1.W) < density).float()
                mask2 = (torch.rand_like(self.l2.W) < density).float()
                self.l1.W.data *= mask1
                self.l2.W.data *= mask2
                self.register_buffer('mask1', mask1)
                self.register_buffer('mask2', mask2)
        self.density = density

    def reset(self, batch_size: int, device='cpu'):
        self.l1.reset(batch_size, device)
        self.l2.reset(batch_size, device)

    def forward_time(self, spike_train: torch.Tensor, stdp: bool = True):
        """spike_train: (B, T, n_in). Returns (B, n_out) logits from cumulative hidden₂ spikes."""
        B, T, _ = spike_train.shape
        self.reset(B, device=spike_train.device)
        cum_h2 = torch.zeros(B, self.l2.n_out, device=spike_train.device)
        for t in range(T):
            s1 = self.l1.step(spike_train[:, t, :])
            s2 = self.l2.step(s1)
            cum_h2 = cum_h2 + s2
            if stdp:
                self.l1.stdp_update()
                self.l2.stdp_update()
                # Re-apply sparsity mask
                if self.density < 1.0:
                    with torch.no_grad():
                        self.l1.W.data *= self.mask1
                        self.l2.W.data *= self.mask2
        logits = self.readout(cum_h2 / max(T, 1))
        return logits, cum_h2


def poisson_spike_train(x: torch.Tensor, T: int, rate_scale: float = 0.5) -> torch.Tensor:
    """x: (B, n_in) in [0,1]. Returns (B, T, n_in) Poisson spikes with rate x*rate_scale per step."""
    B, D = x.shape
    p = x.unsqueeze(1).expand(B, T, D) * rate_scale      # (B, T, D)
    spikes = (torch.rand_like(p) < p).float()
    return spikes


if __name__ == '__main__':
    # Smoke: one forward pass
    torch.manual_seed(0)
    m = SNN(n_in=784, n_h1=100, n_h2=100, n_out=10, density=1.0)
    x = torch.rand(4, 784)
    st = poisson_spike_train(x, T=30)
    logits, h2 = m.forward_time(st, stdp=True)
    print(f'logits shape: {tuple(logits.shape)}, h2 cum: {tuple(h2.shape)}')
    print(f'total synapses: {m.l1.W.numel() + m.l2.W.numel() + m.readout.weight.numel():,}')
