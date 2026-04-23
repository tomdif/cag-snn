# CAG-SNN — does the CAG spectral gap predict SNN learning speed?

A minimal numerical test of the claim: for a feedforward LIF spiking neural network,
the convergence rate of gradient-based training is governed by the **CAG spectral gap γ
(Fiedler value of the normalized Laplacian)** of the network's neuron-graph Hasse diagram.

Initial draft targeted STDP as the training rule; during the experiment's bug-fix
phase we caught that pair-STDP at ~500 neurons on MNIST doesn't reliably converge
without added lateral inhibition / homeostasis. We pivoted to surrogate-gradient
BPTT (the standard practical SNN training method), which converges reliably and
keeps the γ ↔ convergence-rate question intact. See PRECOMMIT.md and
`experiment_sg.py`. The STDP scaffold (`snn.py`, `experiment.py`) is retained
as a failed-attempt artifact for honesty.

## The claim, stated precisely

For a 3-layer feedforward LIF SNN with synapse-weighted neuron-graph G:

> Let Ã = (A + Aᵀ)/2 be the symmetrized adjacency, D = diag(row-sums), and
> L_norm = I − D^{−1/2} Ã D^{−1/2} the normalized Laplacian. Define γ as the
> Fiedler value (second-smallest eigenvalue) of L_norm.
>
> **Hypothesis**: the training-sample index at which the SNN (trained via
> unsupervised STDP on the hidden layers + supervised linear readout) first
> crosses a held-out accuracy threshold is monotonically related to γ, with
> higher γ → faster convergence.

## Why it might be true

- γ of the normalized Laplacian is the canonical algebraic-connectivity /
  mixing-time measure of a graph
- Spike correlations propagate through the network like a diffusion, and
  diffusion mixing times are controlled by γ
- STDP weight updates are driven by these correlations, so the weight-learning
  dynamics should inherit the γ rate

## Why it might not be true

- STDP is nonlinear and non-diffusion-like
- The "effective" graph for learning may not be the initial synapse graph but
  the weight-updated one, which is itself evolving
- Finite-size effects at ~500-neuron scale may dominate

The only way to tell is to measure.

## Install

```bash
git clone https://github.com/tomdif/cag-snn.git
cd cag-snn
pip install -r requirements.txt
```

## Run

```bash
# Sanity: just compute γ for 5 configs (~30 sec on Mac)
python cag_gamma.py

# Main experiment: surrogate-gradient training (5 configs × 1 seed)
python experiment_sg.py --quick          # 3 configs, ~3 min
python experiment_sg.py                  # full 5 configs, ~15 min CPU / ~3 min GPU

# Ablation: 10 configs × 3 seeds + partial-correlation controls (~1 h on GPU)
python ablations.py --quick              # 4 configs × 2 seeds, ~10 min
python ablations.py                      # full 10 × 3 = 30 runs, ~1 h on 4060
```

Output:
- `results_sg.json` / `results_sg_stats.json`: main experiment
- `results_ablation.json` / `results_ablation_stats.json`: ablation + partial correlations

## Go / no-go (locked precommit)

**Main experiment:**
- STRONG GO: Spearman ρ(γ, conv_step) < −0.7 AND log-log fit R² > 0.6
- GO: ρ < −0.7 (monotone inverse, may not fit log-log cleanly)
- PARTIAL: ρ in [−0.7, −0.3]
- NO-GO: ρ > −0.3 or sign reversed

**Ablation (after main):**
- STRONG: raw ρ(γ) survives partial-correlation control for P (partial-ρ > 0.5 in magnitude)
- WEAK: raw ρ(γ) strong but partial-ρ near 0 — γ is a proxy for param count
- PARTIAL: partial-ρ in (0.3, 0.5)
- INCONCLUSIVE: raw ρ(γ) weak

**Initial 5-config run (tomdif's RTX 4060, 2026-04-22):**
Spearman ρ = −0.894, log-log R² = 0.798 → STRONG GO on main. Ablation pending.

## File layout

| File | Purpose |
|---|---|
| `PRECOMMIT.md` | Locked hypothesis and criteria (see for details) |
| `snn.py` | Minimal LIF SNN + pair-based STDP, pure PyTorch |
| `cag_gamma.py` | Compute Fiedler value of the SNN's neuron graph |
| `experiment.py` | Sweep 5 configs, train, measure γ vs convergence |

## Caveats

- ~500-neuron scale; bigger nets needed to match neuromorphic hardware regimes
- Rate-coded Poisson input, not event-based DVS data
- Pair-STDP only; triplet/reward-modulated STDP not tested
- γ measured at init; varies during training (we report both)
- 5 configs is thin for a clean R² fit; more configs needed for publishable result

This is a fast proof-of-concept, not a publishable study.

## License

MIT
