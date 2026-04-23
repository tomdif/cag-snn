---
name: CAG spectral gap ↔ SNN-STDP convergence rate
date: 2026-04-22
locked_before_running: true
---

# Precommit — does CAG γ of a network's Hasse diagram predict STDP learning rate?

## Hypothesis

**H1**: For a feedforward 3-layer LIF SNN trained with pair-based STDP on a
fixed task, the convergence rate (measured as training step at which
classification accuracy first exceeds a threshold τ) is monotonically related
to the CAG spectral gap γ of the network's synapse-weighted Hasse diagram.

**Stronger form (H2)**: log(convergence_step) ∝ −log(γ) with R² > 0.6 across
architectures at fixed total parameter budget.

## CAG γ, precisely

Given the feedforward connectivity W₁ ∈ ℝ^{N₁×N₀}, W₂ ∈ ℝ^{N₂×N₁}, W₃ ∈ ℝ^{N₃×N₂}:

1. Build a neuron-level directed graph G on the set of all neurons
   V = {input} ∪ {hidden₁} ∪ {hidden₂} ∪ {output} with edges weighted by |W_{ij}|.
2. Symmetrize: Ã = (G + Gᵀ)/2.
3. Row-normalize to a stochastic matrix M.
4. Compute eigenvalues {1 = λ₁ ≥ |λ₂| ≥ …}.
5. **γ := 1 − |λ₂|** (spectral gap of the random walk on the Hasse diagram).

This is the same quantity that controls mixing time of a Markov chain; the
analog of your CAG γ_2 = 0.276... constant, specialized to the specific
network topology.

## Network configs to test (5 points for γ vs convergence plot)

All feedforward LIF SNNs, 784 input → H₁ → H₂ → 10 output, with ~120k synapses:

| name  | H₁ | H₂ | density | total synapses |
|-------|----|----|---------|-----|
| wide  | 150 | 150 | 1.0  | 140k |
| bottleneck | 300 | 20 | 1.0 | 240k + 200 |
| balanced | 100 | 100 | 1.0 | 78k + 1k |
| deep-narrow | 60 | 60 | 1.0 | 47k + 600 |
| sparse | 200 | 100 | 0.5 | ~80k |

Varying H₁, H₂, and density gives a range of γ values (we don't know the
range yet — part of the test is measuring what γ each config produces).

## Training protocol

- Task: MNIST, first 1000 training samples, 10 classes
- Input encoding: Poisson spike trains (rate ∝ pixel intensity) over 100 ms, dt=1 ms
- Hidden layers: LIF (τ_mem = 20 ms, threshold = 1.0, reset to 0, refractory 5 ms)
- STDP on W₁, W₂: pair-based, τ_pre = τ_post = 20 ms, A_pre = 0.01, A_post = −0.012
- Readout: linear classifier trained supervised on hidden-2 spike counts (this
  isolates the CLAIM to hidden-layer STDP convergence)
- Evaluate accuracy on held-out 200 samples every 100 training samples
- **Convergence step**: first training-sample index where accuracy > 35%
  (higher than chance 10%, below saturation to leave signal)

## Go / no-go

- **STRONG GO**: log(step) ∝ −log(γ) with R² > 0.6 AND slope has expected sign
  (higher γ → faster convergence)
- **GO**: Spearman ρ < −0.7 (monotone inverse relation) across 5 configs
- **PARTIAL**: Spearman ρ in [-0.7, -0.3]
- **NO-GO**: ρ > -0.3 or sign reversed

## Retraction triggers

- If one config fails to reach 35% accuracy in the budget, re-examine metric
  (perhaps threshold too high for the hardest config).
- If γ values are all nearly identical, configs don't discriminate — need
  wider variation.
- Changing the configs, threshold, or metric after seeing the γ numbers:
  STOP and retract.

## What this does NOT prove

- Not a proof of any CAG theorem for SNNs
- Not an engineering claim for real neuromorphic hardware
- Not a claim that γ is the tight bound — only that it *scales with* convergence
- Not validated beyond MNIST + 3-layer feedforward + pair-STDP + 5 configs

## If STRONG GO: what's next

Test on richer networks (add recurrence, skip connections); test with different
STDP variants (triplet, reward-modulated); test on spike-encoded physical
signals (DVS camera events). The real target is a clean theorem relating CAG γ
to learning-dynamics mixing time; this experiment is the numerical bait to
see if the relationship is worth chasing analytically.
