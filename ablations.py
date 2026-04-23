"""Ablation run: robustness of the γ ↔ convergence-rate finding.

Tests four questions:
  (a) Does the correlation survive with more configs (n=10)?
  (b) Does it survive across seeds (mean ± std per config from 3 seeds)?
  (c) Do trivial baselines (parameter count, edge count, mean degree) predict
      conv_step equally well? If so, γ is a proxy, not a causal predictor.
  (d) Partial correlation of γ with conv_step CONTROLLING FOR P (total params).
      If partial-ρ stays strongly negative, γ adds real information beyond size.

Output:
  results_ablation.json              — per (config, seed) training records
  results_ablation_stats.json        — summary table + all correlations
"""
import argparse
import json
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from snn_sg import SGSNN, poisson_spike_train
from cag_gamma import compute_gamma_for_snn
from experiment_sg import train_one


# 10 configs: the original 5 + 5 new ones chosen to widen γ, P, and
# H1/H2/density variation.
CONFIGS = [
    ('wide',         150, 150, 1.0),
    ('bottleneck',   300,  20, 1.0),
    ('balanced',     100, 100, 1.0),
    ('deep_narrow',   60,  60, 1.0),
    ('sparse',       200, 100, 0.5),
    ('tall_wide',    250, 250, 1.0),   # more params — tests size confound
    ('tall_narrow',   80,  40, 1.0),   # fewer params — tests size confound
    ('very_sparse',  200, 200, 0.25),  # very sparse — tests density effect
    ('asymmetric',    50, 200, 1.0),   # reversed H1/H2 shape
    ('block_sparse', 150, 150, 0.3),   # wide but sparse
]


def count_params(h1, h2, density):
    """Approx. total trainable params (synapses) in the SNN."""
    w1 = 784 * h1 * density
    w2 = h1 * h2 * density
    wr = h2 * 10
    return int(w1 + w2 + wr)


def edge_count(h1, h2, density):
    return int((784 * h1 + h1 * h2) * density)


def mean_degree(h1, h2, density):
    """Mean degree of the symmetrized neuron graph over all nodes."""
    n_nodes = 784 + h1 + h2 + 10
    # Each directed edge contributes degree 1 on each side after symmetrization
    edges = (784 * h1 + h1 * h2) * density + h2 * 10
    return 2 * edges / n_nodes


def partial_spearman(x, y, z):
    """Spearman partial correlation ρ(x, y | z). Takes rank-ordered inputs."""
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    zr = np.argsort(np.argsort(z)).astype(float)
    rxy = np.corrcoef(xr, yr)[0, 1]
    rxz = np.corrcoef(xr, zr)[0, 1]
    ryz = np.corrcoef(yr, zr)[0, 1]
    denom = np.sqrt(max(1 - rxz ** 2, 1e-12) * max(1 - ryz ** 2, 1e-12))
    return (rxy - rxz * ryz) / denom


def loglog_r2(x, y):
    """R² of a log-log linear fit."""
    lx = np.log(x); ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    ss_res = ((ly - pred) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return float(r2), float(slope), float(intercept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train', type=int, default=2000)
    ap.add_argument('--T', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out', default='./results_ablation.json')
    ap.add_argument('--quick', action='store_true',
                    help='Smoke test: 4 configs, 2 seeds, 500 samples')
    args = ap.parse_args()

    if args.quick:
        configs = CONFIGS[:4]
        seeds = args.seeds[:2]
        n_train = 500
        T = 20
    else:
        configs = CONFIGS
        seeds = args.seeds
        n_train = args.n_train
        T = args.T

    device = torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Configs: {[c[0] for c in configs]}')
    print(f'Seeds: {seeds}')
    print(f'n_train={n_train}, T={T}\n')

    rows = []          # per (config, seed)
    t0 = time.time()
    for name, h1, h2, d in configs:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f'[{name} seed={seed}] ', end='', flush=True)
            r = train_one(name, h1, h2, d, n_train, T, args.batch_size, device)
            r['seed'] = seed
            r['P'] = count_params(h1, h2, d)
            r['edges'] = edge_count(h1, h2, d)
            r['mean_degree'] = mean_degree(h1, h2, d)
            print(f'γ={r["gamma_init"]:.3f}  conv={r["convergence_step"]}  '
                  f'acc={r["final_acc"]:.3f}  wall={r["wall_s"]:.0f}s')
            rows.append(r)
            with open(args.out, 'w') as f:  # checkpoint after each run
                json.dump(rows, f, indent=2, default=str)

    print(f'\nTotal wall: {time.time() - t0:.1f}s')

    # --- Analysis ---
    # Aggregate per-config: median convergence_step across seeds (or None if
    # no seed converged; mean γ, etc.)
    agg = {}
    for r in rows:
        agg.setdefault(r['name'], []).append(r)
    cfg_rows = []
    for name, entries in agg.items():
        conv_steps = [e['convergence_step'] for e in entries
                       if e['convergence_step'] is not None]
        if not conv_steps:
            continue
        cfg_rows.append({
            'name': name,
            'gamma': float(np.mean([e['gamma_init'] for e in entries])),
            'conv_step_median': float(np.median(conv_steps)),
            'conv_step_std': float(np.std(conv_steps)) if len(conv_steps) > 1 else 0.0,
            'n_seeds_converged': len(conv_steps),
            'n_seeds_total': len(entries),
            'P': entries[0]['P'],
            'edges': entries[0]['edges'],
            'mean_degree': entries[0]['mean_degree'],
            'final_acc_mean': float(np.mean([e['final_acc'] for e in entries])),
        })

    if len(cfg_rows) < 4:
        print(f'\nOnly {len(cfg_rows)} configs converged in ≥1 seed — insufficient for ablation.')
        return

    gs = np.array([r['gamma'] for r in cfg_rows])
    ss = np.array([r['conv_step_median'] for r in cfg_rows])
    ps = np.array([r['P'] for r in cfg_rows])
    es = np.array([r['edges'] for r in cfg_rows])
    md = np.array([r['mean_degree'] for r in cfg_rows])

    print('\n=== Per-config aggregate ===')
    print(f'{"name":<14} {"γ":>7} {"conv_med":>10} {"P":>10} {"acc":>6} {"seeds":>8}')
    for r in cfg_rows:
        print(f'{r["name"]:<14} {r["gamma"]:>7.3f} {r["conv_step_median"]:>10.0f} '
              f'{r["P"]:>10d} {r["final_acc_mean"]:>6.3f} '
              f'{r["n_seeds_converged"]}/{r["n_seeds_total"]:>3}')

    print('\n=== Pairwise correlations (Spearman ρ; more-negative = stronger) ===')
    for (x, xname) in [(gs, 'γ'), (ps, 'P'), (es, 'edges'), (md, 'mean_degree')]:
        rho, pv = spearmanr(x, ss)
        r2, slope, _ = loglog_r2(x, ss)
        print(f'  ρ({xname:<12}, conv_step) = {rho:+.3f}  (p={pv:.3f})   '
              f'log-log R²={r2:.3f}  slope={slope:+.3f}')

    # Partial correlation of γ with conv_step controlling for P
    print('\n=== Partial correlations (controlling for confounds) ===')
    p_par = partial_spearman(gs, ss, ps)
    print(f'  ρ(γ, conv_step | P)              = {p_par:+.3f}')
    p_par_edges = partial_spearman(gs, ss, es)
    print(f'  ρ(γ, conv_step | edges)          = {p_par_edges:+.3f}')
    p_par_md = partial_spearman(gs, ss, md)
    print(f'  ρ(γ, conv_step | mean_degree)    = {p_par_md:+.3f}')

    # Verdict
    rho_g, _ = spearmanr(gs, ss)
    r2_g, _, _ = loglog_r2(gs, ss)
    print('\n=== Ablation verdict ===')
    if abs(p_par) < 0.3 and abs(rho_g) > 0.7:
        print(f'WEAK — raw ρ(γ)={rho_g:+.3f} is strong but partial-ρ({p_par:+.3f}) is near zero.')
        print('→ γ is probably a proxy for total param count. The CAG-specific claim is NOT supported.')
    elif abs(p_par) > 0.5 and rho_g < -0.7:
        print(f'STRONG — raw ρ(γ)={rho_g:+.3f} survives partialling-out P: partial-ρ={p_par:+.3f}.')
        print('→ γ adds information beyond total parameter count. The CAG-specific claim is supported.')
    elif abs(p_par) > 0.3:
        print(f'PARTIAL — partial-ρ={p_par:+.3f} is nonzero but not strongly so.')
        print('→ γ may carry some signal beyond P, but not clearly.')
    else:
        print(f'INCONCLUSIVE — γ effect weak even before controls (raw ρ={rho_g:+.3f}).')

    out_stats = {
        'per_config': cfg_rows,
        'spearman_raw': {
            'gamma': float(spearmanr(gs, ss)[0]),
            'P':     float(spearmanr(ps, ss)[0]),
            'edges': float(spearmanr(es, ss)[0]),
            'mean_degree': float(spearmanr(md, ss)[0]),
        },
        'partial_spearman': {
            'gamma_given_P': p_par,
            'gamma_given_edges': p_par_edges,
            'gamma_given_mean_degree': p_par_md,
        },
        'loglog_r2': {
            'gamma': loglog_r2(gs, ss)[0],
            'P':     loglog_r2(ps, ss)[0],
            'edges': loglog_r2(es, ss)[0],
            'mean_degree': loglog_r2(md, ss)[0],
        },
    }
    with open(args.out.replace('.json', '_stats.json'), 'w') as f:
        json.dump(out_stats, f, indent=2, default=str)


if __name__ == '__main__':
    main()
