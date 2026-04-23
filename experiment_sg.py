"""Main experiment (surrogate-gradient variant).

For each of 5 architectures:
  1. Compute γ (Fiedler value of normalized Laplacian) at init.
  2. Train surrogate-gradient SNN on MNIST with Adam.
  3. Record convergence step (first eval where accuracy > 35%).
  4. At end, compute γ_final and Spearman correlation across configs.

Usage:
  python experiment_sg.py --quick    # 3 configs, 500 samples, 3-5 min
  python experiment_sg.py            # 5 configs, 2000 samples, ~15 min CPU
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snn_sg import SGSNN, poisson_spike_train
from cag_gamma import compute_gamma_for_snn


CONFIGS = [
    ('wide',        150, 150, 1.0),
    ('bottleneck',  300,  20, 1.0),
    ('balanced',    100, 100, 1.0),
    ('deep_narrow',  60,  60, 1.0),
    ('sparse',      200, 100, 0.5),
]


def get_loaders(n_train, n_test, batch_size):
    tfm = transforms.Compose([transforms.ToTensor(),
                               transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tfm)
    te = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    tr = torch.utils.data.Subset(tr, range(n_train))
    te = torch.utils.data.Subset(te, range(n_test))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(te, batch_size=batch_size))


@torch.no_grad()
def eval_acc(model, loader, T, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        st = poisson_spike_train(x, T=T)
        logits = model(st)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


def train_one(name, h1, h2, d, n_train, T, batch_size, device,
              convergence_threshold=0.35, lr=1e-3):
    torch.manual_seed(0)
    model = SGSNN(784, h1, h2, 10, density=d).to(device)
    gamma_init, _ = compute_gamma_for_snn(model)
    train_loader, test_loader = get_loaders(n_train, 500, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    sample_count = 0
    log = []
    convergence_step = None
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        st = poisson_spike_train(x, T=T)
        logits = model(st)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        sample_count += x.shape[0]
        if sample_count % (batch_size * 4) == 0 or sample_count >= n_train:
            acc = eval_acc(model, test_loader, T, device)
            log.append({'samples': sample_count, 'acc': acc, 'loss': float(loss.item())})
            if convergence_step is None and acc > convergence_threshold:
                convergence_step = sample_count
        if sample_count >= n_train:
            break
    final_acc = eval_acc(model, test_loader, T, device)
    gamma_final, _ = compute_gamma_for_snn(model)
    return {'name': name, 'n_h1': h1, 'n_h2': h2, 'density': d,
            'gamma_init': gamma_init, 'gamma_final': gamma_final,
            'convergence_step': convergence_step, 'final_acc': final_acc,
            'wall_s': time.time() - t0, 'log': log}


def analyze(results):
    from scipy.stats import spearmanr
    valid = [r for r in results if r['convergence_step'] is not None]
    print('\n=== Analysis ===')
    if len(valid) < 3:
        print(f'Only {len(valid)}/{len(results)} configs converged — cannot fit correlation.')
        return {'error': 'insufficient convergence', 'n_converged': len(valid)}
    gs = np.array([r['gamma_init'] for r in valid])
    ss = np.array([r['convergence_step'] for r in valid])
    rho, p = spearmanr(gs, ss)
    slope, intercept = np.polyfit(np.log(gs), np.log(ss), 1)
    pred = np.exp(slope * np.log(gs) + intercept)
    ss_tot = ((np.log(ss) - np.log(ss).mean()) ** 2).sum()
    ss_res = ((np.log(ss) - np.log(pred)) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    print(f'n_converged = {len(valid)}/{len(results)}')
    print(f'γ values: {gs.tolist()}')
    print(f'convergence_step values: {ss.tolist()}')
    print(f'Spearman ρ = {rho:+.3f} (p={p:.3f})')
    print(f'log-log fit: log(step) = {intercept:+.3f} + ({slope:+.3f}) · log(γ)')
    print(f'R² (log) = {r2:.3f}')
    if rho < -0.7 and r2 > 0.6:
        verdict = 'STRONG GO'
    elif rho < -0.7:
        verdict = 'GO'
    elif rho < -0.3:
        verdict = 'PARTIAL'
    else:
        verdict = 'NO-GO'
    print(f'Verdict: {verdict}')
    return {'rho': rho, 'p': p, 'slope': slope, 'r2': r2, 'verdict': verdict,
            'n_converged': len(valid), 'n_total': len(results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='./results_sg.json')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.quick:
        configs = CONFIGS[:3]; n_train, T = 500, 20
    else:
        configs = CONFIGS; n_train, T = 2000, 30

    results = []
    for name, h1, h2, d in configs:
        print(f'\n[{name}] H1={h1} H2={h2} density={d}')
        r = train_one(name, h1, h2, d, n_train, T, 32, device)
        print(f'  γ_init={r["gamma_init"]:.4f}  γ_final={r["gamma_final"]:.4f}  '
              f'conv_step={r["convergence_step"]}  final_acc={r["final_acc"]:.3f}  '
              f'wall={r["wall_s"]:.1f}s')
        results.append(r)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    stats = analyze(results)
    with open(args.out.replace('.json', '_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)


if __name__ == '__main__':
    main()
