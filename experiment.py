"""Main experiment: measure γ ↔ STDP convergence correlation across 5 architectures.

For each config: (i) compute γ at initialization, (ii) train with STDP on MNIST
(unsupervised hidden, supervised readout), (iii) measure convergence step
(first training-sample index where held-out accuracy > 35%).

Output: a table + JSON with (name, gamma, convergence_step) per config,
and a final Spearman + log-log fit across configs.

Usage:
  python experiment.py --quick     # 3 configs, 300 samples, ~3 min
  python experiment.py             # full 5 configs, 1000 samples, ~15 min CPU / ~3 min GPU
"""
import os
import json
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snn import SNN, poisson_spike_train
from cag_gamma import compute_gamma_for_snn


CONFIGS = [
    # name, n_h1, n_h2, density
    ('wide',        150, 150, 1.0),
    ('bottleneck',  300,  20, 1.0),
    ('balanced',    100, 100, 1.0),
    ('deep_narrow',  60,  60, 1.0),
    ('sparse',      200, 100, 0.5),
]


def get_mnist_loaders(n_train: int, n_test: int, batch_size: int, synthetic: bool = False):
    if synthetic:
        # Synthetic fallback: random digit-class-conditioned images
        rng = np.random.default_rng(0)
        train_x = torch.rand(n_train, 784)
        train_y = torch.randint(0, 10, (n_train,))
        test_x = torch.rand(n_test, 784)
        test_y = torch.randint(0, 10, (n_test,))
        # Add class-conditional structure so there's SOMETHING to learn
        for i in range(n_train):
            c = int(train_y[i])
            train_x[i, c * 78:(c + 1) * 78] += 0.3
        for i in range(n_test):
            c = int(test_y[i])
            test_x[i, c * 78:(c + 1) * 78] += 0.3
        train_x = train_x.clamp(0, 1); test_x = test_x.clamp(0, 1)
        class TinyDS(torch.utils.data.Dataset):
            def __init__(self, x, y): self.x, self.y = x, y
            def __len__(self): return len(self.x)
            def __getitem__(self, i): return self.x[i], self.y[i]
        return (DataLoader(TinyDS(train_x, train_y), batch_size=batch_size, shuffle=True),
                DataLoader(TinyDS(test_x, test_y), batch_size=batch_size))
    else:
        tfm = transforms.Compose([transforms.ToTensor(),
                                   transforms.Lambda(lambda x: x.view(-1))])
        try:
            tr = datasets.MNIST('./data', train=True, download=True, transform=tfm)
            te = datasets.MNIST('./data', train=False, download=True, transform=tfm)
        except Exception as e:
            print(f'MNIST download failed ({e}), using synthetic.')
            return get_mnist_loaders(n_train, n_test, batch_size, synthetic=True)
        # Subset
        tr = torch.utils.data.Subset(tr, range(n_train))
        te = torch.utils.data.Subset(te, range(n_test))
        return (DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(te, batch_size=batch_size))


@torch.no_grad()
def eval_accuracy(model: SNN, loader, T: int, device):
    model.eval()
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        st = poisson_spike_train(x, T=T)
        logits, _ = model.forward_time(st, stdp=False)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / max(total, 1)


def train_config(name, n_h1, n_h2, density, n_train, n_test, T, batch_size,
                 device, lr_readout=1e-2, convergence_threshold=0.35):
    torch.manual_seed(0)
    model = SNN(n_in=784, n_h1=n_h1, n_h2=n_h2, n_out=10, density=density).to(device)
    # γ at init
    gamma_init, lam1 = compute_gamma_for_snn(model)

    train_loader, test_loader = get_mnist_loaders(n_train, n_test, batch_size)

    # Readout optimizer — supervised on cum_h2 spikes
    opt = torch.optim.Adam(model.readout.parameters(), lr=lr_readout)

    sample_count = 0
    convergence_step = None
    log = []
    t0 = time.time()
    for epoch in range(1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            st = poisson_spike_train(x, T=T)
            # Forward with STDP on hidden layers
            logits, _ = model.forward_time(st, stdp=True)
            # Readout supervised loss
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            sample_count += x.shape[0]
            # Periodically eval
            if sample_count % (batch_size * 4) == 0:
                acc = eval_accuracy(model, test_loader, T, device)
                log.append({'samples': sample_count, 'acc': acc,
                             'loss': float(loss.item())})
                if convergence_step is None and acc > convergence_threshold:
                    convergence_step = sample_count
                if sample_count >= n_train:
                    break
        if sample_count >= n_train:
            break
    final_acc = eval_accuracy(model, test_loader, T, device)
    gamma_final, _ = compute_gamma_for_snn(model)
    return {
        'name': name, 'n_h1': n_h1, 'n_h2': n_h2, 'density': density,
        'gamma_init': gamma_init, 'gamma_final': gamma_final,
        'convergence_step': convergence_step, 'final_acc': final_acc,
        'wall_s': time.time() - t0, 'log': log,
    }


def analyze(results):
    from scipy.stats import spearmanr
    valid = [r for r in results if r['convergence_step'] is not None]
    if len(valid) < 3:
        print('Too few converged configs for correlation.')
        return
    gs = np.array([r['gamma_init'] for r in valid])
    ss = np.array([r['convergence_step'] for r in valid])
    rho, p = spearmanr(gs, ss)
    # Log-log fit: log(step) = a - b log(gamma)  (we expect b > 0)
    slope, intercept = np.polyfit(np.log(gs), np.log(ss), 1)
    pred = np.exp(slope * np.log(gs) + intercept)
    ss_res = ((np.log(ss) - np.log(pred)) ** 2).sum()
    ss_tot = ((np.log(ss) - np.log(ss).mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    print('\n=== Analysis ===')
    print(f'Spearman ρ (γ, convergence_step) = {rho:+.3f} (p={p:.3f})')
    print(f'Log-log fit: log(step) = {intercept:+.3f} + ({slope:+.3f}) · log(γ)')
    print(f'R² = {r2:.3f}')
    if rho < -0.7 and r2 > 0.6:
        verdict = 'STRONG GO'
    elif rho < -0.7:
        verdict = 'GO'
    elif rho < -0.3:
        verdict = 'PARTIAL'
    else:
        verdict = 'NO-GO'
    print(f'Verdict: {verdict}')
    return {'rho': rho, 'p': p, 'slope': slope, 'r2': r2, 'verdict': verdict}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    p.add_argument('--out', default='./results.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.quick:
        configs = CONFIGS[:3]; n_train, n_test, T = 300, 100, 30
    else:
        configs = CONFIGS; n_train, n_test, T = 1000, 200, 50

    results = []
    for name, h1, h2, d in configs:
        print(f'\n[{name}] H1={h1} H2={h2} density={d}')
        r = train_config(name, h1, h2, d, n_train, n_test, T, batch_size=32,
                          device=device)
        print(f'  γ_init={r["gamma_init"]:.4f}  γ_final={r["gamma_final"]:.4f}  '
              f'conv_step={r["convergence_step"]}  final_acc={r["final_acc"]:.3f}  '
              f'wall={r["wall_s"]:.1f}s')
        results.append(r)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    stats = analyze(results)
    if stats:
        with open(args.out.replace('.json', '_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
