"""Compute CAG spectral gap γ of a feedforward SNN's Hasse diagram.

γ := 1 − |λ₂| of the random walk on the symmetrized synapse-weighted graph.

For a 3-layer SNN (input → H1 → H2 → output):
  - Nodes: all neurons (n_in + n_h1 + n_h2 + n_out)
  - Directed edges with weights |W_{l,ij}| for each layer l
  - Symmetrize: Ã = (A + Aᵀ) / 2 (undirected weighted graph)
  - Row-normalize to Markov chain M
  - γ = 1 − |λ₂| of M
"""
import torch
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import scipy.linalg as sla


def build_adjacency(n_in: int, n_h1: int, n_h2: int, n_out: int,
                     W1: torch.Tensor, W2: torch.Tensor,
                     W_read: torch.Tensor) -> np.ndarray:
    """Build the (n_in + n_h1 + n_h2 + n_out)² adjacency matrix."""
    N = n_in + n_h1 + n_h2 + n_out
    A = np.zeros((N, N), dtype=np.float64)
    # W1: (n_h1, n_in)  -- edges input[i] → h1[j]
    i0 = 0
    j0 = n_in
    A[i0:i0 + n_in, j0:j0 + n_h1] = W1.detach().cpu().numpy().T
    # W2: (n_h2, n_h1)
    i0 = n_in
    j0 = n_in + n_h1
    A[i0:i0 + n_h1, j0:j0 + n_h2] = W2.detach().cpu().numpy().T
    # W_read: (n_out, n_h2)
    i0 = n_in + n_h1
    j0 = n_in + n_h1 + n_h2
    A[i0:i0 + n_h2, j0:j0 + n_out] = W_read.detach().cpu().numpy().T
    # Take abs values (edge magnitudes only)
    return np.abs(A)


def spectral_gap(A: np.ndarray) -> float:
    """γ := second-smallest eigenvalue (Fiedler value) of the normalized Laplacian
          L_norm = I − D^{-1/2} Ã D^{-1/2} where Ã = (A + Aᵀ)/2.

    This is the canonical expansion/connectivity measure; unlike random-walk |λ₂|
    it's always nonzero for a connected graph (no bipartite degeneracy).
    Larger γ ⇒ better connectivity/mixing.
    """
    Ã = 0.5 * (A + A.T)
    d = Ã.sum(axis=1)
    # For disconnected nodes, use degree 1 to avoid div-by-zero (they'll show up
    # as 0 eigenvalues anyway).
    d_safe = np.where(d > 1e-12, d, 1.0)
    D_inv_sqrt = 1.0 / np.sqrt(d_safe)
    L_norm = np.eye(Ã.shape[0]) - Ã * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
    # Eigenvalues in [0, 2]; smallest is 0 (for a connected component);
    # second-smallest is the Fiedler value = algebraic connectivity.
    vals = sla.eigvalsh(L_norm)      # hermitian eigenvalues, ascending
    lam_small = vals[0]
    lam_fiedler = vals[1]
    return float(lam_fiedler), float(lam_small)


def compute_gamma_for_snn(snn):
    # Use masked W if present (for sparse density models)
    W1 = snn.l1.masked_W() if hasattr(snn.l1, 'masked_W') else snn.l1.W
    W2 = snn.l2.masked_W() if hasattr(snn.l2, 'masked_W') else snn.l2.W
    W_read = snn.readout.weight
    A = build_adjacency(snn.l1.n_in, snn.l1.n_out, snn.l2.n_out,
                         snn.readout.out_features, W1, W2, W_read)
    gamma_fiedler, lam_smallest = spectral_gap(A)
    return gamma_fiedler, lam_smallest


if __name__ == '__main__':
    # Sanity: compute γ for a few test networks
    import torch
    from snn import SNN
    torch.manual_seed(0)

    configs = [
        ('wide',       150, 150, 1.0),
        ('bottleneck', 300, 20,  1.0),
        ('balanced',   100, 100, 1.0),
        ('deep-narrow', 60, 60,  1.0),
        ('sparse',     200, 100, 0.5),
    ]
    print(f'{"name":<14} {"H1":>4} {"H2":>4} {"density":>8} {"γ":>10} {"λ₁":>10}')
    for name, h1, h2, d in configs:
        torch.manual_seed(0)
        m = SNN(n_in=784, n_h1=h1, n_h2=h2, n_out=10, density=d)
        g, l1 = compute_gamma_for_snn(m)
        print(f'{name:<14} {h1:>4} {h2:>4} {d:>8.2f} {g:>10.4f} {l1:>10.4f}')
