"""
show_data.py — Load and visualise the 2D Darcy flow dataset.

Dataset layout (after .T transpose in MatRead):
    a_field : (n_samples, s, s)  — permeability / diffusion coefficient a(x,y)
    u_field : (n_samples, s, s)  — pressure / solution field u(x,y)

The PDE being solved is:
    -∇·(a(x,y) ∇u(x,y)) = f(x,y)   on [0,1]²
with homogeneous Dirichlet BCs. The operator learning task is to learn
the map  a  →  u  directly from data.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = 'Darcy_2D_data_train.mat'
TEST_PATH  = 'Darcy_2D_data_test.mat'

# ── Load ──────────────────────────────────────────────────────────────────────
def load(path):
    with h5py.File(path, 'r') as f:
        a = np.array(f['a_field']).T   # (n, s, s)
        u = np.array(f['u_field']).T   # (n, s, s)
    return a, u

a_train, u_train = load(TRAIN_PATH)
a_test,  u_test  = load(TEST_PATH)

# ── Explain dimensions ────────────────────────────────────────────────────────
print("=" * 55)
print("Dataset dimensions")
print("=" * 55)
print(f"  a_train : {a_train.shape}   (n_train, height, width)")
print(f"  u_train : {u_train.shape}")
print(f"  a_test  : {a_test.shape}    (n_test,  height, width)")
print(f"  u_test  : {u_test.shape}")
print()
print(f"  Grid resolution : {a_train.shape[1]} x {a_train.shape[2]}")
print(f"  Training samples: {a_train.shape[0]}")
print(f"  Test samples    : {a_test.shape[0]}")
print()
print("  a_field — input permeability field (piecewise constant, binary-ish)")
print(f"    range: [{a_train.min():.2f}, {a_train.max():.2f}]")
print(f"    mean : {a_train.mean():.4f}")
print()
print("  u_field — output pressure/solution field (smooth)")
print(f"    range: [{u_train.min():.4f}, {u_train.max():.4f}]")
print(f"    mean : {u_train.mean():.4f}")
print("=" * 55)

# ── Plot a few examples ───────────────────────────────────────────────────────
n_examples = 4
fig, axes = plt.subplots(2, n_examples, figsize=(3.5 * n_examples, 6))

for i in range(n_examples):
    # Input: permeability a(x,y)
    ax = axes[0, i]
    cf = ax.contourf(a_train[i], levels=20, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'Sample {i+1}  —  a(x,y)', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')

    # Output: pressure u(x,y)
    ax = axes[1, i]
    cf = ax.contourf(u_train[i], levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'Sample {i+1}  —  u(x,y)', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')

axes[0, 0].set_ylabel('y\n\na(x,y) — input', fontsize=10)
axes[1, 0].set_ylabel('y\n\nu(x,y) — output', fontsize=10)

fig.suptitle('2D Darcy flow: input permeability  →  output pressure\n'
             f'(training set, grid {a_train.shape[1]}×{a_train.shape[2]})',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('darcy_data_preview.png', dpi=150, bbox_inches='tight')
print('\nFigure saved -> darcy_data_preview.png')
plt.show()
