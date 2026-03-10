"""
show_data.py — Load and visualise the 2D Darcy flow dataset (GROUND TRUTH).

Dataset layout (after .T transpose in MatRead):
    a_field : (n_samples, s, s)  — permeability / diffusion coefficient a(x,y)
    u_field : (n_samples, s, s)  — pressure / solution field u(x,y)  ← ground truth

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

# Shared colour ranges per row, computed over the displayed samples
a_levels = np.linspace(a_train[:n_examples].min(), a_train[:n_examples].max(), 21)
u_levels = np.linspace(u_train[:n_examples].min(), u_train[:n_examples].max(), 21)

for i in range(n_examples):
    ax = axes[0, i]
    cf_a = ax.contourf(a_train[i], levels=a_levels, cmap='RdBu_r')
    ax.set_title(f'Sample {i+1}', fontsize=10)
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel('a(x,y)  [input]', fontsize=10)

    ax = axes[1, i]
    cf_u = ax.contourf(u_train[i], levels=u_levels, cmap='viridis')
    ax.set_title(f'Sample {i+1}', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel('u(x,y)  [ground truth]', fontsize=10)

# One shared colorbar per row, on the right
fig.colorbar(cf_a, ax=axes[0].tolist(), location='right', shrink=0.95, label='a(x,y)')
fig.colorbar(cf_u, ax=axes[1].tolist(), location='right', shrink=0.95, label='u(x,y)')

fig.suptitle('2D Darcy flow — ground truth\n'
             f'(training set, grid {a_train.shape[1]}\u00d7{a_train.shape[2]})',
             fontsize=12)
plt.savefig('darcy_data_preview.png', dpi=150, bbox_inches='tight')
print('\nFigure saved -> darcy_data_preview.png')
plt.show()
