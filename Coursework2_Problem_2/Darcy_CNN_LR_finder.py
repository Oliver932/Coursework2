"""
Leslie Smith's LR Range Test for Darcy_CNN.
Exponentially increases the learning rate over one sweep of batches,
recording the loss at each step. Plot loss vs LR to identify the
steepest downward slope — use that LR (or slightly below) for training.
"""

import copy
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data

from Darcy_CNN import LpLoss, MatRead, UnitGaussianNormalizer, CNN

# ─── Hyperparameters ───────────────────────────────────────────────────────────
WIDTH      = 4       # must match the model you intend to train
LR_START   = 1e-7
LR_END     = 1.0
NUM_ITER   = 200     # number of batches to sweep over
BATCH_SIZE = 20
SMOOTHING  = 0.9     # exponential moving average smoothing factor
DIVERGE_THRESHOLD = 4.0   # stop if loss > DIVERGE_THRESHOLD * best_loss
# ───────────────────────────────────────────────────────────────────────────────


def lr_finder(net, train_loader, loss_func, u_normalizer,
              lr_start=LR_START, lr_end=LR_END,
              num_iter=NUM_ITER, smoothing=SMOOTHING,
              diverge_threshold=DIVERGE_THRESHOLD):
    """Run Leslie Smith's LR range test.

    Returns:
        lrs        : list of learning rates tested
        losses     : corresponding smoothed losses
        steep_idx  : index of steepest loss descent
    """
    # Work on a fresh copy so the original model weights are unchanged
    net = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_start)

    mult = (lr_end / lr_start) ** (1.0 / num_iter)

    lrs, losses = [], []
    avg_loss, best_loss = 0.0, float('inf')
    lr = lr_start

    # Cycle through batches, restarting the loader if needed
    loader_iter = iter(train_loader)
    net.train()
    for step in range(num_iter):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)

        outputs = u_normalizer.decode(net(inputs))
        loss = loss_func(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Exponential moving average smoothing
        avg_loss = smoothing * avg_loss + (1 - smoothing) * loss.item()
        smoothed = avg_loss / (1 - smoothing ** (step + 1))   # bias correction

        lrs.append(lr)
        losses.append(smoothed)

        if smoothed < best_loss:
            best_loss = smoothed
        if smoothed > diverge_threshold * best_loss:
            print(f"Loss diverged at step {step}, lr={lr:.2e} — stopping.")
            break

        # Update LR for next step
        lr *= mult
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    return lrs, losses


def plot_lr_finder(lrs, losses, save_path='cnn_lr_finder.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate (log scale)')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title("LR Range Test (Leslie Smith's method)")
    ax.grid(which='both', linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'LR finder plot saved -> {save_path}')
    plt.show()


if __name__ == '__main__':
    # ── Load data ────────────────────────────────────────────────────────────
    data_reader = MatRead('Darcy_2D_data_train.mat')
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)

    u_normalizer = UnitGaussianNormalizer(u_train)

    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, BATCH_SIZE, shuffle=True)

    # ── Build untrained model ─────────────────────────────────────────────────
    net = CNN(width=WIDTH)
    loss_func = LpLoss()

    print(f"Running LR finder: lr {LR_START:.0e} → {LR_END:.0e} over {NUM_ITER} iterations...")
    lrs, losses = lr_finder(net, train_loader, loss_func, u_normalizer)
    plot_lr_finder(lrs, losses)
