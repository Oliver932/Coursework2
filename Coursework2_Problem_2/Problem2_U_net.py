import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import h5py

# Loss function: Binary Cross-Entropy for survive/fail classification
class Lossfunc(object):
    def __init__(self):
        self._bce = nn.BCELoss()

    def __call__(self, prediction, target):
        return self._bce(prediction, target)

# Data reader for Eiffel_data.mat (HDF5)
class MatRead(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')

    def get_inputs(self):
        """Returns tensor [n_samples, 20] – spatial pressure load profiles."""
        # stored as (20, 1000) -> transpose to (1000, 20)
        x = np.array(self.data['load_apply']).T
        return torch.tensor(x, dtype=torch.float32)

    def get_labels(self):
        """Returns tensor [n_samples, 1] – 0=fail, 1=survive."""
        # stored as (1, 1000) -> transpose to (1000, 1)
        y = np.array(self.data['result']).T
        return torch.tensor(y, dtype=torch.float32)

# Input normaliser (z-score, fitted on training data)
class DataNormalizer(object):
    def __init__(self, data):
        self.mean = data.mean(dim=0)
        self.std  = data.std(dim=0, unbiased=False)
        self.std[self.std == 0] = 1.0

    def encode(self, x):
        return (x - self.mean) / self.std


# 1-D Residual convolution block used in encoder, bottleneck, and decoder.
#
# Main path : Conv1d(k=3,pad=1,bias=F) → BN → ReLU → Conv1d(k=3,pad=1,bias=F) → BN
# Shortcut  : Conv1d(k=1,bias=F) → BN  (always a learned projection — handles any channel change)
# Output    : F.relu(main(x) + shortcut(x))
class ResConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        # Always project: aligns channels and lets the network learn the best skip scaling.
        # bias=False — BN's learnable beta subsumes any constant offset.
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return F.relu(self.main(x) + self.shortcut(x))


# 1-D U-Net classifier with residual conv blocks.
class Classifier_Net(nn.Module):

    def __init__(self, input_dim=20, base_channels=16):
        """
        Args:
            input_dim     : spatial length of the pressure profile (20 for Eiffel data)
            base_channels : number of channels at the shallowest encoder level (c);
                            encoder uses c, 2c; bottleneck uses 4c
        """
        super().__init__()
        c = base_channels

        # ── Encoder: two levels, each doubles channels and halves spatial dim ──
        self.enc0 = ResConvBlock(1, c)       # (batch, 1, 20) → (batch,  c, 20)
        self.enc1 = ResConvBlock(c, 2*c)     # (batch, c, 10) → (batch, 2c, 10)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # stateless, shared

        # ── Bottleneck: deepest representation at spatial length 5 ─────────────
        self.bottleneck = ResConvBlock(2*c, 4*c)  # (batch, 2c, 5) → (batch, 4c, 5)

        # ── Decoder: upsample, concat matching skip, residual-convolve ──────────
        # up1 receives bottleneck output (4c) and concatenates skip1 (2c) → 4c total in
        self.up1  = nn.ConvTranspose1d(4*c, 2*c, kernel_size=2, stride=2)
        self.dec1 = ResConvBlock(4*c, 2*c)   # (batch, 4c, 10) → (batch, 2c, 10)

        # up0 receives dec1 output (2c) and concatenates skip0 (c) → 2c total in
        self.up0  = nn.ConvTranspose1d(2*c, c, kernel_size=2, stride=2)
        self.dec0 = ResConvBlock(2*c, c)     # (batch, 2c, 20) → (batch,  c, 20)

        # ── Head: global average pool then a single linear projection ───────────
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (batch, c, 20) → (batch, c, 1)
        self.classifier  = nn.Linear(c, 1)           # (batch, c)    → (batch, 1)

    @staticmethod
    def _cat(upsampled, skip):
        """Pad upsampled by 1 on the right if odd pooling caused a length shortfall, then cat."""
        if upsampled.shape[-1] < skip.shape[-1]:
            upsampled = F.pad(upsampled, (0, skip.shape[-1] - upsampled.shape[-1]))
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)               # (batch, 1, 20)

        # Encoder — save feature maps before pooling for skip connections
        s0 = self.enc0(x)                       # (batch,  c, 20)
        s1 = self.enc1(self.pool(s0))            # (batch, 2c, 10)

        # Bottleneck
        x = self.bottleneck(self.pool(s1))      # (batch, 4c,  5)

        # Decoder — upsample, prepend skip (channel-wise cat), then residual-convolve
        x = self.dec1(self._cat(self.up1(x), s1))  # (batch, 2c, 10)
        x = self.dec0(self._cat(self.up0(x), s0))  # (batch,  c, 20)

        # Head — pool spatially, project to scalar, squash to probability
        x = self.global_pool(x).squeeze(-1)     # (batch,  c)
        return torch.sigmoid(self.classifier(x))  # (batch,  1)

######################### Data Preparation #############################
def prepare_data(data_path='../Data/Eiffel_data.mat', train_split=0.8):
    """
    Load Eiffel_data.mat and split into normalised train / test sets.

    Returns dict with keys:
        train_inputs, test_inputs   normalised tensors [n, 20]
        train_labels, test_labels   binary label tensors [n, 1]
    """
    reader = MatRead(data_path)
    inputs = reader.get_inputs()   # (1000, 20)
    labels = reader.get_labels()   # (1000,  1)

    n_samples = inputs.shape[0]
    ntrain    = int(train_split * n_samples)
    ntest     = n_samples - ntrain

    print(f"Dataset: {n_samples} samples | input dim = {inputs.shape[1]}")
    print(f"  Train: {ntrain}  |  Test: {ntest}")
    print(f"  Labels  0 (fail): {int((labels==0).sum())}  "
          f"1 (survive): {int((labels==1).sum())}")

    # Shuffle with fixed seed for reproducibility
    torch.manual_seed(42)
    perm   = torch.randperm(n_samples)
    inputs = inputs[perm]
    labels = labels[perm]

    train_inputs_raw = inputs[:ntrain]
    train_labels     = labels[:ntrain]
    test_labels      = labels[ntrain:]
    test_inputs_raw  = inputs[ntrain:]

    # Normalise inputs only (labels are already 0/1)
    normalizer   = DataNormalizer(train_inputs_raw)
    train_inputs = normalizer.encode(train_inputs_raw)
    test_inputs  = normalizer.encode(test_inputs_raw)

    print(f"\nNormaliser mean: {normalizer.mean.numpy().round(3)}")
    print(f"Normaliser std : {normalizer.std.numpy().round(3)}")

    return {
        'train_inputs':    train_inputs,
        'train_labels':    train_labels,
        'test_inputs':     test_inputs,
        'test_labels':     test_labels,
    }

######################### Training Function #############################
# Also callable from an Optuna tuner script
def train_classification_model(
    data_dict,
    base_channels,
    learning_rate,
    weight_decay=1e-4,
    batch_size=32,
    epochs=60,
    trial=None,
):
    """
    Train the 1D residual U-Net classifier (fixed 3-level depth).

    Args:
        data_dict     : dict returned by prepare_data()
        base_channels : channels at the shallowest encoder level
        learning_rate : Adam learning rate
        weight_decay  : Adam L2 penalty (regularises conv weights on the small dataset)
        batch_size    : mini-batch size
        epochs        : number of training epochs
        trial         : Optuna trial object (optional, for pruning)

    Returns: net, train_losses, test_losses, test_accs, final_test_loss
    """
    train_inputs = data_dict['train_inputs']
    train_labels = data_dict['train_labels']
    test_inputs  = data_dict['test_inputs']
    test_labels  = data_dict['test_labels']
    input_dim    = train_inputs.shape[1]

    train_set    = Data.TensorDataset(train_inputs, train_labels)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    net       = Classifier_Net(input_dim=input_dim, base_channels=base_channels)
    loss_func = Lossfunc()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params  = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Network: base_channels={base_channels} (2 levels) → {n_params} parameters')
    print(f'Training for {epochs} epochs (batch size {batch_size})...\n')

    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            pred  = net(x_batch)
            loss  = loss_func(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        net.eval()
        with torch.no_grad():
            test_pred  = net(test_inputs)
            test_loss  = loss_func(test_pred, test_labels).item()
            test_class = (test_pred >= 0.5).float()
            test_acc   = (test_class == test_labels).float().mean().item()

        train_losses.append(epoch_loss / len(train_loader))
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch:3d}: train loss = {train_losses[-1]:.4f}, '
                  f'test loss = {test_loss:.4f}, test acc = {test_acc*100:.1f}%')

        if trial is not None:
            trial.report(test_loss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    print(f'\nFinal test accuracy : {test_accs[-1]*100:.1f}%')
    print(f'Final test BCE loss : {test_losses[-1]:.4f}')

    return net, train_losses, test_losses, test_accs, test_losses[-1]

######################### Plotting Functions #############################
def plot_loss_curves(train_losses, test_losses, save_path='../Figures_2/loss_plot.png'):
    epochs = len(train_losses)
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_losses, label='Train loss', linewidth=2)
    plt.plot(range(epochs), test_losses,  label='Test loss',  linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Training and Test Classification Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Loss plot saved -> {save_path}')
    plt.show()


def plot_accuracy_curve(test_accs, save_path='../Figures_2/accuracy_plot.png'):
    epochs = len(test_accs)
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), [a * 100 for a in test_accs], linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy over Training')
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Accuracy plot saved -> {save_path}')
    plt.show()


def plot_confusion(net, data_dict, save_path='../Figures_2/confusion_matrix.png'):
    """Plot a 2x2 confusion matrix on the test set."""
    test_inputs = data_dict['test_inputs']
    test_labels = data_dict['test_labels']

    net.eval()
    with torch.no_grad():
        probs = net(test_inputs)
        preds = (probs >= 0.5).float().squeeze().numpy()
    true = test_labels.squeeze().numpy()

    tp = int(((preds == 1) & (true == 1)).sum())
    tn = int(((preds == 0) & (true == 0)).sum())
    fp = int(((preds == 1) & (true == 0)).sum())
    fn = int(((preds == 0) & (true == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Fail', 'Pred: Survive'])
    ax.set_yticklabels(['True: Fail', 'True: Survive'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix (Test Set)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Confusion matrix saved -> {save_path}')
    plt.show()


######################### Main execution #############################
if __name__ == "__main__":

    # Step 1: Load data
    print("=" * 60)
    print("Step 1: Preparing data")
    print("=" * 60)
    data_dict = prepare_data(data_path='../Data/Eiffel_data.mat', train_split=0.8)

    torch.manual_seed(42)
    np.random.seed(42)

    # Step 2: Train
    print("\n" + "=" * 60)
    print("Step 2: Training classification model")
    print("=" * 60)
    net, train_losses, test_losses, test_accs, final_loss = train_classification_model(
        data_dict=data_dict,
        base_channels=16,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    # Step 3: Plot results
    print("\n" + "=" * 60)
    print("Step 3: Plotting results")
    print("=" * 60)
    plot_loss_curves(train_losses, test_losses)
    plot_accuracy_curve(test_accs)
    plot_confusion(net, data_dict)
