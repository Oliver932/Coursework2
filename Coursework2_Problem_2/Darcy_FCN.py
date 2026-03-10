import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py

# Define Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)

# Define normalizer, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

# Define network — simple Fully Convolutional Network
# Architecture: lift to w channels, apply n_layers conv blocks (k=3, pad=1, spatial dims preserved),
# then project back to 1 channel with a 1x1 conv.
# Each block: Conv2d -> BN -> ReLU
class FCN(nn.Module):
    def __init__(self, width=64, n_layers=4, kernel_size=3):
        super(FCN, self).__init__()
        # padding = kernel_size // 2 keeps spatial dimensions unchanged for any odd kernel
        padding = kernel_size // 2

        layers = []
        # Lift: 1 -> width
        layers += [nn.Conv2d(1, width, kernel_size=kernel_size, padding=padding),
                   nn.BatchNorm2d(width),
                   nn.ReLU()]
        # Hidden layers: width -> width, spatial size unchanged
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(width, width, kernel_size=kernel_size, padding=padding),
                       nn.BatchNorm2d(width),
                       nn.ReLU()]
        # Project back: width -> 1
        layers += [nn.Conv2d(width, 1, kernel_size=1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = out.squeeze(1)
        return out

######################### Data Preparation #############################
def prepare_data(train_path='Darcy_2D_data_train.mat', test_path='Darcy_2D_data_test.mat'):
    """Load and normalise the Darcy 2D dataset.

    Returns dict with keys:
        a_train, a_test   — normalised input tensors
        u_train, u_test   — unnormalised output tensors (ground truth)
        a_normalizer, u_normalizer — UnitGaussianNormalizer objects
    """
    data_reader = MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = UnitGaussianNormalizer(u_train)

    print(f"a_train: {a_train.shape},  a_test: {a_test.shape}")
    print(f"u_train: {u_train.shape},  u_test: {u_test.shape}")

    return {
        'a_train': a_train, 'a_test': a_test,
        'u_train': u_train, 'u_test': u_test,
        'a_normalizer': a_normalizer, 'u_normalizer': u_normalizer,
    }

######################### Training Function #############################
def train_fcn_model(
    data_dict,
    width,
    n_layers,
    kernel_size,
    learning_rate,
    batch_size=20,
    epochs=60,
    trial=None,
):
    """Train the FCN operator-learning model.

    Args:
        data_dict     : dict returned by prepare_data()
        width         : channel width of the FCN
        n_layers      : number of conv layers
        kernel_size   : kernel size (odd integer)
        learning_rate : Adam learning rate
        batch_size    : mini-batch size
        epochs        : number of training epochs
        trial         : optional Optuna trial for pruning

    Returns: net, loss_train_list, loss_test_list, final_test_loss
    """
    a_train = data_dict['a_train']
    a_test  = data_dict['a_test']
    u_train = data_dict['u_train']
    u_test  = data_dict['u_test']
    u_normalizer = data_dict['u_normalizer']

    train_set    = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    net = FCN(width=width, n_layers=n_layers, kernel_size=kernel_size)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'FCN: width={width}, n_layers={n_layers}, kernel_size={kernel_size} -> {n_params} parameters')

    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Start training FCN for {epochs} epochs...")
    start_time = time()

    loss_train_list = []
    loss_test_list = []
    for epoch in range(epochs):
        net.train(True)
        trainloss = 0
        for i, data in enumerate(train_loader):
            input, target = data
            output = net(input)
            output = u_normalizer.decode(output)
            l = loss_func(output, target)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()

            trainloss += l.item()

        net.eval()
        with torch.no_grad():
            test_output = net(a_test)
            test_output = u_normalizer.decode(test_output)
            testloss = loss_func(test_output, u_test).item()

        if epoch % 10 == 0:
            print(f"  epoch:{epoch}, train loss:{trainloss/len(train_loader):.6f}, test loss:{testloss:.6f}")

        loss_train_list.append(trainloss / len(train_loader))
        loss_test_list.append(testloss)

        if trial is not None:
            trial.report(testloss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    total_time = time() - start_time
    print(f'Training time: {str(datetime.timedelta(seconds=int(total_time)))}')
    print(f"Train loss: {loss_train_list[-1]:.6f}")
    print(f"Test loss:  {loss_test_list[-1]:.6f}")

    return net, loss_train_list, loss_test_list, loss_test_list[-1]

######################### Plotting Functions #############################
def plot_loss_curves(loss_train_list, loss_test_list, save_path='fcn_loss_plot.png'):
    epochs = len(loss_train_list)
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), loss_train_list, label='Train loss')
    plt.plot(range(epochs), loss_test_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.05)
    plt.legend()
    plt.grid()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Loss plot saved -> {save_path}')
    plt.show()


def plot_contour_physical(net, data_dict, n_examples=4, save_path='fcn_output_preview.png', title_extra=''):
    """3-row contour in physical space: input / ground truth / prediction."""
    a_test = data_dict['a_test']
    u_test = data_dict['u_test']
    u_normalizer = data_dict['u_normalizer']

    net.eval()
    with torch.no_grad():
        pred = u_normalizer.decode(net(a_test))

    a_levels = np.linspace(a_test[:n_examples].min(), a_test[:n_examples].max(), 21)
    u_all = np.concatenate([u_test[:n_examples].numpy(), pred[:n_examples].numpy()])
    u_levels = np.linspace(u_all.min(), u_all.max(), 21)

    fig, axes = plt.subplots(3, n_examples, figsize=(3.5 * n_examples, 8))

    for i in range(n_examples):
        ax = axes[0, i]
        cf_a = ax.contourf(a_test[i].numpy(), levels=a_levels, cmap='RdBu_r')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('a(x,y)  [input]', fontsize=10)

        ax = axes[1, i]
        cf_u = ax.contourf(u_test[i].numpy(), levels=u_levels, cmap='viridis')
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('u(x,y)  [truth]', fontsize=10)

        ax = axes[2, i]
        cf_p = ax.contourf(pred[i].numpy(), levels=u_levels, cmap='viridis')
        ax.set_xlabel('x', fontsize=9)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('\u00fb(x,y)  [FCN]', fontsize=10)

    fig.colorbar(cf_a, ax=axes[0].tolist(), location='right', shrink=0.95, label='a(x,y)')
    fig.colorbar(cf_u, ax=axes[1].tolist(), location='right', shrink=0.95, label='u(x,y)')
    fig.colorbar(cf_p, ax=axes[2].tolist(), location='right', shrink=0.95, label='\u00fb(x,y)')

    fig.suptitle(f'FCN: input / truth / prediction  (test set){title_extra}', fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved -> {save_path}')
    plt.show()


def plot_contour_normalised(net, data_dict, n_examples=4, save_path='fcn_output_normalised.png', title_extra=''):
    """3-row contour in normalised space: input / ground truth / prediction."""
    a_test = data_dict['a_test']
    u_test = data_dict['u_test']
    u_normalizer = data_dict['u_normalizer']

    net.eval()
    with torch.no_grad():
        pred_norm = net(a_test)

    u_test_norm = u_normalizer.encode(u_test)

    a_lev = np.linspace(a_test[:n_examples].min(), a_test[:n_examples].max(), 21)
    u_all_n = np.concatenate([u_test_norm[:n_examples].numpy(), pred_norm[:n_examples].numpy()])
    u_lev = np.linspace(u_all_n.min(), u_all_n.max(), 21)

    fig, axes = plt.subplots(3, n_examples, figsize=(3.5 * n_examples, 8))

    for i in range(n_examples):
        ax = axes[0, i]
        cf_a = ax.contourf(a_test[i].numpy(), levels=a_lev, cmap='RdBu_r')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('a(x,y)  [input, norm]', fontsize=10)

        ax = axes[1, i]
        cf_u = ax.contourf(u_test_norm[i].numpy(), levels=u_lev, cmap='viridis')
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('u(x,y)  [truth, norm]', fontsize=10)

        ax = axes[2, i]
        cf_p = ax.contourf(pred_norm[i].numpy(), levels=u_lev, cmap='viridis')
        ax.set_xlabel('x', fontsize=9)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('\u00fb(x,y)  [FCN, norm]', fontsize=10)

    fig.colorbar(cf_a, ax=axes[0].tolist(), location='right', shrink=0.95, label='a(x,y)')
    fig.colorbar(cf_u, ax=axes[1].tolist(), location='right', shrink=0.95, label='u(x,y)')
    fig.colorbar(cf_p, ax=axes[2].tolist(), location='right', shrink=0.95, label='\u00fb(x,y)')

    fig.suptitle(f'FCN: input / truth / prediction  (test set, norm){title_extra}', fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved -> {save_path}')
    plt.show()


def plot_contour_comparison(net, data_dict, n_examples=4, save_path='fcn_output_error.png', title_extra=''):
    """4-row contour: input / ground truth / prediction / error."""
    a_test = data_dict['a_test']
    u_test = data_dict['u_test']
    u_normalizer = data_dict['u_normalizer']

    net.eval()
    with torch.no_grad():
        pred = u_normalizer.decode(net(a_test))

    a_levels = np.linspace(a_test[:n_examples].min(), a_test[:n_examples].max(), 21)
    u_all = np.concatenate([u_test[:n_examples].numpy(), pred[:n_examples].numpy()])
    u_levels = np.linspace(u_all.min(), u_all.max(), 21)
    diff = u_test[:n_examples].numpy() - pred[:n_examples].numpy()
    diff_abs_max = np.abs(diff).max()
    diff_levels = np.linspace(-diff_abs_max, diff_abs_max, 21)

    fig, axes = plt.subplots(4, n_examples, figsize=(3.5 * n_examples, 10))

    for i in range(n_examples):
        ax = axes[0, i]
        cf_a = ax.contourf(a_test[i].numpy(), levels=a_levels, cmap='RdBu_r')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('a(x,y)  [input]', fontsize=10)

        ax = axes[1, i]
        cf_u = ax.contourf(u_test[i].numpy(), levels=u_levels, cmap='viridis')
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('u(x,y)  [truth]', fontsize=10)

        ax = axes[2, i]
        cf_p = ax.contourf(pred[i].numpy(), levels=u_levels, cmap='viridis')
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('\u00fb(x,y)  [FCN]', fontsize=10)

        ax = axes[3, i]
        cf_d = ax.contourf(diff[i], levels=diff_levels, cmap='RdBu_r')
        ax.set_xlabel('x', fontsize=9)
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('u \u2212 \u00fb  [error]', fontsize=10)

    fig.colorbar(cf_a, ax=axes[0].tolist(), location='right', shrink=0.95, label='a(x,y)')
    fig.colorbar(cf_u, ax=axes[1].tolist(), location='right', shrink=0.95, label='u(x,y)')
    fig.colorbar(cf_p, ax=axes[2].tolist(), location='right', shrink=0.95, label='\u00fb(x,y)')
    fig.colorbar(cf_d, ax=axes[3].tolist(), location='right', shrink=0.95, label='u \u2212 \u00fb')

    fig.suptitle(f'FCN: input / truth / prediction / error  (test set){title_extra}', fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved -> {save_path}')
    plt.show()


if __name__ == '__main__':
    ############################# Data processing #############################
    data_dict = prepare_data()

    ############################# Train #############################
    channel_width = 12
    n_layers = 3
    kernel_size = 3
    learning_rate = 0.001
    epochs = 60

    net, loss_train_list, loss_test_list, final_loss = train_fcn_model(
        data_dict,
        width=channel_width,
        n_layers=n_layers,
        kernel_size=kernel_size,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    ############################# Plots #############################
    plot_loss_curves(loss_train_list, loss_test_list)
    plot_contour_physical(net, data_dict)
    plot_contour_normalised(net, data_dict)
    plot_contour_comparison(net, data_dict)

