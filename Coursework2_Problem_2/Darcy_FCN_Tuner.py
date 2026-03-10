import optuna
from optuna.pruners import SuccessiveHalvingPruner
import torch
import numpy as np

from Darcy_FCN import prepare_data, train_fcn_model, plot_loss_curves, plot_contour_comparison


# Default hyperparameters (before tuning)
INITIAL_WIDTH = 12
INITIAL_N_LAYERS = 3
INITIAL_KERNEL_SIZE = 3
INITIAL_LR = 0.001


def objective(trial, data_dict):
    """Optuna objective function."""

    seed = 42 + trial.number
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Suggest hyperparameters
    width = trial.suggest_int('width', 8, 64, step=4)
    n_layers = trial.suggest_int('n_layers', 2, 6)
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    print(f"\nTrial {trial.number}: width={width}, n_layers={n_layers}, "
          f"kernel_size={kernel_size}, lr={learning_rate:.6f}")

    net, train_losses, test_losses, final_test_loss = train_fcn_model(
        data_dict=data_dict,
        width=width,
        n_layers=n_layers,
        kernel_size=kernel_size,
        learning_rate=learning_rate,
        trial=trial,
    )

    print(f"Trial {trial.number} completed with final test loss: {final_test_loss:.2e}")

    return final_test_loss


if __name__ == "__main__":
    print("=" * 60)
    print("FCN Hyperparameter Tuning with Optuna (ASHA Pruner)")
    print("=" * 60)

    # Prepare data once
    print("\nLoading data...")
    data_dict = prepare_data()
    print("Data loaded successfully!")

    # Create study with ASHA pruner and database storage
    pruner = SuccessiveHalvingPruner(min_resource=20)
    study = optuna.create_study(
        study_name='darcy_fcn_tuning',
        direction='minimize',
        pruner=pruner,
        storage='sqlite:///optuna_fcn_study.db',
        load_if_exists=True,
    )

    # Run optimization
    print(f"\nStarting hyperparameter search with {pruner.__class__.__name__}...")
    n_trials = 30

    study.optimize(
        lambda trial: objective(trial, data_dict),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best test loss: {study.best_value:.2e}")
    print(f"Best trial random seed: {42 + study.best_trial.number}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if key == 'learning_rate':
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"\nResults saved to 'optuna_fcn_study.db'")

    # ==================== Retrain & Plot ====================
    # Train initial model (default hyperparameters)
    print("\n" + "=" * 60)
    print("Training INITIAL model (default hyperparameters)...")
    print(f"  width={INITIAL_WIDTH}, n_layers={INITIAL_N_LAYERS}, "
          f"kernel_size={INITIAL_KERNEL_SIZE}, lr={INITIAL_LR}")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)
    initial_net, initial_train_losses, initial_test_losses, _ = train_fcn_model(
        data_dict=data_dict,
        width=INITIAL_WIDTH,
        n_layers=INITIAL_N_LAYERS,
        kernel_size=INITIAL_KERNEL_SIZE,
        learning_rate=INITIAL_LR,
    )

    # Retrain best model (Optuna-tuned hyperparameters)
    best_params = study.best_params
    best_seed = 42 + study.best_trial.number
    print("\n" + "=" * 60)
    print("Retraining BEST model (Optuna-tuned hyperparameters)...")
    print(f"  width={best_params['width']}, n_layers={best_params['n_layers']}, "
          f"kernel_size={best_params['kernel_size']}, lr={best_params['learning_rate']:.6f}")
    print("=" * 60)
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    best_net, best_train_losses, best_test_losses, _ = train_fcn_model(
        data_dict=data_dict,
        width=best_params['width'],
        n_layers=best_params['n_layers'],
        kernel_size=best_params['kernel_size'],
        learning_rate=best_params['learning_rate'],
    )

    # Plot loss curves
    print("\n" + "=" * 60)
    print("Plotting results...")
    print("=" * 60)
    plot_loss_curves(initial_train_losses, initial_test_losses, save_path='fcn_initial_loss.png')
    plot_loss_curves(best_train_losses, best_test_losses, save_path='fcn_best_loss.png')

    # Plot contour comparisons
    plot_contour_comparison(initial_net, data_dict, save_path='fcn_initial_contour.png', title_extra=' [initial]')
    plot_contour_comparison(best_net, data_dict, save_path='fcn_best_contour.png', title_extra=' [tuned]')
