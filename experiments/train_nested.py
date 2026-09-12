'''
Nested cross-validation of the joint VGAE + MLP model.

One outer fold of patients is held out completely: no model of this run ever sees it, not
in training and not in the inner test folds. The remaining "inner" patients are
cross-validated exactly as train_jointly does, giving the usual ensemble of fold models.
Running this for several training seeds, and for every outer fold, yields an ensemble whose
biomarker selection can be validated on patients that took no part in it.

train_jointly.py is the published pipeline and is deliberately left untouched; this file
duplicates it so that the two cannot drift into each other.

The outer split is seeded by outer_seed, which is independent of the training seed, so all
training seeds of one outer fold hold out the same patients.

Note for downstream tools: test_fold_indices.csv describes the INNER patients only, so it
is shorter than the cohort. Tools that mask the full dataset with it -- estimate_propensity
and visualize_overfitting -- would select the wrong patients without raising. GRAIL is safe:
it only checks that the file exists. nested_split.csv maps between the index spaces.

Authors: Hanna M. Tolle
Date: 2026-09-12
License: BSD 3-Clause
'''

import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import *
from experiments.ingredients.vgae_ingredient import *
from experiments.ingredients.mlp_ingredient import *

import os
import torch
import torch.nn
from tqdm import tqdm
from time import time
import copy
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices
from utils.plotting import plot_loss_curves, true_vs_pred_scatter


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_nested', ingredients=[data_ingredient,
                                             vgae_ingredient,
                                             mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'train_nested'
    jobid = 0
    seed = 291
    outer_fold = 0           # Which outer fold is held out.
    num_outer_folds = 7      # 42 patients / 7 = 6 held out per outer fold.
    outer_seed = 0           # Seeds the outer split only; keep independent of seed.
    run_name = f'{exname}_outer{outer_fold}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training configurations
    lr = 0.001            # Learning rate.
    num_epochs = 300      # Number of epochs to train.
    num_z_samples = 1     # 0 for training MLP on the means of VGAE latent variables.
    alpha = 0.5           # Loss = alpha*vgae_loss + (1-alpha)*mlp_loss
    balance_attrs = None  # attrs to balance on for k-fold CV. If None, no balancing.
    num_folds_to_run = None  # If set, only run this many inner folds (debugging only).

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Checks the configs this experiment cannot run without.'''
    num_outer_folds = config.get('num_outer_folds', 7)
    outer_fold = config.get('outer_fold', 0)
    assert 0 <= outer_fold < num_outer_folds, \
        f'outer_fold must be in [0, {num_outer_folds}), got {outer_fold}.'

    assert config.get('balance_attrs', None) is not None, \
        'Nested CV requires balance_attrs (e.g. ["Condition"]) for both splits.'

    dataset = config.get('dataset', {})
    assert dataset.get('val_split', 0.) == 0., \
        'Nested CV assumes no inner validation split, i.e. no model selection.'
    assert not dataset.get('standardise_x', False) \
        and not dataset.get('graph_attrs_to_standardise', []), \
        'Standardisation statistics are not plumbed through to the held-out outer patients.'
    return config

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(vgae, mlp, lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    return torch.optim.Adam(list(mlp.parameters()) + list(vgae.parameters()), lr=lr)

@ex.capture
def train_vgae_mlp(vgae, mlp, loader, optimizer, device, num_z_samples, alpha):
    return train_joint_vgae_mlp(vgae, mlp, loader, optimizer,
                                device=device, num_z_samples=num_z_samples, alpha=alpha)

@ex.capture
def test_vgae_mlp(vgae, mlp, loader, device, num_z_samples):
    vgae_test_loss, mlp_test_loss = test_joint_vgae_mlp(vgae, mlp, loader,
                                                        device=device, num_z_samples=num_z_samples)
    return vgae_test_loss, mlp_test_loss

@ex.capture
def get_outer_split(data, balance_attrs, num_outer_folds, outer_fold, outer_seed):
    '''
    Splits the cohort into inner and held-out patients.

    Returns both as ascending positions into the full dataset. seed and balance_attrs are
    passed explicitly because Sacred does not add a seed to ingredient configs, and an
    unseeded StratifiedKFold would resplit differently on every run.
    '''
    splits = get_balanced_kfold_splits(data,
                                       num_folds=num_outer_folds,
                                       balance_attrs=balance_attrs,
                                       seed=outer_seed)
    inner_positions, outer_positions = splits[outer_fold]
    return np.sort(inner_positions), np.sort(outer_positions)

@ex.capture
def get_inner_dataloaders(inner_data, balance_attrs, seed):
    '''Cross-validates the inner patients; num_folds comes from the dataset config.'''
    return get_balanced_kfold_dataloaders(inner_data, balance_attrs=balance_attrs, seed=seed)

def build_nested_split(data, inner_positions, outer_positions, test_indices, _config):
    '''
    Maps between the three index spaces, one row per patient of the full cohort.

    position:       into the full dataset; GRAIL writes its sub_{position} directories by it
    inner_position: into the inner dataset; the row order of test_fold_indices.csv
    subject_id:     data.subject, 0-indexed, joins to prediction_results.csv
    patient:        subject_id + 1, joins to annotations.csv and GRAIL's own subject_id
    '''
    inner_fold = {int(pos): k for k, fold_positions in enumerate(test_indices)
                  for pos in inner_positions[np.asarray(fold_positions)]}
    inner_rank = {int(pos): i for i, pos in enumerate(inner_positions)}
    is_outer = set(int(pos) for pos in outer_positions)

    rows = []
    for position in range(len(data)):
        subject_id = int(data[position].subject.item())
        rows.append({'position': position,
                     'inner_position': inner_rank.get(position, -1),
                     'subject_id': subject_id,
                     'patient': subject_id + 1,
                     'role': 'outer' if position in is_outer else 'inner',
                     'inner_fold': inner_fold.get(position, -1),
                     'outer_fold': _config['outer_fold'],
                     'outer_seed': _config['outer_seed'],
                     'num_outer_folds': _config['num_outer_folds']})
    return pd.DataFrame(rows)

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Unpack configs
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    alpha = _config['alpha']
    outer_fold = _config['outer_fold']
    num_outer_folds = _config['num_outer_folds']
    num_folds = _config['dataset']['num_folds']
    num_folds_to_run = _config.get('num_folds_to_run') or num_folds
    assert 0 <= outer_fold < num_outer_folds, \
        f'outer_fold must be in [0, {num_outer_folds}), got {outer_fold}.'

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []

    # Hold out the outer fold ------------------------------------------------------
    data = load_data()
    inner_positions, outer_positions = get_outer_split(data)
    inner_data = data[inner_positions]
    outer_data = data[outer_positions]
    logger.info(f'Outer fold {outer_fold}/{num_outer_folds}: {len(inner_data)} inner '
                f'patients, {len(outer_data)} held out.')

    # Cross-validate the inner patients. This reseeds the global RNG with the training
    # seed, which the outer split has just set to outer_seed; the order matters.
    train_loaders, val_loaders, test_loaders, test_indices, mean_std \
        = get_inner_dataloaders(inner_data, seed=seed)
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Record the split before training, so a crashed run still leaves it behind
    nested_split = build_nested_split(data, inner_positions, outer_positions,
                                      test_indices, _config)
    nested_split.to_csv(os.path.join(output_dir, 'nested_split.csv'), index=False)

    # The held-out patients are predicted by every fold model, in a fixed order
    outer_loader = DataLoader(outer_data, batch_size=len(outer_data), shuffle=False)
    position_of = dict(zip(nested_split['subject_id'], nested_split['position']))

    # Train-test loop ------------------------------------------------------------
    start_time = time()

    best_outputs = init_outputs_dict(inner_data)
    vgae_train_loss, vgae_test_loss, vgae_val_loss = {}, {}, {}
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}
    best_vgae_states, best_mlp_states = [], []
    outer_rows = []

    for k in tqdm(range(num_folds_to_run), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        vgae_train_loss[k], vgae_test_loss[k], vgae_val_loss[k] = [], [], []

        # Initialise models and optimizer
        vgae = build_vgae().to(device)
        mlp = build_mlp(latent_dim=vgae.readout_dim).to(device)
        optimizer = get_optimizer(vgae, mlp)

        # Best validation loss and early stopping counter
        best_val_loss = float('inf')
        best_vgae_state = None
        best_mlp_state = None

        for epoch in tqdm(range(_config['num_epochs']), desc='Epochs', disable=not verbose):
            # Train VGAE and MLP
            _ = train_vgae_mlp(vgae, mlp, train_loaders[k], optimizer, device)

            # Compute training losses
            vgae_train_loss_epoch, mlp_train_loss_epoch = test_vgae_mlp(vgae, mlp, train_loaders[k], device)
            vgae_train_loss[k].append(vgae_train_loss_epoch)
            mlp_train_loss[k].append(mlp_train_loss_epoch)

            # Test VGAE and MLP on the inner test fold
            vgae_test_loss_epoch, mlp_test_loss_epoch = test_vgae_mlp(vgae, mlp, test_loaders[k], device)
            vgae_test_loss[k].append(vgae_test_loss_epoch)
            mlp_test_loss[k].append(mlp_test_loss_epoch)

            # Log training and test losses
            ex.log_scalar(f'training/fold{k}/epoch/vgae_loss', vgae_train_loss_epoch)
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/vgae_loss', vgae_test_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss_epoch)

            # Validate models, if applicable
            if len(val_loaders) > 0:
                vgae_val_loss_epoch, mlp_val_loss_epoch = test_vgae_mlp(vgae, mlp, val_loaders[k], device)
                vgae_val_loss[k].append(vgae_val_loss_epoch)
                mlp_val_loss[k].append(mlp_val_loss_epoch)

                # Log validation losses
                ex.log_scalar(f'validation/fold{k}/epoch/vgae_loss', vgae_val_loss_epoch)
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss_epoch)

                # Save the best model if validation loss is at its minimum
                total_val_loss = alpha*vgae_val_loss_epoch + (1-alpha)*mlp_val_loss_epoch
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best_vgae_state = copy.deepcopy(vgae.state_dict())
                    best_mlp_state = copy.deepcopy(mlp.state_dict())

        # Load best model of this fold
        if best_vgae_state is not None:
            vgae.load_state_dict(best_vgae_state)
        if best_mlp_state is not None:
            mlp.load_state_dict(best_mlp_state)

        # Save model weights
        if save_weights:
            torch.save(mlp.state_dict(), os.path.join(output_dir, f'k{k}_mlp_weights.pth'))
            torch.save(vgae.state_dict(), os.path.join(output_dir, f'k{k}_vgae_weights.pth'))

        # Keep a list of model states
        best_vgae_states.append(copy.deepcopy(vgae.state_dict()))
        best_mlp_states.append(copy.deepcopy(mlp.state_dict()))

        # Save the inner test predictions (on VGAE latent means) of the best model
        outputs = get_mlp_outputs_nograd(mlp, test_loaders[k], device,
                                         get_x=get_x_with_vgae,
                                         vgae=vgae, num_z_samples=0)
        update_best_outputs(best_outputs, outputs)

        # Predict the held-out patients with this fold model
        outer_outputs = get_mlp_outputs_nograd(mlp, outer_loader, device,
                                               get_x=get_x_with_vgae,
                                               vgae=vgae, num_z_samples=0)
        fold_outputs = init_outputs_dict(outer_data)
        update_best_outputs(fold_outputs, outer_outputs)
        fold_df = pd.DataFrame(fold_outputs)
        fold_df.insert(0, 'fold', k)
        outer_rows.append(fold_df)

    # Print training time
    end_time = time()
    logger.info(f"Nested training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save the inner results ---------------------------------------------------------
    best_outputs = pd.DataFrame(best_outputs)
    best_outputs = add_drug_condition_to_outputs(best_outputs, _config['dataset']['study'])
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    best_outputs.to_csv(data_file, index=False)

    # Save inner test fold assignments. These index the inner patients, not the cohort.
    test_indices_file = save_test_indices(test_indices, output_dir)

    # Save final prediction results
    r, p, mae, mae_std = evaluate_regression(best_outputs)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Inner results: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

    # Save the held-out results ------------------------------------------------------
    outer_df = pd.concat(outer_rows, ignore_index=True)
    outer_df = add_drug_condition_to_outputs(outer_df, _config['dataset']['study'])
    outer_df.insert(2, 'position', outer_df['subject_id'].map(position_of))
    outer_df.to_csv(os.path.join(output_dir, 'outer_predictions.csv'), index=False)

    # The ensemble prediction of a held-out patient is the mean over the fold models
    ensemble = (outer_df.groupby('subject_id')
                .agg(prediction=('prediction', 'mean'),
                     prediction_std=('prediction', 'std'),
                     label=('label', 'first'),
                     Condition=('Condition', 'first'))
                .reset_index())
    outer_r, outer_p, outer_mae, outer_mae_std = evaluate_regression(ensemble)
    outer_results = {'seed': seed, 'outer_fold': outer_fold,
                     'num_models': len(outer_rows), 'num_patients': len(ensemble),
                     'r': outer_r, 'p': outer_p, 'mae': outer_mae, 'mae_std': outer_mae_std}
    pd.DataFrame(outer_results, index=[0]).to_csv(
        os.path.join(output_dir, 'outer_metrics.csv'), index=False)

    for key, value in outer_results.items():
        ex.log_scalar(f'outer_prediction/{key}', value)
    logger.info(f"Held-out results: r={outer_r:.4f}, p={outer_p:.4e}, "
                f"mae={outer_mae:.4f} ± {outer_mae_std:.4f}.")

    # Plot results ----------------------------------------------------------------
    # The VGAE reconstruction figures are skipped: get_test_reconstructions indexes the
    # cohort with fold indices, which are inner-space here.
    plot_loss_curves(vgae_train_loss, vgae_test_loss, vgae_val_loss, save_path=os.path.join(output_dir, 'vgae_loss_curves.png'))
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=os.path.join(output_dir, 'mlp_loss_curves.png'))
    image_files += [os.path.join(output_dir, 'vgae_loss_curves.png'), os.path.join(output_dir, 'mlp_loss_curves.png')]

    # True vs predicted scatters, for the inner folds and for the held-out patients
    title = f'Inner: r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    true_vs_pred_scatter(best_outputs, title=title, save_path=os.path.join(output_dir, 'true_vs_predicted.png'))
    image_files.append(os.path.join(output_dir, 'true_vs_predicted.png'))

    title = (f'Held-out (ensemble of {len(outer_rows)}): r={outer_r:.4f}, '
             f'p={outer_p:.4e}, MAE={outer_mae:.4f} ± {outer_mae_std:.4f}')
    true_vs_pred_scatter(ensemble, title=title,
                         save_path=os.path.join(output_dir, 'outer_true_vs_predicted.png'))
    image_files.append(os.path.join(output_dir, 'outer_true_vs_predicted.png'))

    # Log images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close()
