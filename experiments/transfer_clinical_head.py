"""
Loads a pre-trained prediction head that reads clinical data only, and evaluates it
zero-shot on a new dataset. This is the no-VGAE, no fine-tuning version of transfer_and_finetune. 

Author: Hanna M. Tolle
Date: 2026-08-18
License: BSD-3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import *
from experiments.ingredients.mlp_ingredient import *

import os
import torch
import torch.nn
from torch_geometric.loader import DataLoader
import copy
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.plotting import true_vs_pred_scatter
from utils.configs import load_ingredient_configs, match_ingredient_configs
from utils.statsalg import correlation_permutation_test
from utils.harmonisation import graph_attr_matrix, write_train_stats


# Create experiment and logger -------------------------------------------------
ex = Experiment('transfer_clinical_head', ingredients=[data_ingredient,
                                                       mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Heads that are not torch modules are loaded through SklearnLinearModelWrapper, which reads
# the coefficients saved next to the pickled estimator by train_linreg_on_clinical.py.
SKLEARN_HEAD = 'SklearnLinearModelWrapper'

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'transfer_clinical_head'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = False  # Nothing is trained here; kept so callers can set it uniformly.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory with pre-trained model weights
    weights_dir = os.path.join('outputs', 'weights')

    # One head per fold model of the source cohort; all of them are evaluated.
    weight_filenames = {'mlp': ['k0_mlp_weights.pth']}

    # Zero-shot evaluation configurations
    # Clinical attributes to harmonise onto the pretrained model's training scale. If empty,
    # only the no_harmonisation condition is evaluated.
    harmonise_graph_attrs = []
    # Which graph_attrs the pretrained model received in standardised form. Leave as None to
    # read it from the pretrained model's config.json, which only works if that file stores
    # it as a plain list.
    source_standardised_attrs = None
    n_permutations = 10000  # Permutation test of the mean-vote correlation.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the dataset and MLP configs from weights_dir
    previous_config = load_ingredient_configs(weights_dir, ['dataset', 'mlp_model'])

    # A head that was not built by the MLP ingredient (the sklearn regressions) has no
    # mlp_model config to match against; the caller declares its model_type instead.
    ingredients = ['dataset']
    if previous_config['mlp_model']:
        ingredients.append('mlp_model')

    # Various dataset related configs may mismatch, but other configs must match
    exceptions = ['num_nodes', 'atlas',
                  'num_folds', 'batch_size', 'val_split',
                  'study', 'session', 'target', 'graph_attrs_to_standardise',
                  'dropout', 'reg_strength', 'layernorm', 'mse_reduction']
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Other config checks
    num_pretrained_models = previous_config['dataset']['num_folds']
    head_type = config_updates.get('mlp_model', {}).get('model_type', 'RegressionMLP')
    stem = 'linearregression_params' if head_type == SKLEARN_HEAD else 'mlp_weights'
    default_weight_filenames = {'mlp': [f'k{i}_{stem}.pth' for i in range(num_pretrained_models)]}
    weight_filenames = config.get('weight_filenames', default_weight_filenames)
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames

    # Don't support standardisation of x in pretrained models
    if 'standardise_x' in previous_config['dataset']:
        assert not config_updates['dataset']['standardise_x'], \
            "Standardisation of x in pretrained models is not supported."

    return config_updates

# Zero-shot transfer helpers ---------------------------------------------------
def get_transfer_outputs(mlp, clinical, x_raw, labels, subject_ids, device):
    '''
    Predictions of one fold model on the whole new cohort, given clinical inputs that have
    already been mapped into that model's input space.
    '''
    mlp.eval()
    with torch.no_grad():
        x = torch.tensor(clinical, dtype=torch.float32, device=device)
        ypred = mlp(x)
    return {'prediction': ypred.squeeze(-1).cpu().tolist(),
            'label': labels,
            'subject_id': subject_ids,
            'clinical_data': [tuple(row) for row in np.asarray(x_raw).tolist()]}

def load_pretrained_heads(weights_dir, weight_filenames, head_type, device):
    '''Loads one prediction head per fold model of the source cohort.'''
    if head_type == SKLEARN_HEAD:
        return load_sklearn_wrapper(weights_dir, weight_filenames, device=device)

    # These heads read clinical data only, so their input dimension is extra_dim alone.
    latent_dims = [0] * len(weight_filenames)
    return load_trained_mlps(weights_dir, weight_filenames, device=device,
                             latent_dims=latent_dims)

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    seed = _config['seed']
    weights_dir = add_project_root(_config['weights_dir'])

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load pretrained heads. Nothing is fitted on the new cohort, so there are no k-fold
    # splits here: every subject is equally unseen by every fold model.
    pretrained_mlps = load_pretrained_heads(weights_dir=weights_dir,
                                            weight_filenames=_config['weight_filenames']['mlp'],
                                            head_type=_config['mlp_model']['model_type'],
                                            device=device)
    num_pretrained_models = len(pretrained_mlps)

    # Make an output directory for each pretrained model
    output_dirs = []
    for i in range(num_pretrained_models):
        pretrained_model_dir = os.path.join(output_dir, f'pretrained_model{i}')
        os.makedirs(pretrained_model_dir, exist_ok=True)
        output_dirs.append(pretrained_model_dir)

    # Zero-shot evaluation --------------------------------------------------------
    # Every pretrained model is evaluated on the whole new dataset, under each condition.
    eval_loader = DataLoader(data, batch_size=len(data), shuffle=False)
    batch = next(iter(eval_loader)).to(device)
    labels = get_labels(batch, num_z_samples=0).squeeze(-1).cpu().tolist()
    subject_ids = batch.subject.cpu().tolist()

    # Map the clinical scores into each fold model's own input space
    source_dataset_config = load_ingredient_configs(weights_dir, ['dataset'])['dataset']
    transfer = build_transfer_inputs(
        data=data,
        weights_dir=weights_dir,
        num_models=num_pretrained_models,
        graph_attrs=_config['dataset']['graph_attrs'],
        harmonise=_config['harmonise_graph_attrs'],
        source_standardised_attrs=_config['source_standardised_attrs'],
        source_dataset_config=source_dataset_config)
    clinical_inputs = transfer['inputs']
    conditions = transfer['conditions']
    x_raw = graph_attr_matrix(data)
    logger.info(f"Zero-shot inputs: standardised at training = {transfer['standardised_attrs']}, "
                f"harmonised = {transfer['harmonised_attrs']} "
                f"(leave-one-out statistics of {len(data)} subjects).")

    # Save the record of what the mapping did, for the before/after distribution figure
    if not transfer['record'].empty:
        transfer['record'].to_csv(os.path.join(output_dir, 'clinical_inputs.csv'), index=False)
    if transfer['train_stats']:
        write_train_stats(transfer['train_stats'],
                          os.path.join(output_dir, 'harmonisation_stats.csv'))
    if transfer['source_values'] is not None:
        # Raw source-cohort scores: the reference distribution the new cohort is mapped onto
        source_attrs = transfer['source_attrs']
        pd.DataFrame({a: transfer['source_values'][:, source_attrs.index(a)]
                      for a in transfer['train_stats']}
                     ).to_csv(os.path.join(output_dir, 'source_cohort_clinical.csv'), index=False)

    # Evaluate predictions --------------------------------------------------------
    all_results = []
    for condition in conditions:
        suffix = '' if condition == NO_HARMONISATION else f'_{condition}'
        predictions = np.zeros((len(data), num_pretrained_models))

        for i in range(num_pretrained_models):
            initial_outputs = init_outputs_dict(data)
            outputs = get_transfer_outputs(pretrained_mlps[i], clinical_inputs[condition][i],
                                           x_raw, labels, subject_ids, device)
            update_best_outputs(initial_outputs, outputs)

            # Record metrics for each pretrained model
            initial_outputs = pd.DataFrame(initial_outputs)
            predictions[:, i] = initial_outputs['prediction'].values
            r, p, mae, mae_std = evaluate_regression(initial_outputs)
            results = {'condition': condition,
                       'pretrained_model': i,
                       'seed': seed,
                       'r': r,
                       'p': p,
                       'mae': mae,
                       'mae_std': mae_std}
            all_results.append(results)
            for k, v in results.items():
                if k not in ('pretrained_model', 'condition'):
                    ex.log_scalar(f'initial_prediction{suffix}/{k}', v)
            logger.info(f"Initial results for pretrained model {i} ({condition}): "
                        f"r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

            # True vs predicted scatter plot
            title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
            save_path = os.path.join(output_dirs[i],
                                     f'initial_true_vs_predicted_model{i}{suffix}.png')
            true_vs_pred_scatter(initial_outputs, title=title, save_path=save_path)
            image_files.append(save_path)

            # Close all plots
            plt.close('all')

            # Save & log initial prediction results
            file = os.path.join(output_dirs[i], f'initial_prediction_results{suffix}.csv')
            initial_outputs.to_csv(file, index=False)

        # Initial predictions mean voting --------------------------------------
        df = initial_outputs.copy()
        df['prediction'] = np.mean(predictions, axis=1)
        df['prediction_std'] = np.std(predictions, axis=1)
        df.to_csv(os.path.join(output_dir,
                               f'initial_prediction_results_mean_vote{suffix}.csv'), index=False)

        # Evaluate the mean predictions
        r, p, mae, mae_std = evaluate_regression(df)

        # Permutation test of the mean-vote correlation.
        null_path = os.path.join(output_dir, f'permutation_nulls{suffix}.png')
        perm = correlation_permutation_test(
            np.asarray(df['label'].values, dtype=float),
            np.asarray(df['prediction'].values, dtype=float),
            n_permutations=_config['n_permutations'], seed=seed,
            make_plot=True, save_path=null_path,
            title=f'Zero-shot on {_config["dataset"]["study"]} ({condition}), seed {seed}\n')
        image_files.append(null_path)
        plt.close('all')
        pd.DataFrame({'r': perm['null_distribution']}).to_csv(
            os.path.join(output_dir, f'permutation_null_mean_vote{suffix}.csv'), index=False)

        results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std,
                   'perm_p': perm['p_value'],
                   'null_mean': perm['null_mean'],
                   'null_sd': perm['null_std'],
                   'n_permutations': _config['n_permutations']}
        pd.DataFrame(results, index=[0]).to_csv(
            os.path.join(output_dir, f'initial_metrics_mean_vote{suffix}.csv'), index=False)

        # Log final metrics
        for k, v in results.items():
            ex.log_scalar(f'initial_prediction_mean_vote{suffix}/{k}', v)
        logger.info(f"Initial results for mean voting ({condition}): r={r:.4f}, p={p:.4e}, "
                    f"permutation p={perm['p_value']:.4f} "
                    f"(null {perm['null_mean']:+.4f} ± {perm['null_std']:.4f}), "
                    f"mae={mae:.4f} ± {mae_std:.4f}.")

        # True vs predicted scatter
        title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
        save_path = os.path.join(output_dir, f'initial_true_vs_predicted_mean_vote{suffix}.png')
        true_vs_pred_scatter(df, title=title, save_path=save_path)
        image_files.append(save_path)

        # Close all plots
        plt.close('all')

    # Save initial metrics of all pretrained models
    all_results = pd.DataFrame(all_results)
    all_results.to_csv(os.path.join(output_dir, 'initial_metrics_summary.csv'), index=False)

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)
