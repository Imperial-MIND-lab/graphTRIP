"""
Permutation importance for the graph-level regression MLP.

Two modes, differing in how the models see the data:

    within_fold  (default) The models were cross-validated on this dataset, so each subject
                 is scored by the one fold model that held it out, and a feature block is
                 shuffled WITHIN each test fold. Score: negative MAE.

    mean_vote    The models were trained on a different cohort and are transferred
                 zero-shot, so every fold model predicts every subject and the prediction
                 is their mean vote. A feature block is shuffled across the WHOLE cohort,
                 and the SAME shuffle is applied to every fold model -- otherwise mean
                 voting averages the perturbation away and the importance collapses to
                 zero. Score: correlation with the outcome, so importance reads as the drop
                 in r. Clinical inputs are mapped into each model's input space first,
                 optionally with harmonisation, exactly as in transfer_and_finetune.

Two p-values are reported:
- p_value/p_value_fdr tests whether the drop is consistent across repeats
- p_perm tests if the fraction of shuffles whose score reaches the intact one is significant

Author: Hanna Tolle
Date: 2024-12-30
License: BSD 3-Clause
"""

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import torch
import torch.nn
import numpy as np
from scipy.stats import ttest_1samp, pearsonr
import logging
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import pandas as pd

from utils.files import add_project_root
from utils.configs import *
from utils.helpers import fix_random_seed, get_logger, check_weights_exist


# Create experiment and logger -------------------------------------------------
ex = Experiment('permutation_importance', ingredients=[data_ingredient, 
                                                       vgae_ingredient, 
                                                       mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'permutation_importance'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join(project_root(), 'outputs', 'runs', run_name)

    # Model weights directory, filenames and number of permutations
    weights_dir = os.path.join('outputs', 'weights', 'final_config_screening', f'job{jobid}_seed{seed}')
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'mlp': [f'k{k}_mlp_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}
    n_repeats = 30

    # "within_fold" for cross-validated models on their own dataset, 
    # "mean_vote" for models transferred zero-shot to a new cohort
    mode = 'within_fold'

    # Clinical attributes to harmonise onto the pretrained model's training scale, and which
    # attributes that model received standardised. Both only apply in mean_vote mode.
    harmonise_graph_attrs = []
    source_standardised_attrs = None

    # Manage log level/ verbosity
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE, MLP and dataset configs from weights_dir
    previous_config = load_ingredient_configs(weights_dir, ['vgae_model', 'mlp_model', 'dataset'])

    # Match configs of relevant ingredients
    ingredients = ['dataset', 'vgae_model', 'mlp_model']
    exceptions = ['num_nodes', 'drug_condition']
    if config.get('mode', 'within_fold') == 'mean_vote':
        # The models come from a different cohort, so everything describing which dataset
        # is being evaluated is expected to differ.
        exceptions += ['atlas', 'num_folds', 'batch_size', 'val_split',
                       'study', 'session', 'target', 'graph_attrs_to_standardise',
                       'dropout', 'reg_strength', 'layernorm', 'mse_reduction']
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Other compatibiltiy checks
    # In mean_vote mode there is one model per SOURCE fold, which need not match the number
    # of folds configured for the new dataset.
    num_folds = previous_config['dataset']['num_folds'] \
        if config.get('mode', 'within_fold') == 'mean_vote' \
        else config_updates['dataset']['num_folds']
    weight_filenames = config_updates.get('weight_filenames', None)
    if weight_filenames is None:
        weight_filenames =  {'vgae': [f'k{k}_vgae_weights.pth' for k in range(num_folds)],
                             'mlp': [f'k{k}_mlp_weights.pth' for k in range(num_folds)],
                             'test_fold_indices': ['test_fold_indices.csv']}
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames

    return config_updates

# Helper functions ------------------------------------------------------------
def get_inputs_and_labels(data, vgaes, testfold_indices, device):
    '''Returns all input features and true labels from the data as NumPy arrays.
    
    Parameters:
    ----------
    data (Dataset): The complete dataset
    vgaes (List[VGAE]): List of trained VGAE models, one for each fold
    testfold_indices (numpy array): Maps each data sample to its test fold
        Maps each data sample to its test fold
    device (torch.device): Device to run computations on
    '''
    num_samples = len(data)
    num_folds = len(vgaes)
    
    # Get labels and clinical data for all samples at once
    batch = next(iter(DataLoader(data, batch_size=num_samples, shuffle=False))).to(device)
    ytrue = batch.y.cpu().numpy()
    clinical_data = batch.graph_attr.cpu().numpy()
    treatment = get_treatment(batch, num_z_samples=0)
    
    # Initialize array for VGAE outputs
    z_readout = np.zeros((num_samples, vgaes[0].readout_dim))
    
    # Process each fold
    for k in range(num_folds):
        # Get indices for samples in this test fold
        fold_indices = np.where(testfold_indices == k)[0]
        
        # Get the subset of data for this fold
        fold_data = data[fold_indices]
        fold_batch = next(iter(DataLoader(fold_data, batch_size=len(fold_indices), shuffle=False))).to(device)
        
        # Get VGAE model for this fold
        vgae = vgaes[k]
        vgae.eval()
        
        with torch.no_grad():
            # Get context and VGAE latent representations for this fold
            context = get_context(fold_batch)
            out = vgae(fold_batch)
            fold_z_readout = vgae.readout(out.mu, context, fold_batch.batch).cpu().numpy()
            
            # Store the results
            z_readout[fold_indices] = fold_z_readout

    return z_readout, clinical_data, ytrue, treatment

def compute_score_with_fold_permutation(kfold_models, 
                                        testfold_indices, 
                                        mlp_inputs, 
                                        ytrue, 
                                        treatment,
                                        device, 
                                        feature_idx=None):
    """
    Computes score with option to permute a feature within each fold.
    
    Parameters:
    ----------
    kfold_models (List[LatentMLP]): list of trained MLPs models,
        each trained on a different fold.
    testfold_indices (numpy array): maps each data sample to its test fold
    mlp_inputs (numpy array): [z, clinical_data]
    ytrue (numpy array): true labels
    device (torch.device): Device to run computations on
    feature_idx (int or slice): If provided, this feature/features will be permuted within each fold
    """
    num_folds = max(testfold_indices) + 1
    ypreds = np.zeros_like(ytrue, dtype=float)
    
    x_permuted = mlp_inputs.copy()
    for k in range(num_folds):
        # Get the MLP that was tested on this fold
        mlp = kfold_models[k]
        test_indices = np.where(testfold_indices == k)[0]
        
        # If feature_idx provided, permute the specified feature(s) within this fold
        if feature_idx is not None:
            fold_permutation = np.random.permutation(len(test_indices))
            x_permuted[test_indices[:, None], feature_idx] = \
                x_permuted[test_indices[fold_permutation, None], feature_idx]
        
        x = torch.tensor(x_permuted[test_indices, :], dtype=torch.float32).to(device)
        
        mlp.eval()
        with torch.no_grad():
            if treatment is None:
                ypreds[test_indices] = mlp(x).cpu().numpy().flatten()
            else:
                testfold_treatment = treatment[test_indices]
                ypreds[test_indices] = mlp(x, testfold_treatment).cpu().numpy().flatten()

    return -np.mean(np.abs(ypreds - ytrue))

def compute_mean_vote_score(kfold_models,
                            readouts,
                            clinical,
                            ytrue,
                            treatment,
                            device,
                            feature_idx=None,
                            permutation=None):
    """
    Mean-voted score of all fold models on the whole cohort, optionally with one block of
    MLP inputs permuted across subjects.

    Used when the models were trained on a different cohort, so every model predicts every
    subject and none of them has a test fold here. The same permutation must be applied to
    every fold model: with an independent shuffle per model, mean voting averages the
    perturbation out and every feature looks unimportant.

    Parameters:
    ----------
    kfold_models (List): trained MLPs, one per source fold.
    readouts (List[np.ndarray]): VGAE readout of each fold model, [n_subjects, readout_dim].
    clinical (List[np.ndarray]): clinical inputs of each fold model, already mapped into
        that model's input space, [n_subjects, n_graph_attrs].
    ytrue (np.ndarray): true labels.
    device (torch.device): Device to run computations on.
    feature_idx (int or slice): If provided, this feature/these features are permuted.
    permutation (np.ndarray): subject permutation to apply, shared across all fold models.

    Returns:
    -------
    (r, mae) of the mean-voted prediction.
    """
    preds = np.zeros((len(ytrue), len(kfold_models)))
    for k, mlp in enumerate(kfold_models):
        x = np.concatenate([readouts[k], clinical[k]], axis=1)
        if feature_idx is not None:
            x[:, feature_idx] = x[permutation][:, feature_idx]

        x = torch.tensor(x, dtype=torch.float32).to(device)
        mlp.eval()
        with torch.no_grad():
            if treatment is None:
                preds[:, k] = mlp(x).cpu().numpy().flatten()
            else:
                preds[:, k] = mlp(x, treatment).cpu().numpy().flatten()

    mean_vote = preds.mean(axis=1)
    return pearsonr(mean_vote, ytrue)[0], np.mean(np.abs(mean_vote - ytrue))

def get_mean_vote_inputs(data, vgaes, device):
    """
    VGAE readout of every fold model on every subject.

    Unlike get_inputs_and_labels(), which assigns each subject to the readout of the model
    that held it out, this returns one full readout per model, because in a zero-shot
    transfer no model has seen any of these subjects.
    """
    batch = next(iter(DataLoader(data, batch_size=len(data), shuffle=False))).to(device)
    ytrue = batch.y.squeeze(-1).cpu().numpy()
    treatment = get_treatment(batch, num_z_samples=0)

    readouts = []
    for vgae in vgaes:
        vgae.eval()
        with torch.no_grad():
            out = vgae(batch)
            readouts.append(vgae.readout(out.mu, get_context(batch), batch.batch).cpu().numpy())
    return readouts, ytrue, treatment

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment ------------------------------------------------------
    seed = _config['seed']
    verbose = _config['verbose']
    n_repeats = _config['n_repeats']
    output_dir = add_project_root(_config['output_dir'])
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']
    is_cfrnet = _config['mlp_model']['model_type'] == 'CFRHead'
    mode = _config['mode']
    if mode not in ('within_fold', 'mean_vote'):
        raise ValueError(f"Unknown mode '{mode}'; expected 'within_fold' or 'mean_vote'.")

    # Make output directory, get device and fix random seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    device = torch.device(_config['device'])

    # Load data and trained models
    data = load_data()
    if is_cfrnet:
        # Add treatment transform to the dataset (required for CFRHead)
        add_treatment_transform(data)
    vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device)
    mlps = load_trained_mlps(weights_dir, weight_filenames['mlp'], device,
                             latent_dims=[vgae.readout_dim for vgae in vgaes])

    # If data has no labels (e.g. X-learner),
    # load the prediction results from weights_dir and get labels from there
    if data[0].y is None:
        pre_results = pd.read_csv(os.path.join(weights_dir, 'prediction_results.csv'))
        labels = dict(zip(pre_results['subject_id']+1, pre_results['label']))
        addlabel_tfm = AddLabel(labels)
        data.transform = T.Compose([*data.transform.transforms, addlabel_tfm])

    graph_attrs = list(data[0].attr_names.graph)

    if mode == 'within_fold':
        testfold_indices = np.loadtxt(
            os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)

        # Get MLP inputs and labels
        z_readout, clinical_data, ytrue, treatment = get_inputs_and_labels(
            data, vgaes, testfold_indices, device)
        mlp_inputs = np.concatenate([z_readout, clinical_data], axis=1)
        readout_dim = z_readout.shape[1]

        def score(feature_idx=None, permutation=None):
            return compute_score_with_fold_permutation(
                mlps, testfold_indices, mlp_inputs, ytrue, treatment, device,
                feature_idx=feature_idx)

    else:
        # Zero-shot transfer: map the clinical scores into each model's own input space
        # first, exactly as transfer_and_finetune does, so the importances describe the
        # inputs the reported predictions were actually made from.
        transfer = build_transfer_inputs(
            data=data,
            weights_dir=weights_dir,
            num_models=len(mlps),
            graph_attrs=graph_attrs,
            harmonise=_config['harmonise_graph_attrs'],
            source_standardised_attrs=_config['source_standardised_attrs'],
            source_dataset_config=load_ingredient_configs(weights_dir, ['dataset'])['dataset'])
        # Measure importance on the inputs the reported predictions were made from
        condition = HARMONISED if _config['harmonise_graph_attrs'] else NO_HARMONISATION
        clinical_data = transfer['inputs'][condition]
        readouts, ytrue, treatment = get_mean_vote_inputs(data, vgaes, device)
        readout_dim = readouts[0].shape[1]
        logger.info(f"Permutation importance in mean_vote mode, condition '{condition}': "
                    f"{len(mlps)} models x {len(data)} subjects, "
                    f"harmonised = {transfer['harmonised_attrs']}.")

        def score(feature_idx=None, permutation=None):
            r, _ = compute_mean_vote_score(mlps, readouts, clinical_data, ytrue, treatment,
                                           device, feature_idx=feature_idx,
                                           permutation=permutation)
            return r

    # Initialize lists to store results
    features_agg = []
    scores_agg = []

    # Compute baseline score without permutation
    baseline_score = score()

    # The blocks whose importance is measured: all latent features together, then each
    # clinical attribute on its own.
    blocks = [('Z_whole', slice(0, readout_dim))]
    blocks += [(name, readout_dim + j) for j, name in enumerate(graph_attrs)]

    rng = np.random.default_rng(seed)
    p_perm_agg = []
    for name, feature_idx in blocks:
        feature_scores = np.zeros(n_repeats)
        permuted_scores = np.zeros(n_repeats)
        for i in range(n_repeats):
            permuted_scores[i] = score(feature_idx=feature_idx,
                                       permutation=rng.permutation(len(ytrue)))
            feature_scores[i] = baseline_score - permuted_scores[i]
        scores_agg.append(feature_scores)
        features_agg.append(name)
        # Subject-level test: how often does destroying this block still reach the intact
        # score? Unlike p_value below, this is a statement about the subjects, not about
        # the consistency of the repeats.
        p_perm_agg.append(float(np.mean(permuted_scores >= baseline_score)))

    # Compute statistics for aggregated scores
    scores_agg = np.array(scores_agg)
    means_agg = np.mean(scores_agg, axis=1)
    stds_agg = np.std(scores_agg, axis=1)
    sems_agg = stds_agg / np.sqrt(n_repeats)
    t_stats_agg, p_values_agg = ttest_1samp(scores_agg, 0, axis=1)
    # A block that is constant across subjects (e.g. Condition on a single-arm cohort) has
    # zero importance in every repeat, so its t-test is undefined. Correct only the
    # well-defined p-values -- passing the NaN through fdrcorrection would return NaN for
    # every feature, not just that one.
    p_values_fdr_agg = np.full_like(p_values_agg, np.nan, dtype=float)
    finite = np.isfinite(p_values_agg)
    if finite.any():
        p_values_fdr_agg[finite] = fdrcorrection(p_values_agg[finite], alpha=0.05)[1]

    # Save aggregated importance scores
    agg_stats = pd.DataFrame({
        'feature': features_agg,
        'mean': means_agg,
        'std': stds_agg,
        'se': sems_agg,
        't_stat': t_stats_agg,
        'p_value': p_values_agg,
        'p_value_fdr': p_values_fdr_agg,
        'p_perm': p_perm_agg,
        'baseline_score': baseline_score,
        'mode': mode,
        'score': 'r' if mode == 'mean_vote' else 'neg_mae'
    })
    agg_stats.to_csv(os.path.join(output_dir, 'importance_scores_aggregated.csv'), index=False)
    
    # Plot aggregated importance scores
    fig, ax = plt.subplots(1, 1, figsize=(len(features_agg)*0.5, 6))
    bars = ax.bar(features_agg, means_agg, yerr=sems_agg, align='center', capsize=5)
    ax.set_ylabel('Drop in r when shuffled' if mode == 'mean_vote' else 'Importance Scores')
    ax.set_xticks(range(len(features_agg)))
    ax.set_xticklabels(features_agg, rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add significance markers
    offset = 0.05 * max(means_agg + sems_agg)
    for i, (bar, p_value_fdr) in enumerate(zip(bars, p_values_fdr_agg)):
        if p_value_fdr < 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset,
                   '*', ha='center', va='bottom', color='red', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'importance_scores_aggregated.png'), dpi=300)
    ex.add_artifact(os.path.join(output_dir, 'importance_scores_aggregated.png'))
    if not verbose:
        plt.close()
