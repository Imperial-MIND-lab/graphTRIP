'''
In this experiment, we train a single CATE model on combined data from both conditions
by loading a pre-trained VGAE and training a sklearn regression head on the 
latent representations + clinical data from all samples.
The labels are ITEs computed from t-learner predictions.
We use LOOCV, ensuring each held-out test sample uses the VGAE that had it in its test fold.

Dependencies:
- data/raw/{study}/annotations.csv
- Pre-trained VGAE weights in weights_dir
- T-learner prediction results files (t0_pred_file and t1_pred_file)

Authors: Hanna M. Tolle
Date: 2025-04-11
License: BSD 3-Clause
'''

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.vgae_ingredient import *
from experiments.ingredients.mlp_ingredient import * 

import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from typing import Dict
from sklearn.preprocessing import StandardScaler

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.plotting import true_vs_pred_scatter, ESCIT, PSILO
from utils.configs import load_ingredient_configs, match_ingredient_configs
from utils.statsalg import correlation_permutation_test
from models.utils import freeze_model
from datasets import AddLabel


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_scate', ingredients=[data_ingredient, 
                                              vgae_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'train_scate'
    jobid = 0
    seed = 0
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory with pre-trained model weights
    weights_dir = os.path.join('outputs', 'weights')
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']} 

    # Training configurations
    prediction_head_type = 'Ridge'
    n_pca_components = 0  # If > 0, apply PCA before fitting regression head
    standardize_data = True  # If True, standardize features before fitting regression head
    n_permutations = 1000  # Number of permutations for correlation permutation test

    # Condition settings
    annotations_file = 'data/raw/psilodep2/annotations.csv'
    subject_id_col = 'Patient' # annotations[subject_id_col] == data[i].subject+1
    condition_specs = {'cond0': 'E', 'cond1': 'P'}  # Maps cond0/cond1 to actual condition names

    # T-learner prediction results files
    t0_pred_file = os.path.join('outputs', 'x_graphtrip', 'tlearners', 'prediction_results_P_from_E_avg.csv')
    t1_pred_file = os.path.join('outputs', 'x_graphtrip', 'tlearners', 'prediction_results_E_from_P_avg.csv')

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE and dataset configs from weights_dir
    ingredients = ['vgae_model', 'dataset']     
    previous_config = load_ingredient_configs(weights_dir, ingredients)

    # Various dataset related configs may mismatch, but other configs must match
    config_updates = copy.deepcopy(config)
    exceptions = ['graph_attrs', 'target', 'context_attrs', 
                  'graph_attrs_to_standardise', 'pooling_cfg']
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Make sure the annotations file exists
    annotations_file = config.get('annotations_file', 'data/raw/psilodep2/annotations.csv')
    annotations_file = add_project_root(annotations_file)
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f'Annotations file not found: {annotations_file}')
    
    # Check compatibility of the annotations file
    annotations = pd.read_csv(annotations_file)

    # "Condition" column must be present in the annotations file
    if 'Condition' not in annotations.columns:
        raise ValueError('"Condition" column not found in annotations file.')
    
    # Make sure the subject ID column is present in the annotations file
    subject_id_col = config.get('subject_id_col', 'Patient')
    if subject_id_col not in annotations.columns:
        raise ValueError(f'Subject ID column not found in annotations file: {subject_id_col}')
    
    # Validate condition_specs
    condition_specs = config.get('condition_specs', {'cond0': 'E', 'cond1': 'P'})
    if 'cond0' not in condition_specs or 'cond1' not in condition_specs:
        raise ValueError('condition_specs must contain both "cond0" and "cond1" keys.')
    
    # Assert that condition values exist in annotations
    unique_conditions = annotations['Condition'].unique()
    cond0_name = condition_specs['cond0']
    cond1_name = condition_specs['cond1']
    if cond0_name not in unique_conditions:
        raise ValueError(f'Condition "{cond0_name}" (cond0) not found in annotations. Available conditions: {unique_conditions}')
    if cond1_name not in unique_conditions:
        raise ValueError(f'Condition "{cond1_name}" (cond1) not found in annotations. Available conditions: {unique_conditions}')
    
    config_updates['condition_specs'] = condition_specs
    
    # Other config checks
    num_pretrained_models = previous_config['dataset']['num_folds']
    default_weight_filenames = {'vgae': [f'k{i}_vgae_weights.pth' for i in range(num_pretrained_models)]}
    weight_filenames = config.get('weight_filenames', default_weight_filenames)
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames
    
    # Check t-learner files exist
    assert "t0_pred_file" in config and "t1_pred_file" in config, \
        "t0_pred_file and t1_pred_file must be specified in the config."
    t0_pred_file = add_project_root(config['t0_pred_file'])
    t1_pred_file = add_project_root(config['t1_pred_file'])

    if not os.path.exists(t0_pred_file) or not os.path.exists(t1_pred_file):
        raise FileNotFoundError(f'T-learner files {t0_pred_file} and {t1_pred_file} do not exist.')
    
    # Dataset target should be None
    if 'dataset' in config:
        target = config['dataset'].get('target', None)
        if target is not None:
            raise ValueError("Dataset target should be None for CATE models.")
        config_updates['dataset']['target'] = None    
    else:
        config_updates['dataset'] = {}
        config_updates['dataset']['target'] = None
    
    return config_updates

# Helper functions ------------------------------------------------------------
def extract_latent_representations(vgae, data, device, ite_labels=None):
    '''
    Extracts latent representations (z) from a VGAE for the given data.
    
    Parameters:
    ----------
    vgae: Trained VGAE model
    data: Dataset or list of data samples
    device: torch device
    ite_labels: dict mapping subject_id (0-indexed) to ITE label, or None to use batch.y
    
    Returns:
    -------
    z: numpy array of shape (n_samples, latent_dim) - latent representations
    y: numpy array of shape (n_samples,) - target values (ITEs if ite_labels provided)
    clinical_data: numpy array of shape (n_samples, n_clinical_features) - clinical data
    subject_ids: numpy array of shape (n_samples,) - subject IDs
    '''
    vgae = vgae.eval().to(device)
    with torch.no_grad():
        batch = next(iter(DataLoader(data, batch_size=len(data), shuffle=False))).to(device)
        context = get_context(batch)
        out = vgae(batch)
        z = vgae.readout(out.mu, context, batch.batch)
        
        # Convert to numpy
        z = z.cpu().numpy()
        if ite_labels is not None:
            # Use ITE labels from dictionary
            subject_ids = batch.subject.cpu().numpy().flatten()
            y = np.array([ite_labels[sid] for sid in subject_ids])
        else:
            y = batch.y.cpu().numpy().flatten()
        clinical_data = batch.graph_attr.cpu().numpy()
        subject_ids = batch.subject.cpu().numpy().flatten()
    
    return z, y, clinical_data, subject_ids

def create_prediction_model(model_type: str, seed: int):
    '''
    Creates a prediction model based on the model type with fixed parameters.
    
    Parameters:
    ----------
    model_type: str - One of 'Ridge', 'ElasticNet', 'RandomForestRegressor', 'HistGradientBoostingRegressor'
    seed: int - Random seed
    
    Returns:
    -------
    model: sklearn model instance
    '''
    if model_type == 'Ridge':
        return Ridge(alpha=1.0, random_state=seed)
    elif model_type == 'ElasticNet':
        return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=seed)
    elif model_type == 'RandomForestRegressor':
        return RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=seed)
    elif model_type == 'HistGradientBoostingRegressor':
        return HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=5, random_state=seed)
    else:
        raise ValueError(f'Unknown model_type: {model_type}.')

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    prediction_head_type = _config['prediction_head_type']
    n_pca_components = _config['n_pca_components']
    standardize_data = _config['standardize_data']
    n_permutations = _config['n_permutations']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']

    # Annotations config
    annotations_file = add_project_root(_config['annotations_file'])
    subject_id_col = _config['subject_id_col']

    # T-learner files
    t0_pred_file = add_project_root(_config['t0_pred_file'])
    t1_pred_file = add_project_root(_config['t1_pred_file'])

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []  

    # Load T-learner prediction results
    t0_pred_results = pd.read_csv(t0_pred_file)  # results from model trained on condition 0 (usually escitalopram)
    t1_pred_results = pd.read_csv(t1_pred_file)  # results from model trained on condition 1 (usually psilocybin)

    # Compute the ITEs
    t0_pred_results['ITE'] = t0_pred_results['label'] - t0_pred_results['prediction']  # true_cond1 - pred_cond0
    t1_pred_results['ITE'] = t1_pred_results['prediction'] - t1_pred_results['label']  # pred_cond1 - true_cond0

    # Verify subject IDs are unique across t0 and t1 results
    assert len(set(t0_pred_results['subject_id']).intersection(set(t1_pred_results['subject_id']))) == 0, \
        "Subject IDs must be unique across t0 and t1 results."
    
    # Create combined ITE label dictionary (subject_id is 0-indexed in the data)
    # For cond0: use ITEs from t1_pred_results (trained on cond1, predicting for cond0 patients)
    # For cond1: use ITEs from t0_pred_results (trained on cond0, predicting for cond1 patients)
    ite_labels_cond0 = dict(zip(t1_pred_results['subject_id'], t1_pred_results['ITE']))
    ite_labels_cond1 = dict(zip(t0_pred_results['subject_id'], t0_pred_results['ITE']))
    # Combine into single dictionary for all subjects
    ite_labels = {**ite_labels_cond0, **ite_labels_cond1}

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load annotations to get condition mappings
    # Note: the subject IDs in the annotations file are 1-indexed, but the subject IDs in the data are 0-indexed !
    annotations = pd.read_csv(annotations_file)
    
    # Get condition names from config
    condition_specs = _config.get('condition_specs', {'cond0': 'E', 'cond1': 'P'})
    cond0 = condition_specs['cond0']
    cond1 = condition_specs['cond1']
    logger.info(f'Training single CATE model on combined data from both conditions: {cond0} and {cond1}')

    # Create condition mapping: subject_id -> condition
    subject_to_condition = {}
    for _, row in annotations.iterrows():
        subject_id = row[subject_id_col] - 1  # Convert to 0-indexed
        condition = row['Condition']
        subject_to_condition[subject_id] = condition

    # Load pretrained VGAEs with exclude_module='pooling'
    pretrained_vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], 
                                          device=device, exclude_module='pooling')

    # Get test fold indices
    test_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)

    # Assert LOOCV: each sample should have a corresponding VGAE that had it in its test set
    assert len(data) == max(test_indices) + 1, \
        f"Expected LOOCV: len(data)={len(data)}, max(test_indices)={max(test_indices)}. " \
        f"Should have len(data) == max(test_indices) + 1"

    # Create mapping from data index to condition
    data_idx_to_condition = {}
    for i in range(len(data)):
        subject_id = data[i].subject.item()
        if subject_id in subject_to_condition:
            data_idx_to_condition[i] = subject_to_condition[subject_id]

    # Get sample indices for each condition (for logging only)
    cond0_indices = np.array([i for i in range(len(data)) if data_idx_to_condition.get(i) == cond0])
    cond1_indices = np.array([i for i in range(len(data)) if data_idx_to_condition.get(i) == cond1])
    
    logger.info(f'Found {len(cond0_indices)} samples for {cond0}, {len(cond1_indices)} samples for {cond1} (total: {len(data)})')

    # Load and freeze all VGAEs
    for i, vgae in enumerate(pretrained_vgaes):
        pretrained_vgaes[i] = freeze_model(vgae.to(device))

    # Train single CATE model on combined data with LOOCV -----------------------
    start_time = time()

    # Store results for all samples
    outputs = init_outputs_dict(data)
    pred_models = []
    pca_transformers = []
    pca_var_explained = []
    scalers = []
    
    # Get all sample indices (for LOOCV over all samples)
    all_indices = np.arange(len(data))

    logger.info(f'Training single CATE model on combined data (LOOCV over {len(all_indices)} samples)')
    
    for held_out_idx in tqdm(all_indices, desc='LOOCV (combined)', disable=not verbose):
        # Get the VGAE that had this held-out sample in its test set during pretraining
        vgae_idx = test_indices[held_out_idx]
        vgae = pretrained_vgaes[vgae_idx]
        
        # Training samples: all other samples (from both conditions)
        train_indices = all_indices[all_indices != held_out_idx]
        train_data_subset = data[train_indices]
        
        # Extract latent representations for training set with ITE labels
        z_train, y_train, clinical_train, subject_train = extract_latent_representations(
            vgae, train_data_subset, device, ite_labels=ite_labels)
        
        # Prepare training features
        x_train = np.concatenate([z_train, clinical_train], axis=1)
        original_dim = x_train.shape[1]
        
        # Standardize features before PCA or model fitting
        scaler = None
        if standardize_data:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
        scalers.append(scaler)

        # Apply PCA if requested
        pca = None
        if n_pca_components > 0:
            pca = PCA(n_components=n_pca_components, random_state=seed)
            x_train = pca.fit_transform(x_train)
            var_expl_dict = {f'PC{i+1}': pca.explained_variance_ratio_[i] for i in range(n_pca_components)}
            pca_var_explained.append(var_expl_dict)
            if verbose:
                logger.info(f"LOOCV, held_out_{held_out_idx}: Applied PCA, reduced from {original_dim} to {n_pca_components} dimensions")
                logger.info(f"LOOCV, held_out_{held_out_idx}: PCA variance explained: {var_expl_dict}")
        else:
            pca_var_explained.append(None)
        pca_transformers.append(pca)

        # Create and train prediction model
        pred_model = create_prediction_model(prediction_head_type, seed)
        pred_model.fit(x_train, y_train)
        pred_models.append(pred_model)
        
        # Evaluate on held-out sample
        held_out_data = data[np.array([held_out_idx])]
        z_test, y_test, clinical_test, subject_test = extract_latent_representations(
            vgae, held_out_data, device, ite_labels=ite_labels)
        x_test = np.concatenate([z_test, clinical_test], axis=1)
        
        # Apply standardization transformation if used
        if scaler is not None:
            x_test = scaler.transform(x_test)
        
        # Apply PCA transformation if used
        if pca is not None:
            x_test = pca.transform(x_test)
        
        y_pred = pred_model.predict(x_test)
        
        # Store outputs
        outputs_dict = {'prediction': y_pred, 
                       'label': y_test, 
                       'subject_id': subject_test,
                       'clinical_data': []}
        for sub in range(len(subject_test)):
            outputs_dict['clinical_data'].append(tuple(clinical_test[sub, :]))
        
        update_best_outputs(outputs, outputs_dict, _config['dataset']['graph_attrs'])
        
        # Evaluate on held-out sample
        if len(y_test) > 0:
            mae = np.mean(np.abs(y_pred - y_test))
            ex.log_scalar(f'loocv/held_out_{held_out_idx}/mae', mae)
        
        # Save model weights if requested
        if save_weights:
            import pickle
            model_path = os.path.join(output_dir, f'held_out_{held_out_idx}_pred_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(pred_model, f)
            if scaler is not None:
                scaler_path = os.path.join(output_dir, f'held_out_{held_out_idx}_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            if pca is not None:
                pca_path = os.path.join(output_dir, f'held_out_{held_out_idx}_pca.pkl')
                with open(pca_path, 'wb') as f:
                    pickle.dump(pca, f)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save PCA variance explained -----------------------------------------------
    if len(pca_var_explained) > 0 and any(v is not None for v in pca_var_explained):
        # Filter out None values (where PCA was not applied)
        var_expl_list = [v for v in pca_var_explained if v is not None]
        if len(var_expl_list) > 0:
            pca_var_explained_df = pd.DataFrame(var_expl_list)
            pca_var_explained_file = os.path.join(output_dir, 'pca_var_explained.csv')
            pca_var_explained_df.to_csv(pca_var_explained_file, index=False)
            logger.info(f"Saved PCA variance explained to {pca_var_explained_file}")
            
            # Log summary statistics for variance explained
            if n_pca_components > 0:
                for pc_idx in range(n_pca_components):
                    pc_name = f'PC{pc_idx+1}'
                    mean_var = pca_var_explained_df[pc_name].mean()
                    std_var = pca_var_explained_df[pc_name].std()
                    ex.log_scalar(f'pca_var_explained/{pc_name}/mean', mean_var)
                    ex.log_scalar(f'pca_var_explained/{pc_name}/std', std_var)
                    logger.info(f"PCA {pc_name}: mean={mean_var:.4f}, std={std_var:.4f}")

    # Process and save outputs --------------------------------------------------
    outputs_df = pd.DataFrame(outputs)
    outputs_df = add_drug_condition_to_outputs(outputs_df, _config['dataset']['study'])
    
    # Save prediction results
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    outputs_df.to_csv(data_file, index=False)
    
    # Calculate and save metrics
    r, p, mae, mae_std = evaluate_regression(outputs_df)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)
    
    # Log metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Results: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")
    
    # Permutation test for correlation coefficient
    y_true = outputs_df['label'].values
    y_pred = outputs_df['prediction'].values
    perm_hist_path = os.path.join(output_dir, 'perm_rs.csv')
    perm_plot_path = os.path.join(output_dir, 'corr_permutation_hist.png')
    perm_title = 'Permutation test for correlation'
    
    perm_results = correlation_permutation_test(
        y_true=y_true,
        y_pred=y_pred,
        n_permutations=n_permutations,
        seed=seed,
        make_plot=True,
        save_path=perm_plot_path,
        title=perm_title
    )
    
    # Save permutation correlations
    pd.DataFrame({'perm_r': perm_results['null_distribution']}).to_csv(perm_hist_path, index=False)
    
    # Log p-value
    logger.info(f"Permutation test: observed r = {perm_results['observed_r']:.4f}, "
                f"null mean = {perm_results['null_mean']:.4f}, "
                f"null sd = {perm_results['null_std']:.4f}, "
                f"p = {perm_results['p_value']:.4g}, n_perm = {len(perm_results['null_distribution'])}")
    
    # Log permutation test results
    ex.log_scalar(f'permutation_test/observed_r', perm_results['observed_r'])
    ex.log_scalar(f'permutation_test/null_mean', perm_results['null_mean'])
    ex.log_scalar(f'permutation_test/null_std', perm_results['null_std'])
    ex.log_scalar(f'permutation_test/p_value', perm_results['p_value'])
    
    image_files.append(perm_plot_path)
    if not verbose:
        plt.close()
    
    # Create true vs predicted scatter plot
    title = f'CATE predictions: r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    save_path = os.path.join(output_dir, 'true_vs_predicted.png')
    true_vs_pred_scatter(outputs_df, title=title, save_path=save_path)
    image_files.append(save_path)

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close('all')
