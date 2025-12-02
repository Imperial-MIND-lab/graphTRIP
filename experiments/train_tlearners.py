'''
In this experiment, we train t-learners (separate models for each condition) 
by loading a pre-trained VGAE and training Ridge regression heads on the 
latent representations + clinical data for each condition separately.

Dependencies:
- data/raw/{study}/annotations.csv
- Pre-trained VGAE weights in weights_dir

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
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA
from typing import Dict
from sklearn.preprocessing import StandardScaler

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.plotting import true_vs_pred_scatter, PSILO, ESCIT
from utils.configs import load_ingredient_configs, match_ingredient_configs, load_configs_from_json, fetch_job_config, make_config_grid
from utils.statsalg import correlation_permutation_test, analyze_coefficient_sparsity
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_tlearners', ingredients=[data_ingredient, 
                                              vgae_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'train_tlearners'
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
    prediction_head_cfg_file = 'experiments/configs/sklearn_heads.json'
    prediction_head_cfg_id_0 = 0  # Config ID for models trained on cond0
    prediction_head_cfg_id_1 = None  # Config ID for models trained on cond1; if None use the same as cond0
    n_pca_components = 0  # If > 0, apply PCA before fitting regression head
    standardize_data = True  # If True, standardize features before fitting regression head
    n_permutations = 1000  # Number of permutations for correlation permutation test
    treated_pooling_type = None # If None, use the same pooling for both conditions

    # Condition settings
    annotations_file = 'data/raw/psilodep2/annotations.csv'
    subject_id_col = 'Patient' # annotations[subject_id_col] == data[i].subject+1
    condition_specs = {'cond0': 'E', 'cond1': 'P'}  # Maps cond0/cond1 to actual condition names

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
                  'batch_size', 'num_folds', 'val_split' # training LOOCV, so this has no effect
                  ]
    # Exclude pooling since we're reinitializing it
    exceptions.append('pooling_cfg')
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

    # Pooling config
    treated_pooling_type = config.get('treated_pooling_type', None)
    if treated_pooling_type is not None:
        valid_pooling_types = ['MeanPooling', 'MeanStdPooling', 'DeepSetsMomentPooling']
        if treated_pooling_type not in valid_pooling_types:
            raise ValueError(f'treated_pooling_type must be one of {valid_pooling_types}, got {treated_pooling_type}')
    config_updates['treated_pooling_type'] = treated_pooling_type
    
    return config_updates

# Helper functions ------------------------------------------------------------
def extract_latent_representations(vgae, data, device):
    '''
    Extracts latent representations (z) from a VGAE for the given data.
    
    Parameters:
    ----------
    vgae: Trained VGAE model
    data: Dataset or list of data samples
    device: torch device
    
    Returns:
    -------
    z: numpy array of shape (n_samples, latent_dim) - latent representations
    y: numpy array of shape (n_samples,) - target values
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
        y = batch.y.cpu().numpy().flatten()
        clinical_data = batch.graph_attr.cpu().numpy()
        subject_ids = batch.subject.cpu().numpy().flatten()
    
    return z, y, clinical_data, subject_ids

def get_model_type_and_config(prediction_head_cfg_file: str, prediction_head_cfg_id: int):
    '''
    Gets the model type and configuration from the config file based on the ID.
    
    Parameters:
    ----------
    prediction_head_cfg_file: str - Path to JSON file with model configs
    prediction_head_cfg_id: int - Index of config to use across all model types
    
    Returns:
    -------
    model_type: str - The model type
    model_config: dict - The configuration for the model
    '''
    # Load configs from JSON file
    configs = load_configs_from_json(prediction_head_cfg_file)
    
    # Iterate through model types in order and accumulate configs
    current_id = 0
    for model_type, model_config_ranges in configs.items():
        # Generate all config combinations for this model type
        model_configs = list(make_config_grid(model_config_ranges))
        num_configs = len(model_configs)
        
        # Check if the requested ID falls within this model type's configs
        if prediction_head_cfg_id < current_id + num_configs:
            # Get the specific config for this model type
            local_id = prediction_head_cfg_id - current_id
            model_config = model_configs[local_id]
            return model_type, model_config
        
        current_id += num_configs
    
    # If we get here, the ID exceeds all available configs
    total_configs = sum(len(list(make_config_grid(ranges))) for ranges in configs.values())
    raise IndexError(f'prediction_head_cfg_id {prediction_head_cfg_id} exceeds the number of available configurations ({total_configs}).')

@ex.capture
def create_prediction_model(model_type: str, model_config: dict, seed: int):
    '''Creates a prediction model based on the model type and configuration.'''
    # Map model type strings to sklearn classes
    model_classes = {
        'Ridge': Ridge,
        'Lasso': Lasso,
        'ElasticNet': ElasticNet,
        'RandomForestRegressor': RandomForestRegressor,
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'SVR': SVR,
        'KernelRidge': KernelRidge,
        'KNeighborsRegressor': KNeighborsRegressor,
        'MLPRegressor': MLPRegressor,
        'GaussianProcessRegressor': GaussianProcessRegressor,
    }
    
    if model_type not in model_classes:
        raise ValueError(f'Unknown model_type: {model_type}. Available types: {list(model_classes.keys())}')
    
    model_class = model_classes[model_type]
    
    # Replace "<seed>" placeholder with actual seed value
    def replace_seed_placeholder(obj):
        if isinstance(obj, dict):
            return {k: replace_seed_placeholder(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_seed_placeholder(item) for item in obj]
        elif obj == "<seed>":
            return seed
        else:
            return obj
    
    model_config = replace_seed_placeholder(model_config)
    
    # Instantiate model with config
    return model_class(**model_config)

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    n_pca_components = _config['n_pca_components']
    standardize_data = _config['standardize_data']
    n_permutations = _config['n_permutations']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']

    # Pooling config
    treated_pooling_type = _config['treated_pooling_type']
    control_pooling_type = _config['vgae_model']['pooling_cfg']['model_type']
    change_pooling = (treated_pooling_type is not None) and (treated_pooling_type != control_pooling_type)

    # Require prediction_head_cfg_file
    prediction_head_cfg_file = _config['prediction_head_cfg_file']
    prediction_head_cfg_id_0 = _config['prediction_head_cfg_id_0']
    prediction_head_cfg_id_1 = _config['prediction_head_cfg_id_1'] or prediction_head_cfg_id_0
    if prediction_head_cfg_file is None:
        raise ValueError('prediction_head_cfg_file must be specified in config.')
    prediction_head_cfg_file = add_project_root(prediction_head_cfg_file)
    if not os.path.exists(prediction_head_cfg_file):
        raise FileNotFoundError(f'Config file not found: {prediction_head_cfg_file}')

    # Annotations config
    annotations_file = add_project_root(_config['annotations_file'])
    subject_id_col = _config['subject_id_col']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []  

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load annotations to get condition mappings
    # Note: the subject IDs in the annotations file are 1-indexed, but the subject IDs in the data are 0-indexed !
    annotations = pd.read_csv(annotations_file)
    
    # Get condition names from config
    condition_specs = _config.get('condition_specs', {'cond0': 'E', 'cond1': 'P'})
    cond1 = condition_specs['cond0']  # Keep naming as cond1 for backward compatibility
    cond2 = condition_specs['cond1']  # Keep naming as cond2 for backward compatibility
    logger.info(f'Training t-learners for conditions: {cond1} (cond0) and {cond2} (cond1)')

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

    # Get sample indices for each condition
    cond1_indices = np.array([i for i in range(len(data)) if data_idx_to_condition.get(i) == cond1])
    cond2_indices = np.array([i for i in range(len(data)) if data_idx_to_condition.get(i) == cond2])
    
    logger.info(f'Found {len(cond1_indices)} samples for {cond1}, {len(cond2_indices)} samples for {cond2}')

    # Load and freeze all VGAEs
    for i, vgae in enumerate(pretrained_vgaes):
        pretrained_vgaes[i] = freeze_model(vgae.to(device))

    # Train t-learners for each condition with LOOCV ---------------------------
    start_time = time()

    # Store results: same-condition (trained on cond, evaluated on cond)
    # and other-condition (trained on cond, evaluated on other cond)
    same_cond_outputs = {cond1: init_outputs_dict(data), 
                        cond2: init_outputs_dict(data)}
    other_cond_outputs = {cond1: init_outputs_dict(data),  # trained on cond1, evaluated on cond2
                         cond2: init_outputs_dict(data)}   # trained on cond2, evaluated on cond1
    pred_models = {cond1: [], cond2: []}
    pca_transformers = {cond1: [], cond2: []}
    pca_var_explained = {cond1: [], cond2: []}
    scalers = {cond1: [], cond2: []}
    prediction_head_types = {cond1: None, cond2: None}  # Store model types for each condition
    n_clinical_features = None  # Will be set during first iteration

    for train_cond in [cond1, cond2]:
        logger.info(f'Training t-learners for condition: {train_cond} (LOOCV)')
        
        # Get all sample indices for this condition
        train_cond_indices = cond1_indices if train_cond == cond1 else cond2_indices
        other_cond_indices = cond2_indices if train_cond == cond1 else cond1_indices
        other_cond = cond2 if train_cond == cond1 else cond1
        
        # Determine prediction head config ID for this condition
        # cond1 corresponds to cond0, cond2 corresponds to cond1
        if train_cond == cond1:
            prediction_head_cfg_id = prediction_head_cfg_id_0
        else:  # train_cond == cond2
            prediction_head_cfg_id = prediction_head_cfg_id_1
        
        # Infer prediction_head_type from config file and ID
        prediction_head_type, prediction_head_config = get_model_type_and_config(prediction_head_cfg_file, prediction_head_cfg_id)
        prediction_head_types[train_cond] = prediction_head_type
        logger.info(f'Using prediction head type for {train_cond}: {prediction_head_type} with config: {prediction_head_config}')
        
        for held_out_idx in tqdm(train_cond_indices, desc=f'LOOCV ({train_cond})', disable=not verbose):
            # Get the VGAE that had this held-out sample in its test set during pretraining
            vgae_idx = test_indices[held_out_idx]
            vgae = pretrained_vgaes[vgae_idx]
            
            # Reinitialize pooling for cond1 if treated_pooling_cfg is specified
            if train_cond == cond1 and change_pooling:
                latent_dim = _config['vgae_model']['params']['latent_dim']
                treated_pooling_cfg = {
                    'model_type': treated_pooling_type,
                    'params': {'pooling_dim': latent_dim} 
                }
                vgae.reinit_pooling(treated_pooling_cfg)
                if verbose:
                    logger.info(f"Reinitialized pooling to {treated_pooling_type} for {train_cond} (held_out_{held_out_idx})")
                    print(vgae.pooling)
            
            # Training samples: all other samples from this condition
            train_indices = train_cond_indices[train_cond_indices != held_out_idx]
            train_data_subset = data[train_indices]
            
            # Extract latent representations for training set
            z_train, y_train, clinical_train, subject_train = extract_latent_representations(
                vgae, train_data_subset, device)
            
            # Store number of clinical features (should be the same for all samples)
            if n_clinical_features is None:
                n_clinical_features = clinical_train.shape[1]
            
            # Prepare training features
            x_train = np.concatenate([z_train, clinical_train], axis=1)
            original_dim = x_train.shape[1]
            
            # Standardize features before PCA or model fitting
            scaler = None
            if standardize_data:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
            scalers[train_cond].append(scaler)

            # Apply PCA if requested
            pca = None
            if n_pca_components > 0:
                pca = PCA(n_components=n_pca_components, random_state=seed)
                x_train = pca.fit_transform(x_train)
                var_expl_dict = {f'PC{i+1}': pca.explained_variance_ratio_[i] for i in range(n_pca_components)}
                pca_var_explained[train_cond].append(var_expl_dict)
                if verbose:
                    logger.info(f"LOOCV ({train_cond}), held_out_{held_out_idx}: Applied PCA, reduced from {original_dim} to {n_pca_components} dimensions")
                    logger.info(f"LOOCV ({train_cond}), held_out_{held_out_idx}: PCA variance explained: {var_expl_dict}")
            else:
                pca_var_explained[train_cond].append(None)
            pca_transformers[train_cond].append(pca)

            # Create and train prediction model
            pred_model = create_prediction_model(prediction_head_type, prediction_head_config, seed=seed)
            pred_model.fit(x_train, y_train)
            pred_models[train_cond].append(pred_model)
            
            # Evaluate on held-out sample (same condition)
            held_out_data = data[np.array([held_out_idx])]
            z_test_same, y_test_same, clinical_test_same, subject_test_same = extract_latent_representations(
                vgae, held_out_data, device)
            x_test_same = np.concatenate([z_test_same, clinical_test_same], axis=1)
            
            # Apply standardization transformation if used
            if scaler is not None:
                x_test_same = scaler.transform(x_test_same)
            
            # Apply PCA transformation if used
            if pca is not None:
                x_test_same = pca.transform(x_test_same)
            
            y_pred_same = pred_model.predict(x_test_same)
            
            # Store outputs for same condition
            outputs_same = {'prediction': y_pred_same, 
                           'label': y_test_same, 
                           'subject_id': subject_test_same,
                           'clinical_data': []}
            for sub in range(len(subject_test_same)):
                outputs_same['clinical_data'].append(tuple(clinical_test_same[sub, :]))
            update_best_outputs(same_cond_outputs[train_cond], outputs_same, _config['dataset']['graph_attrs'])
            
            # Evaluate on all samples from the other condition
            if len(other_cond_indices) > 0:
                other_cond_data = data[other_cond_indices]
                z_test_other, y_test_other, clinical_test_other, subject_test_other = extract_latent_representations(
                    vgae, other_cond_data, device)
                x_test_other = np.concatenate([z_test_other, clinical_test_other], axis=1)
                
                # Apply standardization transformation if used
                if scaler is not None:
                    x_test_other = scaler.transform(x_test_other)
                
                # Apply PCA transformation if used
                if pca is not None:
                    x_test_other = pca.transform(x_test_other)
                
                y_pred_other = pred_model.predict(x_test_other)
                
                # Store outputs for other condition
                outputs_other = {'prediction': y_pred_other, 
                                'label': y_test_other, 
                                'subject_id': subject_test_other,
                                'clinical_data': []}
                for sub in range(len(subject_test_other)):
                    outputs_other['clinical_data'].append(tuple(clinical_test_other[sub, :]))
                update_best_outputs(other_cond_outputs[train_cond], outputs_other, _config['dataset']['graph_attrs'])
            
            # Evaluate on held-out sample
            if len(y_test_same) > 0:
                mae = np.mean(np.abs(y_pred_same - y_test_same))
                ex.log_scalar(f'loocv/{train_cond}/held_out_{held_out_idx}/mae', mae)
            
            # Save model weights if requested
            if save_weights:
                import pickle
                model_path = os.path.join(output_dir, f'held_out_{held_out_idx}_pred_model_{train_cond}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(pred_model, f)
                if scaler is not None:
                    scaler_path = os.path.join(output_dir, f'held_out_{held_out_idx}_scaler_{train_cond}.pkl')
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)
                if pca is not None:
                    pca_path = os.path.join(output_dir, f'held_out_{held_out_idx}_pca_{train_cond}.pkl')
                    with open(pca_path, 'wb') as f:
                        pickle.dump(pca, f)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Analyze coefficient sparsity (Gini coefficients) ---------------------------
    # Only if we don't use dimensionality reduction, and if we use a linear model
    linear_models = ['Ridge', 'Lasso', 'ElasticNet']
    # Check if both conditions use linear models
    both_linear = (n_pca_components == 0 and 
                   prediction_head_types[cond1] in linear_models and 
                   prediction_head_types[cond2] in linear_models)
    if both_linear:
        logger.info(f"Analyzing coefficient sparsity with n_clinical_features={n_clinical_features}")
        sparsity_results = analyze_coefficient_sparsity(pred_models, n_clinical_features)
        
        # Save sparsity results to CSV
        sparsity_csv_path = os.path.join(output_dir, 'coefficient_sparsity_results.csv')
        sparsity_results.to_csv(sparsity_csv_path, index=False)
        logger.info(f"Saved coefficient sparsity results to {sparsity_csv_path}")
        
        # Create violin plot with jittered points
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Prepare data for plotting
        cond1_gini = sparsity_results[sparsity_results['Condition'] == cond1]['Gini_Coefficient'].values
        cond2_gini = sparsity_results[sparsity_results['Condition'] == cond2]['Gini_Coefficient'].values
        
        # Create violin plot positions
        positions = [0, 1]
        violin_data = [cond1_gini, cond2_gini]
        condition_labels = [cond1, cond2]
        
        # Create violin plots
        violin_parts = ax.violinplot(violin_data, positions=positions, vert=True,
                                    showmeans=False, showextrema=False, widths=0.6)
        
        # Customize violins
        colors = [ESCIT, PSILO]
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)
            pc.set_edgecolor(colors[i])
            pc.set_linewidth(1.5)
        
        # Add jittered points
        np.random.seed(seed)  # For reproducibility
        for i, (pos, data_vals) in enumerate(zip(positions, violin_data)):
            x_jitter = np.random.normal(pos, 0.05, size=len(data_vals))
            ax.scatter(x_jitter, data_vals, color=colors[i], alpha=0.6, s=30, zorder=3)
        
        # Customize plot
        ax.set_xticks(positions)
        ax.set_xticklabels(condition_labels)
        ax.set_ylabel('Gini Coefficient', fontsize=12)
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_title('Coefficient Sparsity (Gini Coefficient) by Condition', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean lines
        for i, (pos, data_vals) in enumerate(zip(positions, violin_data)):
            mean_val = np.mean(data_vals)
            # Draw horizontal line segment for mean
            ax.plot([pos-0.3, pos+0.3], [mean_val, mean_val], 
                    color=colors[i], linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        
        # Set x-axis limits to ensure violins are visible
        ax.set_xlim(-0.5, 1.5)
        
        plt.tight_layout()
        
        # Save figure
        gini_plot_path = os.path.join(output_dir, 'gini_violins.png')
        plt.savefig(gini_plot_path)
        logger.info(f"Saved Gini coefficient violin plot to {gini_plot_path}")
        image_files.append(gini_plot_path)
        
        if not verbose:
            plt.close()
        
        # Log summary statistics
        for cond in [cond1, cond2]:
            cond_data = sparsity_results[sparsity_results['Condition'] == cond]['Gini_Coefficient']
            mean_gini = cond_data.mean()
            std_gini = cond_data.std()
            logger.info(f"Gini coefficient for {cond}: mean={mean_gini:.4f}, std={std_gini:.4f}")
            ex.log_scalar(f'coefficient_sparsity/{cond}/mean_gini', mean_gini)
            ex.log_scalar(f'coefficient_sparsity/{cond}/std_gini', std_gini)

    # Save PCA variance explained for each condition ---------------------------
    for cond in [cond1, cond2]:
        if len(pca_var_explained[cond]) > 0 and any(v is not None for v in pca_var_explained[cond]):
            # Filter out None values (where PCA was not applied)
            var_expl_list = [v for v in pca_var_explained[cond] if v is not None]
            if len(var_expl_list) > 0:
                pca_var_explained_df = pd.DataFrame(var_expl_list)
                pca_var_explained_file = os.path.join(output_dir, f'pca_var_explained_{cond}.csv')
                pca_var_explained_df.to_csv(pca_var_explained_file, index=False)
                logger.info(f"Saved PCA variance explained for {cond} to {pca_var_explained_file}")
                
                # Log summary statistics for variance explained
                if n_pca_components > 0:
                    for pc_idx in range(n_pca_components):
                        pc_name = f'PC{pc_idx+1}'
                        mean_var = pca_var_explained_df[pc_name].mean()
                        std_var = pca_var_explained_df[pc_name].std()
                        ex.log_scalar(f'pca_var_explained/{cond}/{pc_name}/mean', mean_var)
                        ex.log_scalar(f'pca_var_explained/{cond}/{pc_name}/std', std_var)
                        logger.info(f"PCA {pc_name} for {cond}: mean={mean_var:.4f}, std={std_var:.4f}")

    # Process and save outputs for each condition ------------------------------
    # Same-condition predictions (trained on cond, evaluated on cond)
    for cond in [cond1, cond2]:
        # Convert to DataFrame and add condition info
        outputs_df = pd.DataFrame(same_cond_outputs[cond])
        # Filter to only include samples from this condition
        cond_subject_ids = annotations[annotations['Condition'] == cond][subject_id_col].to_numpy() - 1
        cond_mask = outputs_df['subject_id'].isin(cond_subject_ids)
        outputs_df = outputs_df[cond_mask].reset_index(drop=True)
        
        outputs_df = add_drug_condition_to_outputs(outputs_df, _config['dataset']['study'])
        
        # Save prediction results
        data_file = os.path.join(output_dir, f'prediction_results_{cond}_same_cond.csv')
        outputs_df.to_csv(data_file, index=False)
        
        # Calculate and save metrics
        r, p, mae, mae_std = evaluate_regression(outputs_df)
        results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
        pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, f'final_metrics_{cond}_same_cond.csv'), index=False)
        
        # Log metrics
        for k, v in results.items():
            ex.log_scalar(f'final_prediction/{cond}_same_cond/{k}', v)
        logger.info(f"Results for {cond} (same condition): r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")
        
        # Permutation test for correlation coefficient
        y_true = outputs_df['label'].values
        y_pred = outputs_df['prediction'].values
        perm_hist_path = os.path.join(output_dir, f'perm_rs_{cond}_same_cond.csv')
        perm_plot_path = os.path.join(output_dir, f'corr_permutation_hist_{cond}_same_cond.png')
        perm_title = f'Permutation test for correlation ({cond}, same condition)'
        
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
        logger.info(f"Permutation test ({cond}, same condition): observed r = {perm_results['observed_r']:.4f}, "
                    f"null mean = {perm_results['null_mean']:.4f}, "
                    f"null sd = {perm_results['null_std']:.4f}, "
                    f"p = {perm_results['p_value']:.4g}, n_perm = {len(perm_results['null_distribution'])}")
        
        # Log permutation test results
        ex.log_scalar(f'permutation_test/{cond}_same_cond/observed_r', perm_results['observed_r'])
        ex.log_scalar(f'permutation_test/{cond}_same_cond/null_mean', perm_results['null_mean'])
        ex.log_scalar(f'permutation_test/{cond}_same_cond/null_std', perm_results['null_std'])
        ex.log_scalar(f'permutation_test/{cond}_same_cond/p_value', perm_results['p_value'])
        
        image_files.append(perm_plot_path)
        if not verbose:
            plt.close()
        
        # Create true vs predicted scatter plot
        title = f'{cond} (same condition): r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
        save_path = os.path.join(output_dir, f'true_vs_predicted_{cond}_same_cond.png')
        true_vs_pred_scatter(outputs_df, title=title, save_path=save_path)
        image_files.append(save_path)

    # Other-condition predictions (trained on cond1, evaluated on cond2, and vice versa)
    for train_cond in [cond1, cond2]:
        other_cond = cond2 if train_cond == cond1 else cond1
        
        # Convert to DataFrame
        outputs_df = pd.DataFrame(other_cond_outputs[train_cond])
        
        # Filter to only include samples from the other condition
        other_cond_subject_ids = annotations[annotations['Condition'] == other_cond][subject_id_col].to_numpy() - 1
        cond_mask = outputs_df['subject_id'].isin(other_cond_subject_ids)
        outputs_df = outputs_df[cond_mask].reset_index(drop=True)
        
        if len(outputs_df) == 0:
            logger.warning(f'No predictions for {other_cond} from models trained on {train_cond}')
            continue
        
        # Average predictions by subject_id (each subject has multiple predictions from different CV folds)
        # Identify columns to aggregate
        core_cols = ['prediction', 'label', 'subject_id']
        clinical_cols = [col for col in outputs_df.columns if col not in core_cols]
        
        # Build aggregation dictionary
        agg_dict = {
            'prediction': 'mean',
            'label': 'first',  # Labels should be the same for all rows of the same subject
        }
        # Add all clinical data columns (take first occurrence for each subject)
        for col in clinical_cols:
            agg_dict[col] = 'first'
        
        # Group by subject_id and aggregate
        grouped = outputs_df.groupby('subject_id').agg(agg_dict).reset_index()
        
        outputs_df_avg = grouped
        outputs_df_avg = add_drug_condition_to_outputs(outputs_df_avg, _config['dataset']['study'])
        
        # Save averaged prediction results
        data_file = os.path.join(output_dir, f'prediction_results_{other_cond}_from_{train_cond}_avg.csv')
        outputs_df_avg.to_csv(data_file, index=False)
        
        # Calculate and save metrics
        r, p, mae, mae_std = evaluate_regression(outputs_df_avg)
        results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
        pd.DataFrame(results, index=[0]).to_csv(
            os.path.join(output_dir, f'final_metrics_{other_cond}_from_{train_cond}_avg.csv'), index=False)
        
        # Log metrics
        for k, v in results.items():
            ex.log_scalar(f'final_prediction/{other_cond}_from_{train_cond}_avg/{k}', v)
        logger.info(f"Results for {other_cond} (from {train_cond}, averaged): r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")
        
        # Create true vs predicted scatter plot
        title = f'{other_cond} (from {train_cond}, averaged): r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
        save_path = os.path.join(output_dir, f'true_vs_predicted_{other_cond}_from_{train_cond}_avg.png')
        true_vs_pred_scatter(outputs_df_avg, title=title, save_path=save_path)
        image_files.append(save_path)

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close('all')
