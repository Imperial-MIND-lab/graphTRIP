'''
In this experiment, we train t-learners (separate models for each condition) 
by loading a pre-trained VGAE and training PyTorch MLP regression heads on the 
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
from typing import Dict

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.plotting import true_vs_pred_scatter, PSILO, ESCIT, plot_loss_curves
from utils.configs import load_ingredient_configs, match_ingredient_configs
from utils.statsalg import correlation_permutation_test
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_tlearners_torch', ingredients=[data_ingredient, 
                                              vgae_ingredient,
                                              mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'train_tlearners_torch'
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
    num_epochs = 100       # Number of epochs for MLP training
    mlp_lr = 0.001         # MLP learning rate
    num_z_samples = 1      # 0 for training MLP on the means of VGAE latent variables
    n_permutations = 1000  # Number of permutations for correlation permutation test

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
    exceptions = ['graph_attrs', 'target', 'context_attrs', 'batch_size', 'num_folds']
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
    
    return config_updates

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(mlp, mlp_lr):
    '''Creates the optimizer for MLP training.'''
    return torch.optim.Adam(mlp.parameters(), lr=mlp_lr)

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    num_epochs = _config['num_epochs']
    num_z_samples = _config['num_z_samples']
    n_permutations = _config['n_permutations']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']
    batch_size = _config['dataset']['batch_size']

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
    mlp_train_losses = {cond1: [], cond2: []}

    for train_cond in [cond1, cond2]:
        logger.info(f'Training t-learners for condition: {train_cond} (LOOCV)')
        
        # Get all sample indices for this condition
        train_cond_indices = cond1_indices if train_cond == cond1 else cond2_indices
        other_cond_indices = cond2_indices if train_cond == cond1 else cond1_indices
        other_cond = cond2 if train_cond == cond1 else cond1
        
        for held_out_idx in tqdm(train_cond_indices, desc=f'LOOCV ({train_cond})', disable=not verbose):
            # Get the VGAE that had this held-out sample in its test set during pretraining
            vgae_idx = test_indices[held_out_idx]
            vgae = pretrained_vgaes[vgae_idx]
            vgae = vgae.eval().to(device)
            vgae = freeze_model(vgae)
            
            # Training samples: all other samples from this condition
            train_indices = train_cond_indices[train_cond_indices != held_out_idx]
            train_data_subset = data[train_indices]
            
            # Create DataLoader for training
            actual_batch_size = len(train_data_subset) if batch_size == -1 else batch_size
            train_loader = DataLoader(train_data_subset, batch_size=actual_batch_size, shuffle=True)
            
            # Build MLP (input_dim = latent_dim + clinical_dim)
            mlp = build_mlp(latent_dim=vgae.readout_dim).to(device)
            optimizer = get_optimizer(mlp)
            
            # Train MLP
            train_losses = []
            for epoch in range(num_epochs):
                train_loss = train_mlp(mlp, train_loader, optimizer, device, 
                                       get_x=get_x_with_vgae, vgae=vgae, num_z_samples=num_z_samples)
                train_losses.append(train_loss)
                if verbose and epoch % 10 == 0:
                    logger.info(f"LOOCV ({train_cond}), held_out_{held_out_idx}, epoch {epoch}: train_loss={train_loss:.4f}")
            mlp_train_losses[train_cond].append(train_losses)
            
            # Evaluate on held-out sample (same condition)
            held_out_data = data[np.array([held_out_idx])]
            held_out_loader = DataLoader(held_out_data, batch_size=len(held_out_data), shuffle=False)
            outputs_same = get_mlp_outputs_nograd(mlp, held_out_loader, device, 
                                                 get_x=get_x_with_vgae, vgae=vgae, num_z_samples=0)
            update_best_outputs(same_cond_outputs[train_cond], outputs_same)
            
            # Evaluate on all samples from the other condition
            if len(other_cond_indices) > 0:
                other_cond_data = data[other_cond_indices]
                other_cond_loader = DataLoader(other_cond_data, batch_size=len(other_cond_data), shuffle=False)
                outputs_other = get_mlp_outputs_nograd(mlp, other_cond_loader, device,
                                                      get_x=get_x_with_vgae, vgae=vgae, num_z_samples=0)
                update_best_outputs(other_cond_outputs[train_cond], outputs_other)
            
            # Save model weights if requested
            if save_weights:
                model_path = os.path.join(output_dir, f'held_out_{held_out_idx}_mlp_{train_cond}.pth')
                torch.save(mlp.state_dict(), model_path)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Plot loss curves for each condition --------------------------------------
    for cond in [cond1, cond2]:
        if len(mlp_train_losses[cond]) > 0:
            # Convert loss data structure: from list of lists to dict with fold indices
            # mlp_train_losses[cond] is a list of lists (one per LOOCV fold)
            # plot_loss_curves expects a dict with keys 0, 1, 2, ... and values as lists of losses per epoch
            train_loss_dict = {i: losses for i, losses in enumerate(mlp_train_losses[cond])}
            # Create empty test_loss and val_loss dicts (same structure, but empty lists)
            test_loss_dict = {i: [] for i in range(len(mlp_train_losses[cond]))}
            val_loss_dict = {i: [] for i in range(len(mlp_train_losses[cond]))}
            
            # Plot and save loss curves
            loss_curve_path = os.path.join(output_dir, f'mlp_loss_curves_{cond}.png')
            plot_loss_curves(train_loss_dict, test_loss_dict, val_loss_dict, save_path=loss_curve_path)
            image_files.append(loss_curve_path)
            logger.info(f"Saved loss curves for {cond} to {loss_curve_path}")

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
