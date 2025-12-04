'''
Trains CFRNet model.
Causal modelling approach. Similar to VGAE + MLP join training in train_jointly.py,
but uses two MLP heads, for predicting each treatment condition separately.

Authors: Hanna M. Tolle
Date: 2025-04-16
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
import torch.nn
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices
from utils.plotting import plot_vgae_reconstructions, plot_loss_curves, true_vs_pred_scatter
from preprocessing.metrics import get_rsn_mapping


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_cfrnet', ingredients=[data_ingredient, 
                                              vgae_ingredient, 
                                              mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'train_cfrnet'
    jobid = 0
    seed = 0
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_fold_indices_file = None # If provided, determines the testfold indices

    # Training configurations
    lr = 0.001            # Learning rate.
    num_epochs = 300      # Number of epochs to train.
    num_z_samples = 1     # 0 for training MLP on the means of VGAE latent variables.
    alpha = 0.5           # Loss = alpha*vgae_loss + (1-alpha)*mlp_loss
    beta = 1.0            # Loss = alpha*vgae_loss + (1-alpha)*mlp_loss + beta*mmd_loss
    balance_treatment = True  # whether to balance on treatment for k-fold CV.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # MLP has to be a CFRHead
    if 'mlp_model' in config:
        model_type = config['mlp_model'].get('model_type', 'CFRHead')
        if model_type != 'CFRHead':
            raise ValueError('MLP model_type must be CFRHead.')
        config['mlp_model']['model_type'] = 'CFRHead'
    else:
        config['mlp_model'] = {'model_type': 'CFRHead'}

    # Dataset configuration checks
    if 'dataset' in config:

        # drug_condition has to be None 
        # (because this fixes the treatment for all patients to a constant value)
        drug_condition = config['dataset'].get('drug_condition', None)
        if drug_condition is not None:
            raise ValueError('drug_condition has to be None for CFRNet.')
        config['dataset']['drug_condition'] = None

        # There should be no context attributes
        context_attrs = config['dataset'].get('context_attrs', [])
        if len(context_attrs) > 0:
            raise ValueError('Context attributes have to be empty for CFRNet.')
        config['dataset']['context_attrs'] = []

        # The drug condition should not be part of the graph attributes
        graph_attrs = config['dataset'].get('graph_attrs', [])
        if 'Condition' in graph_attrs:
            raise ValueError('Condition has to be excluded from graph attributes for CFRNet.')
        config['dataset']['graph_attrs'] = [attr for attr in graph_attrs if attr != 'Condition']

    else:
        config['dataset'] = {'drug_condition': None, 
                             'context_attrs': [],
                             'graph_attrs': []}

    # Test fold indices file
    test_fold_indices_file = config.get('test_fold_indices_file', None)
    if test_fold_indices_file is not None:
        test_fold_indices_file = add_project_root(test_fold_indices_file)
        if not os.path.exists(test_fold_indices_file):
            raise FileNotFoundError(f"{test_fold_indices_file} not found")
        config['test_fold_indices_file'] = test_fold_indices_file

        # Check that the test_fold_indices_file comes from a balanced (on 'Condition') run if balance_treatment is True (or unset/default True)
        balance_treatment = config.get('balance_treatment', True)
        if balance_treatment:
            import json
            testfold_dir = os.path.dirname(test_fold_indices_file)
            config_json_file = os.path.join(testfold_dir, 'config.json')
            if not os.path.exists(config_json_file):
                raise FileNotFoundError(f"Config file {config_json_file} not found in test_fold_indices_file directory.")
            with open(config_json_file, 'r') as f:
                testfold_config = json.load(f)
            balance_attrs = testfold_config.get('balance_attrs', None)
            if not (isinstance(balance_attrs, list) and 'Condition' in balance_attrs):
                raise RuntimeError("Cannot balance treatment if test_fold_indices_file from an unbalanced run is provided.")

    return config

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(vgae, mlp, lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    return torch.optim.Adam(list(mlp.parameters()) + list(vgae.parameters()), lr=lr)

def median_pairwise_distance(x):
    with torch.no_grad():
        dist = torch.cdist(x, x, p=2)  # shape (n, n)
        triu = dist.triu(1)  # upper triangle without diagonal
        return torch.median(triu[triu > 0])

@ex.capture
def compute_mmd(x_treated, x_control):
    # Estimate the MMD kernel width from the data
    mmd_sigma = median_pairwise_distance(torch.cat([x_treated, x_control], dim=0))

    # Compute the MMD
    def rbf_kernel(a, b, sigma):
        a_norm = (a ** 2).sum(1).reshape(-1, 1)
        b_norm = (b ** 2).sum(1).reshape(1, -1)
        dist = a_norm + b_norm - 2.0 * torch.mm(a, b.T)
        return torch.exp(-dist / (2 * sigma ** 2))

    K_tt = rbf_kernel(x_treated, x_treated, mmd_sigma)
    K_cc = rbf_kernel(x_control, x_control, mmd_sigma)
    K_tc = rbf_kernel(x_treated, x_control, mmd_sigma)

    m = x_treated.shape[0]
    n = x_control.shape[0]

    mmd = K_tt.sum() / (m * m) + K_cc.sum() / (n * n) - 2 * K_tc.sum() / (m * n)
    return mmd

@ex.capture
def train_models(vgae, mlp, loader, optimizer, device, 
                 num_z_samples, alpha, beta):
    '''
    Joint training of VGAE and MLP with MMD regularisation (CFRNet-like).
    
    alpha: weight for VGAE loss
    beta: weight for MMD loss
    '''
    vgae.train()
    mlp.train()
    train_loss = 0.

    for batch in loader: 
        batch = batch.to(device)             
        ytrue = get_labels(batch, num_z_samples) # (batch_size*num_z_samples,1)
        treatment = get_treatment(batch, num_z_samples).to(device)  # (batch_size*num_z_samples,1)

        optimizer.zero_grad()

        # Forward: VGAE representation + clinical data
        x, vgae_loss = get_x_with_vgae_loss(batch, device, vgae, num_z_samples)

        # MLP forward and loss
        ypred = mlp(x, treatment)
        mlp_loss = mlp.loss(ypred, ytrue)

        # MMD Loss (between treatment groups)
        treated_mask = (treatment == 1)   # psilocybin (1)
        control_mask = (treatment != 1)   # escitalopram (0)
        if treated_mask.sum() > 1 and control_mask.sum() > 1:
            mmd_loss = compute_mmd(x[treated_mask], x[control_mask])
        else:
            mmd_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = alpha * vgae_loss + (1 - alpha) * mlp_loss + beta * mmd_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(loader)

@ex.capture
def test_models(vgae, mlp, loader, device, num_z_samples):
    '''
    Evaluates the CFRNet model (VGAE + MLP with MMD regularization).
    
    Parameters:
    ----------
    vgae (torch.nn.Module): trained VGAE model
    cfr_head (torch.nn.Module): trained CFRNet head
    loader (torch.utils.data.DataLoader): test data loader
    device (torch.device): Device to use for testing
    num_z_samples (int): Number of samples to draw from the latent space

    Returns:
    -------
    vgae_test_loss (float): average VGAE loss across test dataset
    mlp_test_loss (float): average MLP loss across test dataset
    mmd_test_loss (float): average MMD loss across test dataset
    '''
    vgae.eval()
    mlp.eval()

    vgae_test_loss = 0.
    mlp_test_loss = 0.
    mmd_test_loss = 0. 

    with torch.no_grad():
        for batch in loader:
            # Get inputs and labels
            batch = batch.to(device)
            ytrue = get_labels(batch, num_z_samples)
            treatment = get_treatment(batch, num_z_samples).to(device)
            
            # Forward: VGAE representation + clinical data
            x, vgae_loss = get_x_with_vgae_loss(batch, device, vgae, num_z_samples)
            
            # MLP forward and loss
            ypred = mlp(x, treatment)
            mlp_loss = mlp.loss(ypred, ytrue)
            
            # MMD Loss (between treatment groups)
            treated_mask = (treatment == 1)   # psilocybin (1)
            control_mask = (treatment != 1)  # escitalopram (0)
            if treated_mask.sum() > 1 and control_mask.sum() > 1:
                mmd_loss = compute_mmd(x[treated_mask], x[control_mask])
            else:
                mmd_loss = torch.tensor(0.0, device=device)
            
            # Accumulate losses
            vgae_test_loss += vgae_loss.item()
            mlp_test_loss += mlp_loss.item()
            mmd_test_loss += mmd_loss.item()
    
    # Average losses across batches
    vgae_test_loss = vgae_test_loss / len(loader)
    mlp_test_loss = mlp_test_loss / len(loader)
    mmd_test_loss = mmd_test_loss / len(loader)
    
    return vgae_test_loss, mlp_test_loss, mmd_test_loss

@mlp_ingredient.capture
def get_cfrnet_outputs_nograd(mlp, loader, device, get_x, *args, **kwargs):
    '''
    Returns the outputs of the MLP. No gradients are computed.
    Parameters:
    ----------
    mlp (torch.nn.Module): trained MLP model.
    loader (Dataloader): pytorch geometric test data loader.
    device (torch.device): Device to use for testing.
    get_x (function): function to get the input data for the MLP.
                      This should be implemented by the experiment.
                      
    Returns:
    -------
    outputs (dict): clinical data, prediction and label of each pred.
                    {'prediction': [0]*len(test_dataset),
                     'label': [0]*len(test_dataset),
                     'clinical_data': [(data1, data2, ...)]*len(test_dataset),
                     f'std_{n_samples}': [0]*len(test_dataset)}
    '''
    mlp.eval()
    outputs = {key: [] for key in ['clinical_data', 'prediction', 'label', 'subject_id']}
    with torch.no_grad():
        for batch in loader:
            # Get inputs and labels
            batch = batch.to(device)
            ytrue = get_labels(batch, num_z_samples=0)
            treatment = get_treatment(batch, num_z_samples=0).to(device)
            clinical_data = batch.graph_attr
            subject_id = batch.subject
            
            # Get MLP predictions on latent means
            if 'num_z_samples' in kwargs:
                kwargs['num_z_samples'] = 0
            x = get_x(batch, device, *args, **kwargs) 
            ypred = mlp(x, treatment)                                   

            # Save outputs
            batch_size = clinical_data.shape[0]
            for sub in range(batch_size):
                outputs['clinical_data'].append(tuple(clinical_data[sub, :].tolist()))
            outputs['prediction'].extend(ypred.squeeze(-1).tolist())
            outputs['label'].extend(ytrue.squeeze(-1).tolist())
            outputs['subject_id'].extend(subject_id.tolist())

    return outputs

@mlp_ingredient.capture
def get_cfrnet_counterfactual_outputs(mlp, loader, device, get_x, k_fold, *args, **kwargs):
    '''
    Returns counterfactual predictions from both MLP heads (mlp0 and mlp1).
    No gradients are computed.
    
    Parameters:
    ----------
    mlp (torch.nn.Module): trained CFRHead model with mlp0 and mlp1.
    loader (Dataloader): pytorch geometric test data loader.
    device (torch.device): Device to use for testing.
    get_x (function): function to get the input data for the MLP.
    k_fold (int): The k-fold number for this model.
    *args, **kwargs: Additional arguments for get_x.
                      
    Returns:
    -------
    outputs_df (pd.DataFrame): DataFrame with columns:
                               - subject_id: subject IDs
                               - k_fold: k-fold number
                               - prediction_mlp0: predictions from mlp0 (escitalopram)
                               - prediction_mlp1: predictions from mlp1 (psilocybin)
                               - label: true labels
    '''
    mlp.eval()
    outputs = {
        'subject_id': [],
        'k_fold': [],
        'prediction_mlp0': [],
        'prediction_mlp1': [],
        'label': []
    }
    
    with torch.no_grad():
        for batch in loader:
            # Get inputs and labels
            batch = batch.to(device)
            ytrue = get_labels(batch, num_z_samples=0)
            subject_id = batch.subject
            
            # Get MLP input (latent means)
            if 'num_z_samples' in kwargs:
                kwargs['num_z_samples'] = 0
            x = get_x(batch, device, *args, **kwargs)
            
            # Get predictions from both heads
            pred_mlp0 = mlp.mlp0(x)  # escitalopram predictions
            pred_mlp1 = mlp.mlp1(x)  # psilocybin predictions
            
            # Save outputs
            batch_size = subject_id.shape[0]
            outputs['subject_id'].extend(subject_id.tolist())
            outputs['k_fold'].extend([k_fold] * batch_size)
            outputs['prediction_mlp0'].extend(pred_mlp0.squeeze(-1).tolist())
            outputs['prediction_mlp1'].extend(pred_mlp1.squeeze(-1).tolist())
            outputs['label'].extend(ytrue.squeeze(-1).tolist())
    
    return pd.DataFrame(outputs)

@ex.capture
def get_dataloaders(data, balance_treatment, test_fold_indices_file, seed):
    '''
    Returns dataloaders. If balance_attrs is not None, balances k-fold splits based on treatment.
    Note: For CFRNet, balance_attrs should typically be None or we always balance on treatment.
    '''
    # Option 1: Use test fold indices file
    if test_fold_indices_file is not None:
        test_indices_array = np.loadtxt(test_fold_indices_file, dtype=int)
        train_loaders, val_loaders, test_loaders, mean_std \
            = get_dataloaders_from_test_indices(data, test_indices_array, seed=seed)

        # For consistency with how the other options return test_indices
        num_folds = max(test_indices_array) + 1
        test_indices = [np.where(test_indices_array == fold)[0] for fold in range(num_folds)]

    # Option 2: Balance treatment
    elif balance_treatment:
        # For CFRNet, we balance based on treatment (not graph attributes)
        # So we use the treatment-based balancing function
        train_loaders, val_loaders, test_loaders, test_indices, mean_std \
            = get_treatment_balanced_kfold_dataloaders(data, seed=seed)

    # Option 3: Unbalanced
    else:
        train_loaders, val_loaders, test_loaders, test_indices, mean_std \
            = get_kfold_dataloaders(data, seed=seed)
    return train_loaders, val_loaders, test_loaders, test_indices, mean_std

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Unpack configs
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    alpha = _config['alpha']
    beta = _config['beta']
    num_folds = _config['dataset']['num_folds']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Get dataloaders
    data = load_data()
    add_treatment_transform(data)
    train_loaders, val_loaders, test_loaders, test_indices, mean_std \
        = get_dataloaders(data, seed=seed)
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Train-test loop ------------------------------------------------------------
    start_time = time()

    best_outputs = init_outputs_dict(data)
    vgae_train_loss, vgae_test_loss, vgae_val_loss = {}, {}, {}
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}
    mmd_train_loss, mmd_test_loss, mmd_val_loss = {}, {}, {}
    best_vgae_states, best_mlp_states = [], []
    counterfactual_preds_list = []  
    counterfactual_preds_df = None 

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        vgae_train_loss[k], vgae_test_loss[k], vgae_val_loss[k] = [], [], []
        mmd_train_loss[k], mmd_test_loss[k], mmd_val_loss[k] = [], [], []
        
        # Initialise models and optimizer
        vgae = build_vgae().to(device)
        mlp = build_mlp(latent_dim=vgae.readout_dim).to(device)
        optimizer = get_optimizer(vgae, mlp)

        # Best validation loss and early stopping counter
        best_val_loss = float('inf')
        best_vgae_state = None
        best_mlp_state = None

        for epoch in range(_config['num_epochs']):
            # Train VGAE and MLP
            _ = train_models(vgae, mlp, train_loaders[k], optimizer, device)
            
            # Compute training losses
            vgae_train_loss_epoch, mlp_train_loss_epoch, mmd_train_loss_epoch = \
                test_models(vgae, mlp, train_loaders[k], device)
            vgae_train_loss[k].append(vgae_train_loss_epoch)
            mlp_train_loss[k].append(mlp_train_loss_epoch)
            mmd_train_loss[k].append(mmd_train_loss_epoch)
            
            # Test VGAE and MLP
            vgae_test_loss_epoch, mlp_test_loss_epoch, mmd_test_loss_epoch = \
                test_models(vgae, mlp, test_loaders[k], device)
            vgae_test_loss[k].append(vgae_test_loss_epoch)
            mlp_test_loss[k].append(mlp_test_loss_epoch)
            mmd_test_loss[k].append(mmd_test_loss_epoch)

            # Log training and test losses
            ex.log_scalar(f'training/fold{k}/epoch/vgae_loss', vgae_train_loss_epoch)
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/vgae_loss', vgae_test_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/mmd_loss', mmd_test_loss_epoch)

            # Validate models, if applicable
            if len(val_loaders) > 0:
                vgae_val_loss_epoch, mlp_val_loss_epoch, mmd_val_loss_epoch = \
                    test_models(vgae, mlp, val_loaders[k], device)
                vgae_val_loss[k].append(vgae_val_loss_epoch)
                mlp_val_loss[k].append(mlp_val_loss_epoch)
                mmd_val_loss[k].append(mmd_val_loss_epoch)

                # Log validation losses
                ex.log_scalar(f'validation/fold{k}/epoch/vgae_loss', vgae_val_loss_epoch)
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss_epoch)
                ex.log_scalar(f'validation/fold{k}/epoch/mmd_loss', mmd_val_loss_epoch)
                
                # Save the best model if validation loss is at its minimum
                total_val_loss = alpha*vgae_val_loss_epoch + (1-alpha)*mlp_val_loss_epoch 
                total_val_loss += beta*mmd_val_loss_epoch
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

        # Save the test predictions (on VGAE latent means) of the best model
        outputs = get_cfrnet_outputs_nograd(mlp, test_loaders[k], device, 
                                         get_x=get_x_with_vgae, 
                                         vgae=vgae, num_z_samples=0)
        update_best_outputs(best_outputs, outputs)
        
        # Get counterfactual predictions from both MLP heads
        counterfactual_preds_df = get_cfrnet_counterfactual_outputs(
            mlp, test_loaders[k], device, 
            get_x=get_x_with_vgae, 
            k_fold=k,
            vgae=vgae, num_z_samples=0)
        counterfactual_preds_list.append(counterfactual_preds_df)

    # Print training time
    end_time = time()
    logger.info(f"Joint training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save ouputs as csv file
    best_outputs = pd.DataFrame(best_outputs)
    best_outputs = add_drug_condition_to_outputs(best_outputs, _config['dataset']['study'])
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    best_outputs.to_csv(data_file, index=False)
    
    # Combine counterfactual predictions from all folds
    if counterfactual_preds_list:
        counterfactual_preds_df = pd.concat(counterfactual_preds_list, ignore_index=True)
        counterfactual_preds_df = add_drug_condition_to_outputs(counterfactual_preds_df, _config['dataset']['study'])
        counterfactual_file = os.path.join(output_dir, 'counterfactual_predictions.csv')
        counterfactual_preds_df.to_csv(counterfactual_file, index=False)
        logger.info(f"Saved counterfactual predictions to {counterfactual_file}")

    # Save test fold assignments
    test_indices_file = save_test_indices(test_indices, output_dir)

    # Save final prediction results
    r, p, mae, mae_std = evaluate_regression(best_outputs)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Final results: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

    # Plot results ----------------------------------------------------------------
    # Load all trained VGAEs
    vgaes = load_vgaes_from_states_list(best_vgae_states, device)
    
    # Get reconstructions
    adj_orig_rcn, x_orig_rcn, fold_assignments = get_test_reconstructions(
        vgaes, data, test_indices, mean_std=mean_std)
    
    # Evaluate FC reconstructions
    adj_orig_rcn['metrics'] = evaluate_fc_reconstructions(adj_orig_rcn)
    ex.log_scalar('final_reconstruction/fc_corr', np.mean(adj_orig_rcn['metrics']['corr']))
    ex.log_scalar('final_reconstruction/fc_mae', np.mean(adj_orig_rcn['metrics']['mae']))

    # Evaluate node feature reconstructions
    if x_orig_rcn:
        x_orig_rcn['metrics'] = evaluate_x_reconstructions(x_orig_rcn)
        for feature in x_orig_rcn['feature_names']:
            ex.log_scalar(f'final_reconstruction/x_{feature}_corr', x_orig_rcn['metrics']['corr'][feature].mean())
            ex.log_scalar(f'final_reconstruction/x_{feature}_mae', x_orig_rcn['metrics']['mae'][feature].mean())
    
    # Some other details needed for plotting
    rsn_mapping, rsn_labels = get_rsn_mapping(data.atlas)
    save_path = os.path.join(output_dir, 'vgae.png')
    vrange = None
    if _config['dataset']['edge_attrs'][0] == 'functional_connectivity':
        vrange = (-0.7, 0.7)
    full_batch = DataLoader(data, batch_size=len(data), shuffle=False)
    full_batch = next(iter(full_batch))
    conditions = full_batch.treatment*2 - 1
    image_files += plot_vgae_reconstructions(adj_orig_rcn, 
                                             x_orig_rcn, 
                                             fold_assignments, 
                                             conditions=conditions,
                                             rsn_mapping=rsn_mapping,
                                             rsn_labels=rsn_labels,
                                             atlas=data.atlas,
                                             vrange=vrange, 
                                             save_path=save_path)
    
    # Loss curves
    plot_loss_curves(vgae_train_loss, vgae_test_loss, vgae_val_loss, save_path=os.path.join(output_dir, 'vgae_loss_curves.png'))
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=os.path.join(output_dir, 'mlp_loss_curves.png'))
    plot_loss_curves(mmd_train_loss, mmd_test_loss, mmd_val_loss, save_path=os.path.join(output_dir, 'mmd_loss_curves.png'))
    image_files += [os.path.join(output_dir, 'vgae_loss_curves.png'), 
                    os.path.join(output_dir, 'mlp_loss_curves.png'), 
                    os.path.join(output_dir, 'mmd_loss_curves.png')]

    # True vs predicted scatter
    title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    true_vs_pred_scatter(best_outputs, title=title, save_path=os.path.join(output_dir, 'true_vs_predicted.png'))
    image_files.append(os.path.join(output_dir, 'true_vs_predicted.png'))
    
    # Counterfactual predictions scatter plots
    if counterfactual_preds_df is not None:
        # Scatter plot for mlp0 (escitalopram) predictions
        true_vs_pred_scatter(counterfactual_preds_df,
                            xcol='prediction_mlp0', ycol='prediction_mlp1',
                            save_path=os.path.join(output_dir, 'mlp0_vs_mlp1.png'))
        image_files.append(os.path.join(output_dir, 'mlp0_vs_mlp1.png'))

        # Plot observed vs counterfactual predictions for psilocybin and escitalopram
        psilo_df = counterfactual_preds_df[counterfactual_preds_df['Condition'] == 1]
        escit_df = counterfactual_preds_df[counterfactual_preds_df['Condition'] == -1]
        psilo_df = psilo_df.rename(columns={'prediction_mlp0': 'Predicted escitalopram outcome', 
                                            'label': 'Observed psilocybin outcome'})
        escit_df = escit_df.rename(columns={'prediction_mlp1': 'Predicted psilocybin outcome', 
                                            'label': 'Observed escitalopram outcome'})
        true_vs_pred_scatter(psilo_df, xcol='Predicted escitalopram outcome', ycol='Observed psilocybin outcome',
                             save_path=os.path.join(output_dir, 'psilo_cf_vs_true.png'))
        true_vs_pred_scatter(escit_df, xcol='Predicted psilocybin outcome', ycol='Observed escitalopram outcome',
                             save_path=os.path.join(output_dir, 'escit_cf_vs_true.png'))
        image_files.append(os.path.join(output_dir, 'psilo_cf_vs_true.png'))
        image_files.append(os.path.join(output_dir, 'escit_cf_vs_true.png'))

    # Log images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close()
