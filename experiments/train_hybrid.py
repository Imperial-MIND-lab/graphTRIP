'''
Loads 2 types of pretrained VGAEs, freezes their weights, and trains 
a new MLP on the concatenated latent vectors of the 2 VGAEs. 

Authors: Hanna M. Tolle
Date: 2025-01-17
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
import logging
import matplotlib.pyplot as plt

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices, check_weights_exist
from utils.plotting import plot_loss_curves, true_vs_pred_scatter
from utils.configs import load_ingredient_configs, match_ingredient_configs
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_hybrid', ingredients=[data_ingredient, 
                                             vgae_ingredient, 
                                             mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'train_hybrid'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory with pre-trained model weights
    weights_dir = os.path.join('outputs', 'graphTRIP', 'weights')
    weights_dir2 = os.path.join('outputs', 'atlas_bound', 'weights')
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}

    # Training configurations
    num_epochs = 300
    lr = 0.001
    num_z_samples = 3   
    checkpoints = []

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])
    assert 'weights_dir2' in config, "weights_dir2 must be specified in config."
    weights_dir2 = add_project_root(config['weights_dir2'])

    # Check that the models were trained on the same dataset
    previous_config = load_ingredient_configs(weights_dir, ['dataset'])
    previous_config2 = load_ingredient_configs(weights_dir2, ['dataset'])
    current_config = config.get('dataset', {})
    must_match = ['target', 'study', 'session']
    for key in must_match:
        assert previous_config['dataset'][key] == previous_config2['dataset'][key], \
            f"Models must be trained on the same {key}."
        if key in current_config:
            assert previous_config['dataset'][key] == current_config[key], \
                f"Hybrid model must be trained on the same {key} as the previous models."

    # Load and match configs for weights_dir 1 (other are set in the experiment)
    exceptions = ['num_nodes', 'graph_attrs', 'num_graph_attrs', 'target']
    config_updates = load_matching_configs(config, weights_dir, exceptions)
    return config_updates

def load_matching_configs(config, weights_dir, exceptions=[]):
    # Load the VGAE, MLP and dataset configs from weights_dir
    ingredients = ['vgae_model', 'dataset']
    previous_config = load_ingredient_configs(weights_dir, ingredients)

    # Match configs of relevant ingredients
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Ensure that num_graph_attrs and num_context_attrs are set correctly
    config_updates['vgae_model']['params']['num_graph_attrs'] = len(config_updates['dataset']['graph_attrs'])

    return config_updates

def load_trained_vgaes2(vgae2_config, weights_dir, weight_filenames, device):
    '''
    Loads trained VGAEs from a directory.
    '''
    vgaes = []
    for weight_file in weight_filenames:
        vgae = build_vgae_from_config(vgae2_config).to(device)
        state_dict = torch.load(os.path.join(weights_dir, weight_file))
        vgae.load_state_dict(state_dict)
        vgaes.append(vgae)
    return vgaes

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(mlp, lr):
    '''Creates the optimizer for the MLP.'''
    return torch.optim.Adam(mlp.parameters(), lr=lr)

def get_x_with_2vgaes(batch, device, vgae, batch2, vgae2, num_z_samples):
    '''
    Returns the MLP input (clinical data + VGAE latent features) for a batch.
    Use this function only when training the MLP independently of the VGAE, or
    for evaluating the final (trained) MLP. Because the VGAE is in eval mode.
    '''
    x = get_x_with_vgae(batch, device, vgae, num_z_samples)
    x2 = get_x_with_vgae(batch2, device, vgae2, num_z_samples)
    return torch.cat([x, x2], dim=1)

@ex.capture
def train_mlp_with_2vgaes(mlp, loader, loader2, vgae, vgae2, optimizer, device, num_z_samples):
    '''
    Performs training of the RegressionMLP.
    
    Parameters:
    -----------
    mlp (torch.nn.Module): RegressionMLP model
    loader (torch.utils.data.DataLoader): Training data loader
    loader2 (torch.utils.data.DataLoader): Training data loader for VGAE2
    vgae (torch.nn.Module): VGAE model
    vgae2 (torch.nn.Module): VGAE model
    optimizer (torch.optim.Optimizer): Optimizer
    device (torch.device): Device to use for training.
    num_z_samples (int): Number of samples to use for training.
    '''

    mlp.train()
    train_loss = 0.
    for batch, batch2 in zip(loader, loader2): 
        # Fetch data and labels from batch1 (they are the same)
        batch = batch.to(device)             
        ytrue = get_labels(batch, num_z_samples, device)

        # Forward pass
        optimizer.zero_grad()
        x = get_x_with_2vgaes(batch, device, vgae, batch2, vgae2, num_z_samples)
        ypred = mlp(x)                     
        loss = mlp.loss(ypred, ytrue)

        # Backpropagation
        loss.backward()                      
        optimizer.step()                     
        train_loss += loss.item()

    return train_loss/len(loader)

@ex.capture
def test_mlp_with_2vgaes(mlp, loader, loader2, vgae, vgae2, device, num_z_samples):
    '''
    Evaluates the RegressionMLP. 
    Parameters:
    ----------
    mlp (torch.nn.Module): trained RegressionMLP model.
    loader (torch.utils.data.DataLoader): test data loader.
    loader2 (torch.utils.data.DataLoader): test data loader for VGAE2
    device (torch.device): Device to use for testing.
    vgae (torch.nn.Module): VGAE model
    vgae2 (torch.nn.Module): VGAE model
    num_z_samples (int): Number of samples to use for testing.
    Returns:
    -------
    test_loss (float): average loss across test dataset.
    '''
    
    mlp.eval()
    test_loss = 0.
    with torch.no_grad():
        for batch, batch2 in zip(loader, loader2):
            # Get inputs and labels from batch1 (they are the same)
            batch = batch.to(device)
            y_true = get_labels(batch, num_z_samples, device)

            # Get MLP inputs
            x = get_x_with_2vgaes(batch, device, vgae, batch2, vgae2, num_z_samples)

            # Get MLP loss
            y_pred = mlp(x)                                   
            loss = mlp.loss(y_pred, y_true)                   
            test_loss += loss.item()

    return test_loss/len(loader)

def get_mlp_outputs_nograd_2vgaes(mlp, loader, loader2, vgae, vgae2, device, num_z_samples=0):
    '''
    Returns the outputs of the MLP. No gradients are computed.
    Parameters:
    ----------
    mlp (torch.nn.Module): trained MLP model.
    loader (Dataloader): pytorch geometric test data loader.
    loader2 (Dataloader): pytorch geometric test data loader for VGAE2
    device (torch.device): Device to use for testing.
    vgae (torch.nn.Module): VGAE model
    vgae2 (torch.nn.Module): VGAE model
    num_z_samples (int): Number of samples to use for testing.
                      
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
        for batch, batch2 in zip(loader, loader2):
            # Get inputs and labels
            batch = batch.to(device)
            ytrue = get_labels(batch, num_z_samples=0)
            clinical_data = batch.graph_attr
            subject_id = batch.subject
            
            # Get MLP predictions on latent means
            x = get_x_with_2vgaes(batch, device, vgae, batch2, vgae2, num_z_samples)
            ypred = mlp(x)                                   

            # Save outputs
            batch_size = clinical_data.shape[0]
            for sub in range(batch_size):
                outputs['clinical_data'].append(tuple(clinical_data[sub, :].tolist()))
            outputs['prediction'].extend(ypred.squeeze(-1).tolist())
            outputs['label'].extend(ytrue.squeeze(-1).tolist())
            outputs['subject_id'].extend(subject_id.tolist())

    return outputs

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Check weight files ------------------------------------------------------
    weights_dir = add_project_root(_config['weights_dir'])
    weights_dir2 = add_project_root(_config['weights_dir2'])
    weight_filenames = _config['weight_filenames']
    check_weights_exist(weights_dir, weight_filenames)
    check_weights_exist(weights_dir2, weight_filenames)

    # Ensure that both VGAE types had the same train/test split
    test_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)
    test_indices2 = np.loadtxt(os.path.join(weights_dir2, weight_filenames['test_fold_indices'][0]), dtype=int)
    assert np.array_equal(test_indices, test_indices2), "Train/test split must match between VGAE types."
    
    # Set up environment ------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    num_epochs = _config['num_epochs']
    checkpoints = _config['checkpoints']

    # Load and match configs for weights_dir2 
    exceptions = ['graph_attrs', 'num_graph_attrs', 'target']
    config2 = load_matching_configs(config={}, weights_dir=weights_dir2, exceptions=exceptions)

    # Make sure that only one of the VGAEs has graph_attrs
    graph_attrs = _config['dataset']['graph_attrs']
    if len(graph_attrs) > 0:
        config2['dataset']['graph_attrs'] = []

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Load data for VGAE1 and VGAE2
    data = load_data()
    data2 = load_dataset_from_configs(config2['dataset'])
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load trained VGAEs
    vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device)
    vgaes2 = load_trained_vgaes2(config2['vgae_model'], weights_dir2, weight_filenames['vgae'], device)
    
    # Get dataloader using the same testfold indices as the trained models
    train_loaders, val_loaders, test_loaders, _ = get_dataloaders_from_test_indices(data, test_indices, seed=seed)
    train_loaders2, val_loaders2, test_loaders2, _ = get_dataloaders_from_test_indices(data2, test_indices, seed=seed)

    # Train MLP and readout -------------------------------------------------------
    start_time = time()
    num_folds = len(train_loaders)

    best_outputs = init_outputs_dict(data)
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}
    checkpoint_mlp_states = {}

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        checkpoint_mlp_states[k] = []
        
        # Get pretrained VGAEs and freeze them
        vgae = vgaes[k].to(device)
        vgae.eval()
        vgae = freeze_model(vgae)
        vgae2 = vgaes2[k].to(device)
        vgae2.eval()
        vgae2 = freeze_model(vgae2)

        # Build MLP and optimizer
        latent_dim = vgae.readout_dim + vgae2.readout_dim
        mlp = build_mlp(latent_dim=latent_dim).to(device)
        optimizer = get_optimizer(mlp)

        # Best validation loss 
        best_val_loss = float('inf')
        best_mlp_state = None

        for epoch in range(num_epochs):
            # Train readout and MLP
            _ = train_mlp_with_2vgaes(mlp, train_loaders[k], train_loaders2[k], 
                                      vgae, vgae2, optimizer, device)
            
            # Compute training losses
            mlp_train_loss[k].append(test_mlp_with_2vgaes(mlp, train_loaders[k], 
                                                          train_loaders2[k], 
                                                          vgae, vgae2, device))
            
            # Test VGAE and MLP
            mlp_test_loss[k].append(test_mlp_with_2vgaes(mlp, test_loaders[k], 
                                                          test_loaders2[k], 
                                                          vgae, vgae2, device))

            # Log training and test losses
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss[k][-1])
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss[k][-1])
            
            # Validate models, if applicable
            if len(val_loaders) > 0:
                mlp_val_loss[k].append(test_mlp_with_2vgaes(mlp, val_loaders[k], val_loaders2[k], 
                                                            vgae, vgae2, device))
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss[k][-1])
                
                # Save the best model if validation loss is at its minimum
                if mlp_val_loss[k][-1] < best_val_loss:
                    best_val_loss = mlp_val_loss[k][-1]
                    best_mlp_state = copy.deepcopy(mlp.state_dict()) 

            # Save checkpoint if applicable
            if epoch in checkpoints:
                checkpoint_mlp_state = best_mlp_state or copy.deepcopy(mlp.state_dict())
                if save_weights:
                    checkpoint_dir = os.path.join(output_dir, f'epoch_{epoch}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(checkpoint_mlp_state, os.path.join(checkpoint_dir, f'k{k}_mlp_weights.pth'))
                # Save states for later evaluation
                checkpoint_mlp_states[k].append(checkpoint_mlp_state)

        # Load best model of this fold 
        if best_mlp_state is not None:
            mlp.load_state_dict(best_mlp_state)

        # Save model weights
        if save_weights:
            torch.save(mlp.state_dict(), os.path.join(output_dir, f'k{k}_mlp_weights.pth'))

        # Save the test predictions (on VGAE latent means) of the best model
        outputs = get_mlp_outputs_nograd_2vgaes(mlp, test_loaders[k], test_loaders2[k], 
                                                vgae, vgae2, device, num_z_samples=0)
        update_best_outputs(best_outputs, outputs)

    # Print training time
    end_time = time()
    logger.info(f"MLP training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save ouputs as csv file
    best_outputs = pd.DataFrame(best_outputs)
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    best_outputs.to_csv(data_file, index=False)

    # Save test fold assignments
    test_indices_list = [np.where(test_indices == fold)[0] for fold in range(num_folds)]
    if save_weights:
        test_indices_file = save_test_indices(test_indices_list, output_dir)

    # Save final prediction results
    r, p, mae, mae_std = evaluate_regression(best_outputs)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Final results: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

    # Plot results ----------------------------------------------------------------
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=os.path.join(output_dir, 'mlp_loss_curves.png'))
    image_files += [os.path.join(output_dir, 'mlp_loss_curves.png')]

    # True vs predicted scatter
    title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    true_vs_pred_scatter(best_outputs, title=title, save_path=os.path.join(output_dir, 'true_vs_predicted.png'))
    image_files.append(os.path.join(output_dir, 'true_vs_predicted.png'))

    # Evaluate all checkpoints ---------------------------------------------------
    for i, epoch in enumerate(checkpoints):
        checkpoint_outputs = init_outputs_dict(data)
        for k in range(num_folds):
            # Load checkpoint models
            vgae = vgaes[k].to(device)
            vgae.eval()
            vgae2 = vgaes2[k].to(device)
            vgae2.eval()
            
            latent_dim = vgae.readout_dim + vgae2.readout_dim
            mlp = build_mlp(latent_dim=latent_dim).to(device)
            mlp.load_state_dict(checkpoint_mlp_states[k][i])
            mlp.eval()

            # Get MLP outputs for this fold
            outputs = get_mlp_outputs_nograd_2vgaes(mlp, test_loaders[k], test_loaders2[k], 
                                                    vgae, vgae2, device, num_z_samples=0)
            # Update checkpoint outputs
            update_best_outputs(checkpoint_outputs, outputs)

        # Plot predictions
        checkpoint_outputs = pd.DataFrame(checkpoint_outputs)
        r, p, mae, mae_std = evaluate_regression(checkpoint_outputs)
        title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
        true_vs_pred_scatter(checkpoint_outputs, title=title, 
                             save_path=os.path.join(output_dir, f'epoch_{epoch}_true_vs_predicted.png'))
        image_files += [os.path.join(output_dir, f'epoch_{epoch}_true_vs_predicted.png')]

        # Log metrics
        results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
        for key, val in results.items():
            ex.log_scalar(f'checkpoint_prediction/{key}', val)

    # Close all plots if not verbose
    if not verbose:
        plt.close()

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)
