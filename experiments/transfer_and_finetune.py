"""
Loads a pre-trained model and evaluates it on a new dataset. 
Optionally, subsequent finetuning is performed on the new dataset.
Note: Requires consistent graph_attrs and context_attrs between 
the pretrained and new datasets.

Author: Hanna Tolle
Date: 2025-01-12 
License: BSD-3-Clause
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
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices, check_weights_exist
from utils.plotting import plot_vgae_reconstructions, plot_loss_curves, true_vs_pred_scatter
from utils.configs import load_ingredient_configs, match_ingredient_configs
from preprocessing.metrics import get_rsn_mapping
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('transfer_and_finetune', ingredients=[data_ingredient, 
                                                      vgae_ingredient, 
                                                      mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'transfer_and_finetune'
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
    weights_dir = os.path.join('outputs', 'weights')

    # If multiple weights are given, evaluate all but finetune only the first
    weight_filenames = {'vgae': ['k0_vgae_weights.pth'],
                        'mlp': ['k0_mlp_weights.pth']} 

    # Training configurations (only used if num_epochs > 0)
    num_epochs = 50       # Number of epochs for fine-tuning.
    vgae_lr = 0.0         # VGAE learning rate.
    mlp_lr = 0.001        # MLP learning rate.
    num_z_samples = 3     # 0 for training MLP on the means of VGAE latent variables.
    alpha = 0             # Loss = alpha*vgae_loss + (1-alpha)*mlp_loss; if alpha=0, VGAE is frozen.
    rho = 0.5             # mlp_loss = rho*corr_loss + (1-rho)*mse_loss
    leak_rate = 0.9       # Running average leak rate

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
    exceptions = ['num_nodes', 'atlas', 
                  'num_folds', 'batch_size', 'val_split', 
                  'study', 'session', 'target',
                  'dropout', 'reg_strength', 'layernorm', 'mse_reduction']
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Other config checks
    num_pretrained_models = previous_config['dataset']['num_folds']
    default_weight_filenames = {'vgae': [f'k{i}_vgae_weights.pth' for i in range(num_pretrained_models)], 
                                'mlp': [f'k{i}_mlp_weights.pth' for i in range(num_pretrained_models)]}
    weight_filenames = config.get('weight_filenames', default_weight_filenames)
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames

    # Don't support standardisation of x in pretrained models
    if 'standardise_x' in previous_config['dataset']:
        assert not config_updates['dataset']['standardise_x'], \
            "Standardisation of x in pretrained models is not supported."
        
    # If alpha=0, vgae_lr is set to 0 (issue warning if overridden)
    alpha = config_updates.get('alpha', 0.2)
    vgae_lr = config_updates.get('vgae_lr', 0.0001)
    if alpha == 0:
        if vgae_lr != 0:
            print(f"Warning: alpha=0 overrides vgae_lr={vgae_lr} to 0.")
        config_updates['vgae_lr'] = 0
    elif alpha > 0:
        # Make sure that vgae_lr is not 0
        assert vgae_lr != 0, "VGAE learning rate must be non-zero when alpha > 0."
    return config_updates

# Helper class ---------------------------------------------------------------
class LossNormaliser:
    def __init__(self, leak_rate):
        self.leak_rate = leak_rate
        self.training = True
        self.mse_avg = None
        self.corr_avg = None
        self.vgae_avg = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def update(self, vgae_loss, mse_loss, corr_loss):
        # Initialize running averages
        if self.vgae_avg is None:
            self.vgae_avg = vgae_loss.detach()
            self.mse_avg = mse_loss.detach()
            self.corr_avg = corr_loss.detach()

        # Update running averages
        else:
            self.vgae_avg = self.leak_rate * self.vgae_avg + (1 - self.leak_rate) * vgae_loss.detach()
            self.mse_avg = self.leak_rate * self.mse_avg + (1 - self.leak_rate) * mse_loss.detach()
            self.corr_avg = self.leak_rate * self.corr_avg + (1 - self.leak_rate) * corr_loss.detach()
            
    def normalise(self, vgae_loss, mse_loss, corr_loss):
        # Normalize losses by their running averages
        normalized_vgae = vgae_loss / (self.vgae_avg + 1e-8)
        normalized_mse = mse_loss / (self.mse_avg + 1e-8)
        normalized_corr = corr_loss / (self.corr_avg + 1e-8)
        results = {'vgae': normalized_vgae, 
                   'mlp_mse': normalized_mse, 
                   'mlp_corr': normalized_corr}
        return results
    
    def get_losses(self, vgae_loss, mse_loss, corr_loss):
        if self.training:
            self.update(vgae_loss, mse_loss, corr_loss)
        return self.normalise(vgae_loss, mse_loss, corr_loss)

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(vgae, mlp, vgae_lr, mlp_lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    if vgae_lr == 0:
        optimizer = torch.optim.Adam(mlp.parameters(), lr=mlp_lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': vgae.parameters(), 'lr': vgae_lr},
            {'params': mlp.parameters(), 'lr': mlp_lr}])
    return optimizer

@ex.capture
def get_loss_normaliser(leak_rate):
    return LossNormaliser(leak_rate)

@ex.capture
def weight_losses(norm_losses, alpha, rho):
    vgae_loss = norm_losses['vgae']
    mse_loss = norm_losses['mlp_mse']
    corr_loss = norm_losses['mlp_corr']
    mlp_loss = rho*corr_loss + (1-rho)*mse_loss
    total_loss = alpha*vgae_loss + (1-alpha)*mlp_loss
    return total_loss

@ex.capture
def compute_correlation_loss(ypred, ytrue, num_z_samples):
    '''
    Computes the average correlation loss between ypred and ytrue across multiple predictions per subject.
    
    Parameters:
        ypred: tensor of shape (batch_size * num_z_samples,)
        ytrue: tensor of shape (batch_size * num_z_samples,)
        num_z_samples: number of predictions per subject
    
    Returns:
        average correlation loss across z-samples
    '''
    # No need to handle multiple samples if num_z_samples=0 or 1
    if num_z_samples <= 1:
        x = ypred - ypred.mean()
        y = ytrue - ytrue.mean()
        x = x.view(-1)
        y = y.view(-1)
        r = (x * y).sum() / (torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum()) + 1e-8)
        return -r

    # Compute correlation for each z-sample
    num_subjects = len(ypred) // num_z_samples
    correlations = torch.zeros(num_z_samples, device=ypred.device)
    
    for z in range(num_z_samples):
        # Get predictions and true values for this z-sample
        start_idx = z * num_subjects
        end_idx = (z + 1) * num_subjects
        z_pred = ypred[start_idx:end_idx]
        z_true = ytrue[start_idx:end_idx]
        
        # Compute correlation for this z-sample
        x = z_pred - z_pred.mean()
        y = z_true - z_true.mean()
        x = x.view(-1)
        y = y.view(-1)
        r = (x * y).sum() / (torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum()) + 1e-8)
        correlations[z] = r
    
    # Correlation loss = negative correlation
    return -correlations.sum()

@ex.capture
def train_vgae_mlp(vgae, mlp, loader, optimizer, device, loss_normaliser, num_z_samples):
    '''
    Performs joint training of VGAE and MLP models for one epoch.
    
    Parameters:
    -----------
    vgae (torch.nn.Module): VGAE model
    mlp (torch.nn.Module): MLP model
    loader (torch.utils.data.DataLoader): Training data loader
    optimizer (torch.optim.Optimizer): Optimizer
    device (torch.device): Device to use for training.
    loss_normaliser (LossNormaliser): Loss normaliser
    num_z_samples (int): Number of samples to draw from the latent space for training.
    '''
    vgae.train()
    mlp.train()
    loss_normaliser.train()

    train_loss = 0.
    for batch in loader: 
        # Fetch data and labels
        batch = batch.to(device)             
        ytrue = get_labels(batch, num_z_samples)

        # Forward pass
        optimizer.zero_grad()
        x, vgae_loss = get_x_with_vgae_loss(batch, device, vgae, num_z_samples)
        ypred = mlp(x)

        # Compute losses
        mlp_mse_loss = mlp.loss(ypred, ytrue) 
        mlp_corr_loss = compute_correlation_loss(ypred, ytrue)
        norm_losses = loss_normaliser.get_losses(vgae_loss, mlp_mse_loss, mlp_corr_loss)
        loss = weight_losses(norm_losses)

        # Backpropagation
        loss.backward()                      
        optimizer.step()                     
        train_loss += loss.item()

    return train_loss/len(loader)

@ex.capture
def test_vgae_mlp(vgae, mlp, loader, device, num_z_samples):
    '''
    Evaluates the MLP.
    Parameters:
    ----------
    mlp (torch.nn.Module): trained RegressionMLP model.
    loader (torch.utils.data.DataLoader): test data loader.
    vgae (torch.nn.Module): trained VGAE model.
    device (torch.device): Device to use for training.
    num_z_samples (int): Number of samples to draw from the latent space for training.

    Returns:
    -------
    mlp_test_loss (float): average MLP loss across test dataset.
    vgae_test_loss (float): average VGAE loss across test dataset.
    '''
    mlp.eval()
    vgae.eval()

    mlp_mse_test_loss = 0.
    mlp_corr_test_loss = 0.
    vgae_test_loss = 0.
    with torch.no_grad():
        for batch in loader:
            # Get inputs and labels
            batch = batch.to(device)
            ytrue = get_labels(batch, num_z_samples)

            # Forward pass
            x, vgae_loss = get_x_with_vgae_loss(batch, device, vgae, num_z_samples)
            ypred = mlp(x)                                   

            # Compute losses
            mlp_mse_loss = mlp.loss(ypred, ytrue) 
            mlp_corr_loss = compute_correlation_loss(ypred, ytrue)
            
            # Update test losses
            mlp_mse_test_loss += mlp_mse_loss.item()
            mlp_corr_test_loss += mlp_corr_loss.item()
            vgae_test_loss += vgae_loss.item()

    # Average losses across batch
    vgae_test_loss = vgae_test_loss/len(loader)
    mlp_mse_test_loss = mlp_mse_test_loss/len(loader)
    mlp_corr_test_loss = mlp_corr_test_loss/len(loader)

    return vgae_test_loss, mlp_mse_test_loss, mlp_corr_test_loss

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    alpha = _config['alpha']
    num_epochs = _config['num_epochs']
    num_folds = _config['dataset']['num_folds']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')
    train_loaders, val_loaders, test_loaders, test_indices_list, mean_std = \
        get_kfold_dataloaders(data, seed=seed)
    
    # Load pretrained models
    pretrained_vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device=device)
    pretrained_mlps = load_trained_mlps(weights_dir, weight_filenames['mlp'], device=device,
                                        latent_dims=[vgae.readout_dim for vgae in pretrained_vgaes])
    
    # Make an output directory for each pretrained model
    output_dirs = []
    num_pretrained_models = len(pretrained_vgaes)
    for i in range(num_pretrained_models):
        pretrained_model_dir = os.path.join(output_dir, f'pretrained_model{i}')
        os.makedirs(pretrained_model_dir, exist_ok=True)
        output_dirs.append(pretrained_model_dir)

    # Evaluate before fine-tuning -------------------------------------------------
    # Evaluate each set of pretrained models on the whole new dataset
    eval_loader = DataLoader(data, batch_size=len(data), shuffle=False)

    # Evaluate MLP predictions --------------------------------------------------
    all_results = []
    for i in range(num_pretrained_models):
        initial_outputs = init_outputs_dict(data)
        outputs = get_mlp_outputs_nograd(pretrained_mlps[i], eval_loader, device, 
                                         get_x=get_x_with_vgae, 
                                         vgae=pretrained_vgaes[i], num_z_samples=0)
        update_best_outputs(initial_outputs, outputs)
    
        # Record metrics for each pretrained model
        initial_outputs = pd.DataFrame(initial_outputs)
        r, p, mae, mae_std = evaluate_regression(initial_outputs)
        results = {'pretrained_model': i, 
                   'seed': seed, 
                   'r': r, 
                   'p': p, 
                   'mae': mae, 
                   'mae_std': mae_std}
        all_results.append(results)
        for k, v in results.items():
            if k != 'pretrained_model':
                ex.log_scalar(f'initial_prediction/{k}', v)
        logger.info(f"Initial results for pretrained model {i}: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

        # True vs predicted scatter plot
        title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
        save_path = os.path.join(output_dirs[i], f'initial_true_vs_predicted_model{i}.png')
        true_vs_pred_scatter(initial_outputs, title=title, save_path=save_path)
        image_files.append(save_path)

        # Close all plots
        plt.close('all')

        # Save & log initial prediction results
        file = os.path.join(output_dirs[i], 'initial_prediction_results.csv')
        initial_outputs.to_csv(file, index=False)

    # Save initial metrics of all pretrained models
    all_results = pd.DataFrame(all_results)
    all_results.to_csv(os.path.join(output_dir, 'initial_metrics_summary.csv'), index=False)

    # Initial predictions mean voting ------------------------------------------
    # Load the initial predictions for each pretrained model
    initial_outputs = np.zeros((len(data), num_pretrained_models))
    for i in range(num_pretrained_models):
        file = os.path.join(output_dirs[i], 'initial_prediction_results.csv')
        df = pd.read_csv(file)
        initial_outputs[:, i] = df['prediction'].values
    
    # Compute the mean prediction across all pretrained models
    df['prediction'] = np.mean(initial_outputs, axis=1)
    df['prediction_std'] = np.std(initial_outputs, axis=1)
    df.to_csv(os.path.join(output_dir, 'initial_prediction_results_mean_vote.csv'), index=False)

    # Evaluate the mean predictions
    r, p, mae, mae_std = evaluate_regression(df)
    results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'initial_metrics_mean_vote.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'initial_prediction_mean_vote/{k}', v)
    logger.info(f"Initial results for mean voting: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

    # True vs predicted scatter
    title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    save_path = os.path.join(output_dir, f'initial_true_vs_predicted_mean_vote.png')
    true_vs_pred_scatter(df, title=title, save_path=save_path)
    image_files.append(save_path)

    # Close all plots
    plt.close('all')

    # Evaluate VGAE reconstructions ----------------------------------------------
    # Some details needed for plotting
    rsn_mapping, rsn_labels = get_rsn_mapping(data.atlas)
    vrange = None
    if _config['dataset']['edge_attrs'][0] == 'functional_connectivity':
        vrange = (-0.7, 0.7)

    for i in range(num_pretrained_models):
        vgae = pretrained_vgaes[i].eval()
        adj_orig_rcn, x_orig_rcn, fold_assignments = get_test_reconstructions(
            [vgae], data, [np.arange(len(data))], mean_std=None)

        # Evaluate reconstructions
        adj_orig_rcn['metrics'] = evaluate_fc_reconstructions(adj_orig_rcn)
        ex.log_scalar('initial_reconstruction/fc_corr', np.mean(adj_orig_rcn['metrics']['corr']))
        ex.log_scalar('initial_reconstruction/fc_mae', np.mean(adj_orig_rcn['metrics']['mae']))

        if x_orig_rcn is not None and 'feature_names' in x_orig_rcn:
            x_orig_rcn['metrics'] = evaluate_x_reconstructions(x_orig_rcn)
            for feature in x_orig_rcn['feature_names']:
                ex.log_scalar(f'initial_reconstruction/x_{feature}_corr', x_orig_rcn['metrics']['corr'][feature].mean())
                ex.log_scalar(f'initial_reconstruction/x_{feature}_mae', x_orig_rcn['metrics']['mae'][feature].mean())
        
        # Some other details needed for plotting
        save_path = os.path.join(output_dirs[i], f'initial_vgae_model{i}.png')
        image_files += plot_vgae_reconstructions(adj_orig_rcn, 
                                                 x_orig_rcn, 
                                                 fold_assignments, 
                                                 conditions=get_conditions(data),
                                                 rsn_mapping=rsn_mapping,
                                                 rsn_labels=rsn_labels,
                                                 atlas=data.atlas,
                                                 vrange=vrange, 
                                                 save_path=save_path,
                                                 only_boxplots=True)
        # Close all plots
        plt.close('all')

    # Fine-tune VGAE and MLP -----------------------------------------------------
    if num_epochs > 0:
        start_time = time()

        # Use each pretrained model as the initial model for each fold
        for pretrained_idx in tqdm(range(num_pretrained_models), desc='Pretrained models', disable=not verbose):

            # Get a copy of the pretrained model states
            pretrained_vgae_state = copy.deepcopy(pretrained_vgaes[pretrained_idx].state_dict())
            pretrained_mlp_state = copy.deepcopy(pretrained_mlps[pretrained_idx].state_dict())

            # Initialise outputs
            best_outputs = init_outputs_dict(data)
            vgae_train_loss, vgae_test_loss, vgae_val_loss = {}, {}, {}
            mlp_mse_train_loss, mlp_corr_train_loss = {}, {}
            mlp_mse_test_loss, mlp_corr_test_loss = {}, {}
            mlp_mse_val_loss, mlp_corr_val_loss = {}, {}
            finetuned_vgae_states = []

            for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

                # Initialise losses
                mlp_mse_train_loss[k], mlp_corr_train_loss[k] = [], []
                mlp_mse_test_loss[k], mlp_corr_test_loss[k] = [], []
                mlp_mse_val_loss[k], mlp_corr_val_loss[k] = [], []
                vgae_train_loss[k], vgae_test_loss[k], vgae_val_loss[k] = [], [], []
                
                # Build a new VGAE and MLP, then load the pretrained weights
                vgae = build_vgae().to(device)
                vgae.load_state_dict(pretrained_vgae_state)

                if alpha == 0: # freeze the VGAE, if not trained
                    vgae = freeze_model(vgae)

                mlp = build_mlp(latent_dim = vgae.readout_dim).to(device)
                mlp.load_state_dict(pretrained_mlp_state)

                # If alpha=0, only MLP parameters are passed to the optimizer
                optimizer = get_optimizer(vgae, mlp)
                loss_normaliser = get_loss_normaliser()

                # Best validation loss and early stopping counter
                best_val_loss = float('inf')
                best_vgae_state = None
                best_mlp_state = None

                for epoch in range(num_epochs):
                    # Train VGAE and MLP
                    _ = train_vgae_mlp(vgae, mlp, train_loaders[k], optimizer, device, loss_normaliser)
                    
                    # Compute training losses
                    vgae_train_loss_epoch, mlp_mse_train_loss_epoch, mlp_corr_train_loss_epoch = \
                        test_vgae_mlp(vgae, mlp, train_loaders[k], device)
                    vgae_train_loss[k].append(vgae_train_loss_epoch)
                    mlp_mse_train_loss[k].append(mlp_mse_train_loss_epoch)
                    mlp_corr_train_loss[k].append(mlp_corr_train_loss_epoch)
                    
                    # Test VGAE and MLP
                    vgae_test_loss_epoch, mlp_mse_test_loss_epoch, mlp_corr_test_loss_epoch = \
                        test_vgae_mlp(vgae, mlp, test_loaders[k], device)
                    vgae_test_loss[k].append(vgae_test_loss_epoch)
                    mlp_mse_test_loss[k].append(mlp_mse_test_loss_epoch)
                    mlp_corr_test_loss[k].append(mlp_corr_test_loss_epoch)

                    # Log training and test losses
                    ex.log_scalar(f'training/pretrained_model{pretrained_idx}/fold{k}/epoch/vgae_loss', vgae_train_loss_epoch)
                    ex.log_scalar(f'training/pretrained_model{pretrained_idx}/fold{k}/epoch/mlp_loss', mlp_mse_train_loss_epoch)
                    ex.log_scalar(f'training/pretrained_model{pretrained_idx}/fold{k}/epoch/mlp_corr', mlp_corr_train_loss_epoch)
                    ex.log_scalar(f'test/pretrained_model{pretrained_idx}/fold{k}/epoch/vgae_loss', vgae_test_loss_epoch)
                    ex.log_scalar(f'test/pretrained_model{pretrained_idx}/fold{k}/epoch/mlp_loss', mlp_mse_test_loss_epoch)
                    ex.log_scalar(f'test/pretrained_model{pretrained_idx}/fold{k}/epoch/mlp_corr', mlp_corr_test_loss_epoch)

                    # Validate models, if applicable
                    if len(val_loaders) > 0:
                        vgae_val_loss_epoch, mlp_mse_val_loss_epoch, mlp_corr_val_loss_epoch = \
                            test_vgae_mlp(vgae, mlp, val_loaders[k], device)
                        vgae_val_loss[k].append(vgae_val_loss_epoch)
                        mlp_mse_val_loss[k].append(mlp_mse_val_loss_epoch)
                        mlp_corr_val_loss[k].append(mlp_corr_val_loss_epoch)

                        # Log validation losses
                        ex.log_scalar(f'validation/pretrained_model{pretrained_idx}/fold{k}/epoch/vgae_loss', vgae_val_loss_epoch)
                        ex.log_scalar(f'validation/pretrained_model{pretrained_idx}/fold{k}/epoch/mlp_loss', mlp_mse_val_loss_epoch)
                        ex.log_scalar(f'validation/pretrained_model{pretrained_idx}/fold{k}/epoch/mlp_corr', mlp_corr_val_loss_epoch)
                        
                        # Save the best model if validation loss is at its minimum
                        loss_terms = {'vgae': vgae_val_loss_epoch, 
                                      'mlp_mse': mlp_mse_val_loss_epoch, 
                                      'mlp_corr': mlp_corr_val_loss_epoch}
                        total_val_loss = weight_losses(loss_terms)
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
                    torch.save(mlp.state_dict(), os.path.join(output_dirs[pretrained_idx], f'k{k}_mlp_weights.pth'))
                    torch.save(vgae.state_dict(), os.path.join(output_dirs[pretrained_idx], f'k{k}_vgae_weights.pth'))
                finetuned_vgae_states.append(copy.deepcopy(vgae.state_dict())) # for final evaluation

                # Save the test predictions (on VGAE latent means) of the best model
                outputs = get_mlp_outputs_nograd(mlp, test_loaders[k], device, 
                                                get_x=get_x_with_vgae, 
                                                vgae=vgae, num_z_samples=0)
                update_best_outputs(best_outputs, outputs)

            # Print training time
            end_time = time()
            logger.info(f"Joint training completed after {(end_time-start_time)/60:.2f} minutes.")

            # Save ouputs as csv file
            best_outputs = pd.DataFrame(best_outputs)
            data_file = os.path.join(output_dirs[pretrained_idx], 'prediction_results.csv')
            best_outputs.to_csv(data_file, index=False)

            # Save test fold assignments
            if save_weights:
                test_indices_file = save_test_indices(test_indices_list, output_dirs[pretrained_idx])

            # Save final prediction results
            r, p, mae, mae_std = evaluate_regression(best_outputs)
            results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
            pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dirs[pretrained_idx], 'final_metrics.csv'), index=False)

            # Log final metrics
            for k, v in results.items():
                ex.log_scalar(f'final_prediction/{k}', v)
            logger.info(f"Final results for pretrained model {pretrained_idx}: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

            # Plot results ----------------------------------------------------------------
            # Only plot the reconstructions if the VGAE was fine-tuned
            if alpha > 0:
                # Load the finetuned VGAEs
                vgaes = load_vgaes_from_states_list(finetuned_vgae_states, device)

                # Get reconstructions
                adj_orig_rcn, x_orig_rcn, fold_assignments = get_test_reconstructions(
                    vgaes, data, test_indices_list, mean_std=mean_std)
                
                # Evaluate reconstructions
                adj_orig_rcn['metrics'] = evaluate_fc_reconstructions(adj_orig_rcn)
                ex.log_scalar('final_reconstruction/fc_corr', np.mean(adj_orig_rcn['metrics']['corr']))
                ex.log_scalar('final_reconstruction/fc_mae', np.mean(adj_orig_rcn['metrics']['mae']))

                if x_orig_rcn is not None and 'feature_names' in x_orig_rcn:
                    x_orig_rcn['metrics'] = evaluate_x_reconstructions(x_orig_rcn)
                    for feature in x_orig_rcn['feature_names']:
                        ex.log_scalar(f'final_reconstruction/x_{feature}_corr', x_orig_rcn['metrics']['corr'][feature].mean())
                        ex.log_scalar(f'final_reconstruction/x_{feature}_mae', x_orig_rcn['metrics']['mae'][feature].mean())    
                
                # Some other details needed for plotting
                save_path = os.path.join(output_dirs[pretrained_idx], f'vgae_model{pretrained_idx}.png')
                vrange = None
                if _config['dataset']['edge_attrs'][0] == 'functional_connectivity':
                    vrange = (-0.7, 0.7)
                image_files += plot_vgae_reconstructions(adj_orig_rcn, 
                                                        x_orig_rcn, 
                                                        fold_assignments, 
                                                        conditions=get_conditions(data),
                                                        rsn_mapping=rsn_mapping,
                                                        rsn_labels=rsn_labels,
                                                        atlas=data.atlas,
                                                        vrange=vrange, 
                                                        save_path=save_path,
                                                        only_boxplots=True)
            # Loss curves
            save_path_vgae = os.path.join(output_dirs[pretrained_idx], f'vgae_loss_curves_model{pretrained_idx}.png')
            plot_loss_curves(vgae_train_loss, vgae_test_loss, vgae_val_loss, save_path=save_path_vgae)
            save_path_mlp = os.path.join(output_dirs[pretrained_idx], f'mlp_loss_curves_model{pretrained_idx}.png')
            plot_loss_curves(mlp_mse_train_loss, mlp_mse_test_loss, mlp_mse_val_loss, save_path=save_path_mlp)
            save_path_mlp_corr = os.path.join(output_dirs[pretrained_idx], f'mlp_corr_loss_curves_model{pretrained_idx}.png')
            plot_loss_curves(mlp_corr_train_loss, mlp_corr_test_loss, mlp_corr_val_loss, save_path=save_path_mlp_corr)
            image_files += [save_path_vgae, save_path_mlp, save_path_mlp_corr]

            # True vs predicted scatter
            title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
            save_path = os.path.join(output_dirs[pretrained_idx], f'true_vs_predicted_model{pretrained_idx}.png')
            true_vs_pred_scatter(best_outputs, title=title, save_path=save_path)
            image_files.append(save_path)

            # Close all plots
            plt.close('all')

        # Mean voting --------------------------------------------------------
        # Load the final predictions for each pretrained model
        final_outputs = np.zeros((len(data), num_pretrained_models))
        for pretrained_idx in range(num_pretrained_models):
            file = os.path.join(output_dirs[pretrained_idx], 'prediction_results.csv')
            df = pd.read_csv(file)
            final_outputs[:, pretrained_idx] = df['prediction'].values
        
        # Compute the mean prediction across all pretrained models
        df['prediction'] = np.mean(final_outputs, axis=1)
        df['prediction_std'] = np.std(final_outputs, axis=1)
        df.to_csv(os.path.join(output_dir, 'prediction_results_mean_vote.csv'), index=False)

        # Evaluate the mean predictions
        r, p, mae, mae_std = evaluate_regression(df)
        results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
        pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics_mean_vote.csv'), index=False)

        # Log final metrics
        for k, v in results.items():
            ex.log_scalar(f'final_prediction_mean_vote/{k}', v)
        logger.info(f"Final results for mean voting: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")

        # True vs predicted scatter
        title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
        save_path = os.path.join(output_dir, f'true_vs_predicted_mean_vote.png')
        true_vs_pred_scatter(df, title=title, save_path=save_path)
        image_files.append(save_path)

        # Close all plots
        plt.close('all')

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)
