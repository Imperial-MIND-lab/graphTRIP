'''
These models perform the final regression task (i.e., predicting treatment outcome).

Author: Hanna M. Tolle
License: BSD-3-Clause
'''

import sys
sys.path.append('../../')

from sacred import Ingredient
from .data_ingredient import data_ingredient, get_context

import torch
import pandas as pd
import os
from scipy.stats import pearsonr

from models.utils import init_model


# Create the ingredient --------------------------------------------------------
mlp_ingredient = Ingredient('mlp_model', ingredients=[data_ingredient])

# Define configurations --------------------------------------------------------
@mlp_ingredient.config
def mlp_cfg(dataset):
    # Model configurations
    model_type = 'RegressionMLP'
    extra_dim = len(dataset['graph_attrs'])  # Clinical or other extra features.
    params = {'hidden_dim': None,            # if None, hidden_dim = max(latent_dim, extra_dim)
              'output_dim': 1,
              'num_layers': 4,
              'dropout': 0.25,
              'layernorm': False,
              'reg_strength': 0.01,
              'mse_reduction': 'sum'}

# Capture functions ------------------------------------------------------------
@mlp_ingredient.capture
def build_mlp(model_type, params, extra_dim, latent_dim=0):
    '''Initialises an MLP.'''
    update_params = params.copy()
    update_params['input_dim'] = extra_dim + latent_dim
    update_params['hidden_dim'] = update_params['hidden_dim'] or max(latent_dim, extra_dim)
    return init_model(model_type, update_params)

@mlp_ingredient.capture
def load_trained_mlps(weights_dir, weight_filenames, device=None, latent_dims=None):
    '''Loads a pre-trained MLP model.'''
    if device is None:
        device = torch.device('cpu')
    if latent_dims is None:
        latent_dims = [0] * len(weight_filenames)
    mlps = []
    for weight_file, latent_dim in zip(weight_filenames, latent_dims):
        mlp = build_mlp(latent_dim=latent_dim).to(device)
        mlp.load_state_dict(torch.load(os.path.join(weights_dir, weight_file)))
        mlps.append(mlp)
    return mlps

@mlp_ingredient.capture
def get_labels(batch, num_z_samples, device=None):
    '''Returns the labels for the MLP.'''
    # Move labels to device
    if device is None:
        device = torch.device('cpu')
    ytrue = batch.y.view(-1, 1).to(device)

    # No augmentation, if predicting on means (num_z_samples=0), or only one sample.
    if num_z_samples <= 1:
        return ytrue.float()
    
    # Augment labels for multiple samples.
    else:
        n_samples = len(batch)
        ytrue_aug = torch.empty((n_samples * num_z_samples, 1), dtype=torch.float, device=device)
        for i in range(num_z_samples):
            ytrue_aug[i * n_samples: (i + 1) * n_samples, :] = ytrue
        return ytrue_aug
    
def get_num_z_sample_indices(batch_size, num_z_samples, device=None):
    '''
    Returns indices for grouping multiple predictions per subject.
    
    Parameters:
        batch_size: number of subjects in the batch
        num_z_samples: number of predictions per subject
        device: torch device (optional)
    
    Returns:
        tensor of indices where each index corresponds to the subject number
        shape: (batch_size * num_z_samples,)
    
    Example for batch_size=2, num_z_samples=3:
        Returns tensor([0, 1, 2, 0, 1, 2])
    '''
    if device is None:
        device = torch.device('cpu')
        
    # Create repeated indices for each subject
    indices = torch.arange(batch_size, device=device)
    indices = indices.repeat(num_z_samples)
    
    return indices

@mlp_ingredient.capture
def train_mlp(mlp, loader, optimizer, device, get_x, *args, **kwargs):
    '''
    Performs training of the RegressionMLP.
    
    Parameters:
    -----------
    mlp (torch.nn.Module): RegressionMLP model
    loader (torch.utils.data.DataLoader): Training data loader
    optimizer (torch.optim.Optimizer): Optimizer
    device (torch.device): Device to use for training.
    get_x (function): function to get the input data for the MLP.
                      This should be implemented by the experiment.
                      get_x always takes batch and device as arguments.
    '''
    # Check if mlp is trained with vgae and num_z_samples is not 0
    num_z_samples = kwargs.get('num_z_samples', 0)

    mlp.train()
    train_loss = 0.
    for batch in loader: 
        # Fetch data and labels
        batch = batch.to(device)             
        ytrue = get_labels(batch, num_z_samples, device)

        # Forward pass
        optimizer.zero_grad()
        x = get_x(batch, device, *args, **kwargs)
        ypred = mlp(x)                     
        loss = mlp.loss(ypred, ytrue)

        # Backpropagation
        loss.backward()                      
        optimizer.step()                     
        train_loss += loss.item()

    return train_loss/len(loader)

@mlp_ingredient.capture
def test_mlp(mlp, loader, device, get_x, *args, **kwargs):
    '''
    Evaluates the RegressionMLP. 
    Parameters:
    ----------
    mlp (torch.nn.Module): trained RegressionMLP model.
    loader (torch.utils.data.DataLoader): test data loader.
    device (torch.device): Device to use for testing.
    get_x (function): function to get the input data for the MLP.
                      This should be implemented by the experiment.
                      get_x always takes batch and device as arguments.
    Returns:
    -------
    test_loss (float): average loss across test dataset.
    '''
    # Check if mlp is trained with vgae and num_z_samples is not 0
    num_z_samples = kwargs.get('num_z_samples', 0)

    mlp.eval()
    test_loss = 0.
    with torch.no_grad():
        for batch in loader:
            # Get inputs and labels
            batch = batch.to(device)
            y_true = get_labels(batch, num_z_samples, device)

            # Get MLP inputs
            x = get_x(batch, device, *args, **kwargs)

            # Get MLP loss
            y_pred = mlp(x)                                   
            loss = mlp.loss(y_pred, y_true)                   
            test_loss += loss.item()

    return test_loss/len(loader)

def get_x_with_vgae(batch, device, vgae, num_z_samples=0):
    '''
    Returns the MLP input (clinical data + VGAE latent features) for a batch.
    Use this function only when training the MLP independently of the VGAE, or
    for evaluating the final (trained) MLP. Because the VGAE is in eval mode.
    
    Parameters:
    ----------
    batch (torch_geometric.data.Batch): Batch of graphs.
    vgae (torch.nn.Module): Trained NodeFeatureVAE model.
    num_z_samples (int): Number of samples to draw from the latent space for training.

    Returns:
    -------
    x (torch.Tensor): MLP input (n_samples*num_z_samples, n_clinical_features+latent_dim).
    '''
    # Get context (for conditioning the readout)
    context = get_context(batch)
    clinical_data = batch.graph_attr

    # Get VGAE outputs
    vgae.eval()
    out = vgae(batch)
    if num_z_samples == 0:
        # Train the MLP on the means of the latent variables
        x = torch.cat([vgae.readout(out.mu, context, batch.batch), clinical_data], dim=1)

    else:
        # Train the MLP on samples from the latent space
        n_samples = clinical_data.size(0)     # Number of samples
        readout_dim = vgae.readout_dim        # Readout dimension size
        clinical_dim = clinical_data.size(1)  # Clinical data feature size

        # Create x tensor
        x = torch.empty((n_samples * num_z_samples, readout_dim + clinical_dim), device=device)
        for i in range(num_z_samples):
            z = vgae.reparameterize(out.mu, out.logvar)  # Sample latent z's
            x[i * n_samples: (i + 1) * n_samples, :] = torch.cat([vgae.readout(z, context, batch.batch), clinical_data], dim=1)

    return x

def get_x_with_vgae_loss(batch, device, vgae, num_z_samples):
    '''
    Get the latent features and VGAE loss for a batch of data.
    Use this function when training the MLP jointly with the VGAE.
    
    Parameters:
    ----------
    batch (torch_geometric.data.Batch): Batch of graphs.
    device (torch.device): Device to use for training.
    vgae (torch.nn.Module): Trained NodeEmbeddingVGAE model.
    num_z_samples (int): Number of samples to draw from the latent space for training.
    condz (torch.Tensor): Shared node feature data for conditioning the latent space.

    Returns:
    -------
    x (torch.Tensor): MLP input (batch_size*min(1, num_z_samples), n_clinical_features+latent_dim).
    vgae_loss (float): average VGAE loss across num_z_samples.
    '''
    # Get context attributes (for conditioning the readout layer)
    context = get_context(batch)
    clinical_data = batch.graph_attr        

    if num_z_samples == 0:
        # Train the MLP on the means of the latent variables
        out = vgae(batch)
        x = torch.cat([vgae.readout(out.mu, context, batch.batch), clinical_data], dim=1)
        vgae_loss = vgae.loss(out)

    else:
        # Train the MLP on samples from the latent space
        out = vgae(batch)
        n_samples = clinical_data.size(0)     # Number of samples
        readout_dim = vgae.readout_dim        # Readout dimension size
        clinical_dim = clinical_data.size(1)  # Clinical data feature size

        # Create x tensor
        vgae_loss = 0.
        x = torch.empty((n_samples * num_z_samples, readout_dim + clinical_dim), device=device)
        for i in range(num_z_samples):
            # Get new encoding each time
            if i != 0:  
                out = vgae(batch)  
            x[i * n_samples: (i + 1) * n_samples, :] = torch.cat([vgae.readout(out.z, context, batch.batch), clinical_data], dim=1)
            vgae_loss += vgae.loss(out)

        vgae_loss = vgae_loss/num_z_samples

    return x, vgae_loss

def train_joint_vgae_mlp(vgae, mlp, loader, optimizer, device, num_z_samples=0, alpha=0.5):
    '''
    Performs joint training of VGAE and MLP models for one epoch.
    
    Parameters:
    -----------
    vgae (torch.nn.Module): VGAE model
    mlp (torch.nn.Module): MLP model
    loader (torch.utils.data.DataLoader): Training data loader
    optimizer (torch.optim.Optimizer): Optimizer
    device (torch.device): Device to use for training.
    num_z_samples (int): Number of samples to draw from the latent space for training.
    alpha (float): Weight of VGAE loss in the total loss
    '''
    vgae.train()
    mlp.train()
    train_loss = 0.
    for batch in loader: 
        # Fetch data and labels
        batch = batch.to(device)             
        ytrue = get_labels(batch, num_z_samples)

        # Forward pass
        optimizer.zero_grad()
        x, vgae_loss = get_x_with_vgae_loss(batch, device, vgae, num_z_samples)
        ypred = mlp(x)                     
        mlp_loss = mlp.loss(ypred, ytrue)
        loss = alpha*vgae_loss + (1-alpha)*mlp_loss

        # Backpropagation
        loss.backward()                      
        optimizer.step()                     
        train_loss += loss.item()

    return train_loss/len(loader)

def test_joint_vgae_mlp(vgae, mlp, loader, device, num_z_samples=0):
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
    mlp_test_loss = 0.
    vgae_test_loss = 0.
    with torch.no_grad():
        for batch in loader:
            # Get inputs and labels
            batch = batch.to(device)
            y_true = get_labels(batch, num_z_samples)

            # Get VGAE loss
            x, vgae_loss = get_x_with_vgae_loss(batch, device, vgae, num_z_samples)
            vgae_test_loss += vgae_loss.item()

            # Get MLP loss
            y_pred = mlp(x)                                   
            mlp_loss = mlp.loss(y_pred, y_true)                   
            mlp_test_loss += mlp_loss.item()

    # Average losses across batch
    vgae_test_loss = vgae_test_loss/len(loader)
    mlp_test_loss = mlp_test_loss/len(loader)

    return vgae_test_loss, mlp_test_loss

@mlp_ingredient.capture
def get_mlp_outputs_nograd(mlp, loader, device, get_x, *args, **kwargs):
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
            clinical_data = batch.graph_attr
            subject_id = batch.subject
            
            # Get MLP predictions on latent means
            if 'num_z_samples' in kwargs:
                kwargs['num_z_samples'] = 0
            x = get_x(batch, device, *args, **kwargs) 
            ypred = mlp(x)                                   

            # Save outputs
            batch_size = clinical_data.shape[0]
            for sub in range(batch_size):
                outputs['clinical_data'].append(tuple(clinical_data[sub, :].tolist()))
            outputs['prediction'].extend(ypred.squeeze(-1).tolist())
            outputs['label'].extend(ytrue.squeeze(-1).tolist())
            outputs['subject_id'].extend(subject_id.tolist())

    return outputs

@mlp_ingredient.capture
def evaluate_regression(outputs: pd.DataFrame):
    '''Evaluates regression results.'''
    # Calculate Pearson correlation coefficient
    r, p = pearsonr(outputs['label'], outputs['prediction'])

    # Compute mean absolute error and standard deviation
    mae = (outputs['label'] - outputs['prediction']).abs().mean()
    mae_std = (outputs['label'] - outputs['prediction']).abs().std()

    return r, p, mae, mae_std
