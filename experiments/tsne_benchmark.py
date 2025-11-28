'''
Trains an MLP on t-SNE embeddings of flattened graph features.

Author: Hanna M. Tolle
Date: 2025-02-17
License: BSD 3-Clause
'''

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import * 
from experiments.ingredients.mlp_ingredient import * 

import os
import torch
import torch.nn
from torch_geometric.data import Batch
from tqdm import tqdm
from time import time
import copy
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.files import project_root, add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices
from utils.plotting import plot_loss_curves, true_vs_pred_scatter

# Torch dataset for t-SNE embeddings -------------------------------------------
class TSNEDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, targets, clinical_data, subject_ids):
        assert embeddings.shape[0] == targets.shape[0] == clinical_data.shape[0], \
            "Embeddings, targets and clinical data must have the same number of samples."
        
        if isinstance(embeddings, torch.Tensor):
            self.embeddings = embeddings.clone().detach().to(dtype=torch.float32)
        else:
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        if isinstance(targets, torch.Tensor):
            self.targets = targets.clone().detach().to(dtype=torch.float32)
        else:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        
        # Match MLP output shape
        if self.targets.dim() == 1:
            self.targets = self.targets.unsqueeze(-1)
        
        if isinstance(clinical_data, torch.Tensor):
            self.clinical_data = clinical_data.clone().detach().to(dtype=torch.float32)
        else:
            self.clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx], self.clinical_data[idx], self.subject_ids[idx]

# Create experiment and logger -------------------------------------------------
ex = Experiment('tsne_benchmark', ingredients=[data_ingredient, mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'tsne_benchmark'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join(project_root(), 'outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training configurations
    lr = 0.001            # Learning rate
    num_epochs = 200      # Number of epochs to train
    n_components = 3      # Number of t-SNE components
    perplexity = 30       # t-SNE perplexity
    balance_attrs = None  # attrs to balance on for k-fold CV. If None, no balancing.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    n_components = config.get('n_components', 3)
    assert n_components <= 3, f"n_components must be 3 or less. Got {n_components}."
    # Make sure the dataset has no Adjacency thresholding datatransform
    edge_tfm_type = config.get('dataset', {}).get('edge_tfm_type', None)
    permitted_edge_tfm_types = [None, 'ApplyAdjacency']
    if edge_tfm_type not in permitted_edge_tfm_types:
        # Because t-SNE requires an equal number of features for each graph
        raise ValueError(f"edge_tfm_type must be one of {permitted_edge_tfm_types}.")
    return config

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(mlp, lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    return torch.optim.Adam(list(mlp.parameters()), lr=lr)

def get_flattened_features(batch):
    '''
    Get all features from the batch and flatten them.
    All graphs must have the same number of nodes and edges.
    
    Parameters:
    ----------
    batch (torch_geometric.data.Batch): Batch of graphs.
    device (torch.device): Device to use for training.

    Returns:
    -------
    x (torch.Tensor): MLP input 
        (batch_size, 
          num_graph_attrs 
        + num_node_attrs*num_nodes 
        + num_edge_attrs*num_edges).
    '''
    # Get graph, node and edge features and reshape
    batch_size = batch.num_graphs
    node_features = batch.x.view(batch_size, -1)          # [batch_size, num_nodes * num_node_attrs]
    edge_features, _ = get_triu_edges(batch)              # [num_triu_edges*batch_size, num_edge_attr]
    edge_features = edge_features.view(batch_size, -1)    # [batch_size, num_triu_edges * num_edge_attrs]
    
    # Concatenate all features
    flattened_features = torch.cat([node_features, edge_features], dim=1)
    return flattened_features

@ex.capture
def create_tsne_dataset(data, n_components, perplexity, seed):
    '''
    Creates train/val/test datasets from t-SNE embeddings.
    
    Parameters:
    -----------
    data (torch_geometric.data): The entire torch geometric dataset.
    test_indices (numpy.array): Array where test_indices[i] indicates,
      which test fold sample i belongs to.
    n_components (int): Number of t-SNE components.
    perplexity (int): t-SNE perplexity.
    seed (int): Random seed.

    Returns:
    --------
    train_loaders (list): List of train dataloaders.
    val_loaders (list): List of val dataloaders.
    test_loaders (list): List of test dataloaders.
    '''
    # Get flattened features from all graphs
    batch = Batch.from_data_list([graph for graph in data])
    flattened_features = get_flattened_features(batch).cpu().numpy()
    subject_ids = batch.subject

    # Compute t-SNE embeddings
    logger.info('Computing t-SNE embeddings...')
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed)
    embeddings = tsne.fit_transform(flattened_features)
    
    tsne_dataset = TSNEDataset(embeddings=embeddings, 
                               targets=batch.y, 
                               clinical_data=batch.graph_attr,
                               subject_ids=subject_ids)
    return tsne_dataset

def get_x(embeddings, clinical_data):
    '''
    Concatenates the t-SNE embeddings and the clinical data.
    '''
    return torch.cat([clinical_data, embeddings], dim=1)

def train_tsne_mlp(mlp, loader, optimizer, device):
    '''
    Performs training of the RegressionMLP.
    
    Parameters:
    -----------
    mlp (torch.nn.Module): RegressionMLP model
    loader (torch.utils.data.DataLoader): Training data loader
    optimizer (torch.optim.Optimizer): Optimizer
    device (torch.device): Device to use for training.
    '''
    mlp.train()
    train_loss = 0.
    for embeddings, ytrue, clinical_data, _ in loader: 
        # Move data and labels to device
        embeddings = embeddings.to(device)             
        ytrue = ytrue.to(device)
        clinical_data = clinical_data.to(device)

        # Forward pass
        optimizer.zero_grad()
        x = get_x(embeddings=embeddings, clinical_data=clinical_data)
        ypred = mlp(x)                     
        loss = mlp.loss(ypred, ytrue)

        # Backpropagation
        loss.backward()                      
        optimizer.step()                     
        train_loss += loss.item()

    return train_loss/len(loader)

def test_tsne_mlp(mlp, loader, device):
    '''
    Evaluates the RegressionMLP. 
    Parameters:
    ----------
    mlp (torch.nn.Module): trained RegressionMLP model.
    loader (torch.utils.data.DataLoader): test data loader.
    device (torch.device): Device to use for testing.
    Returns:
    -------
    test_loss (float): average loss across test dataset.
    '''
    mlp.eval()
    test_loss = 0.
    with torch.no_grad():
        for embeddings, ytrue, clinical_data, _ in loader:
            # Move data and labels to device
            embeddings = embeddings.to(device)             
            ytrue = ytrue.to(device)
            clinical_data = clinical_data.to(device)

            # Forward pass
            x = get_x(embeddings=embeddings, clinical_data=clinical_data)
            ypred = mlp(x)                     

            # Get MLP loss
            loss = mlp.loss(ypred, ytrue)                   
            test_loss += loss.item()

    return test_loss/len(loader)

def get_tsne_mlp_outputs_nograd(mlp, loader, device):
    '''
    Returns the outputs of the MLP. No gradients are computed.
    Parameters:
    ----------
    mlp (torch.nn.Module): trained MLP model.
    loader (Dataloader): pytorch geometric test data loader.
    device (torch.device): Device to use for testing.
                      
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
        for embeddings, ytrue, clinical_data, subject_id in loader:
            # Move data and labels to device
            embeddings = embeddings.to(device)             
            ytrue = ytrue.to(device)
            clinical_data = clinical_data.to(device)
            
            # Get MLP predictions on latent means
            x = get_x(embeddings=embeddings, clinical_data=clinical_data)
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

    # Unpack configs
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    num_folds = _config['dataset']['num_folds']
    n_components = _config['n_components']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Get data and create t-SNE datasets
    data = load_data()
    assert _config['perplexity'] < len(data), "perplexity must be less than the number of samples."
    tsne_dataset = create_tsne_dataset(data)
    
    # Get dataloaders with optional balancing
    balance_attrs = _config['balance_attrs']
    if balance_attrs is not None:
        train_loaders, val_loaders, test_loaders, test_indices = get_balanced_tsne_dataloaders(
            tsne_dataset, data, balance_attrs, seed=seed)
    else:
        train_loaders, val_loaders, test_loaders, test_indices = get_tsne_dataloaders(tsne_dataset, seed=seed)

    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Train-test loop ------------------------------------------------------------
    start_time = time()

    best_outputs = init_outputs_dict(data)
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}
    best_mlp_states = []

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        
        # Initialise models and optimizer
        mlp = build_mlp(latent_dim=n_components).to(device)
        optimizer = get_optimizer(mlp)

        # Best validation loss
        best_val_loss = float('inf')
        best_mlp_state = None

        for epoch in range(_config['num_epochs']):
            # Train MLP
            _ = train_tsne_mlp(mlp, train_loaders[k], optimizer, device)
            
            # Compute training losses
            mlp_train_loss_epoch = test_tsne_mlp(mlp, train_loaders[k], device)
            mlp_train_loss[k].append(mlp_train_loss_epoch)
            
            # Test MLP
            mlp_test_loss_epoch = test_tsne_mlp(mlp, test_loaders[k], device)
            mlp_test_loss[k].append(mlp_test_loss_epoch)

            # Log training and test losses
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss_epoch)

            # Validate models, if applicable
            if len(val_loaders) > 0:
                mlp_val_loss_epoch = test_tsne_mlp(mlp, val_loaders[k], device)
                mlp_val_loss[k].append(mlp_val_loss_epoch)
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss_epoch)
                
                # Save the best model if validation loss is at its minimum
                if mlp_val_loss_epoch < best_val_loss:
                    best_val_loss = mlp_val_loss_epoch
                    best_mlp_state = copy.deepcopy(mlp.state_dict()) 

        # Load best model of this fold 
        if best_mlp_state is not None:
            mlp.load_state_dict(best_mlp_state)

        # Save model weights
        if save_weights:
            torch.save(mlp.state_dict(), os.path.join(output_dir, f'k{k}_mlp_weights.pth'))

        # Keep a list of model states
        best_mlp_states.append(copy.deepcopy(mlp.state_dict()))

        # Save the test predictions of the best model
        outputs = get_tsne_mlp_outputs_nograd(mlp, test_loaders[k], device)
        update_best_outputs(best_outputs, outputs)

    # Print training time
    end_time = time()
    logger.info(f"MLP training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save ouputs as csv file
    best_outputs = pd.DataFrame(best_outputs)
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    best_outputs.to_csv(data_file, index=False)

    # Save test fold assignments
    if save_weights:
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
    # Loss curves
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=os.path.join(output_dir, 'mlp_loss_curves.png'))
    image_files.append(os.path.join(output_dir, 'mlp_loss_curves.png'))

    # True vs predicted scatter
    title = f'r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
    true_vs_pred_scatter(best_outputs, title=title, save_path=os.path.join(output_dir, 'true_vs_predicted.png'))
    image_files.append(os.path.join(output_dir, 'true_vs_predicted.png'))

    # Log images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close()
