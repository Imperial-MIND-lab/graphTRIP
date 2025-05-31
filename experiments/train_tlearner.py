'''
In this experiment, we train a model (like train_jointly.py) on one condition (e.g. drugs),
and then evaluate it on another condition.

Dependencies:
- data/raw/{study}/annotations.csv

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
import torch.nn
from torch_geometric.loader import DataLoader
from time import time
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger
from utils.plotting import plot_vgae_reconstructions, plot_loss_curves, true_vs_pred_scatter
from preprocessing.metrics import get_rsn_mapping


# Create experiment and logger -------------------------------------------------
ex = Experiment('transfer_tlearner', ingredients=[data_ingredient, 
                                                  vgae_ingredient, 
                                                  mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'transfer_tlearner'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training configurations
    lr = 0.001            # Learning rate.
    num_epochs = 200      # Number of epochs to train.
    num_z_samples = 5     # 0 for training MLP on the means of VGAE latent variables.
    alpha = 0.5           # Loss = alpha*vgae_loss + (1-alpha)*mlp_loss

    # Condition settings
    annotations_file = 'data/raw/psilodep2/annotations.csv'
    subject_id_col = 'Patient' # annotations[subject_id_col] == data[i].subject+1
    train_cond = 'P'   # must match the "Condition" column in annotations.csv
    eval_cond = 'E'    # must match the "Condition" column in annotations.csv

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Make sure the annotations file exists
    annotations_file = config.get('annotations_file', cfg().get('annotations_file'))
    annotations_file = add_project_root(annotations_file)
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f'Annotations file not found: {annotations_file}')
    
    # Check compatibility of the annotations file
    annotations = pd.read_csv(annotations_file)

    # "Condition" column must be present in the annotations file
    if 'Condition' not in annotations.columns:
        raise ValueError('"Condition" column not found in annotations file.')
    
    # Make sure the train and eval conditions are present in the annotations file
    train_cond = config.get('train_cond', cfg().get('train_cond'))
    eval_cond = config.get('eval_cond', cfg().get('eval_cond'))
    if train_cond not in annotations['Condition'].unique():
        raise ValueError(f'Train condition not found in annotations file: {train_cond}')
    if eval_cond not in annotations['Condition'].unique():
        raise ValueError(f'Eval condition not found in annotations file: {eval_cond}')
    
    # Make sure the subject ID column is present in the annotations file
    subject_id_col = config.get('subject_id_col', cfg().get('subject_id_col'))
    if subject_id_col not in annotations.columns:
        raise ValueError(f'Subject ID column not found in annotations file: {subject_id_col}')
    
    # Don't support: node feature standardisation, validation split, number of folds
    dataset = config.get('dataset', {})
    if dataset.get('val_split', 0.) != 0.:
        logger.warning('Validation split not supported for this experiment. Setting val_split=0.')
    if dataset.get('num_folds', 1) != 1:
        logger.warning('K-fold cross validation not supported for this experiment. Setting num_folds=1.')
    if dataset.get('standardise_x', False):
        logger.warning('Node feature standardisation not supported for this experiment. Setting standardise_x=False.')
    config['dataset']['standardise_x'] = False
    config['dataset']['val_split'] = 0.
    config['dataset']['num_folds'] = 1
    
    return config

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(vgae, mlp, lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    return torch.optim.Adam(list(mlp.parameters()) + list(vgae.parameters()), lr=lr)

@ex.capture
def train_vgae_mlp(vgae, mlp, loader, optimizer, device, num_z_samples, alpha):
    return train_joint_vgae_mlp(vgae, mlp, loader, optimizer, 
                                device=device, num_z_samples=num_z_samples, alpha=alpha)

@ex.capture
def test_vgae_mlp(vgae, mlp, loader, device, num_z_samples):
    vgae_test_loss, mlp_test_loss = test_joint_vgae_mlp(vgae, mlp, loader, 
                                                        device=device, num_z_samples=num_z_samples)
    return vgae_test_loss, mlp_test_loss

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Unpack configs
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']

    # Annotations config
    annotations_file = add_project_root(_config['annotations_file'])
    train_cond = _config['train_cond']
    eval_cond = _config['eval_cond']
    subject_id_col = _config['subject_id_col']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Load the full dataset
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load annotations and split the data into train and test conditions
    # Note: the subject IDs in the annotations file are 1-indexed, but the subject IDs in the data are 0-indexed !
    annotations = pd.read_csv(annotations_file)
    train_ids = annotations[annotations['Condition'] == train_cond][subject_id_col].to_numpy()-1
    test_ids = annotations[annotations['Condition'] == eval_cond][subject_id_col].to_numpy()-1

    # Convert to matching indices for slicing the dataset
    train_indices = np.array([i for i in range(len(data)) if data[i].subject.item() in train_ids])
    test_indices = np.array([i for i in range(len(data)) if data[i].subject.item() in test_ids])

    # Split the data into train and test conditions
    train_data = data[train_indices]
    test_data = data[test_indices]

    # Make data loaders
    batch_size = _config['dataset']['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Training loop ------------------------------------------------------------
    start_time = time()

    # Initialise models and optimizer
    vgae = build_vgae().to(device)
    mlp = build_mlp(latent_dim=vgae.readout_dim).to(device)
    optimizer = get_optimizer(vgae, mlp)

    # Track losses
    vgae_train_loss, vgae_test_loss = [], []
    mlp_train_loss, mlp_test_loss = [], []

    for epoch in range(_config['num_epochs']):
        # Train VGAE and MLP
        _ = train_vgae_mlp(vgae, mlp, train_loader, optimizer, device)
        
        # Compute training losses
        vgae_train_loss_epoch, mlp_train_loss_epoch = test_vgae_mlp(vgae, mlp, train_loader, device)
        vgae_train_loss.append(vgae_train_loss_epoch)
        mlp_train_loss.append(mlp_train_loss_epoch)
        
        # Test VGAE and MLP
        vgae_test_loss_epoch, mlp_test_loss_epoch = test_vgae_mlp(vgae, mlp, test_loader, device)
        vgae_test_loss.append(vgae_test_loss_epoch)
        mlp_test_loss.append(mlp_test_loss_epoch)

        # Log training and test losses
        ex.log_scalar(f'training/epoch/vgae_loss', vgae_train_loss_epoch)
        ex.log_scalar(f'training/epoch/mlp_loss', mlp_train_loss_epoch)
        ex.log_scalar(f'test/epoch/vgae_loss', vgae_test_loss_epoch)
        ex.log_scalar(f'test/epoch/mlp_loss', mlp_test_loss_epoch)

    # Save model weights
    if save_weights:
        torch.save(mlp.state_dict(), os.path.join(output_dir, f'mlp_weights_{train_cond}.pth'))
        torch.save(vgae.state_dict(), os.path.join(output_dir, f'vgae_weights_{train_cond}.pth'))

    # Get predictions for both conditions
    train_outputs = init_outputs_dict(train_data)
    outputs = get_mlp_outputs_nograd(mlp, train_loader, device, 
                                     get_x=get_x_with_vgae, 
                                     vgae=vgae, num_z_samples=0)
    update_best_outputs(train_outputs, outputs, _config['dataset']['graph_attrs'])
    
    test_outputs = init_outputs_dict(test_data)
    outputs = get_mlp_outputs_nograd(mlp, test_loader, device, 
                                     get_x=get_x_with_vgae, 
                                     vgae=vgae, num_z_samples=0)
    update_best_outputs(test_outputs, outputs, _config['dataset']['graph_attrs'])
    
    # Process and save outputs for each condition
    for cond, outputs in [(train_cond, train_outputs), (eval_cond, test_outputs)]:
        # Convert to DataFrame and add condition info
        outputs_df = pd.DataFrame(outputs)
        outputs_df = add_drug_condition_to_outputs(outputs_df, _config['dataset']['study'])
        
        # Save prediction results
        data_file = os.path.join(output_dir, f'prediction_results_{cond}.csv')
        outputs_df.to_csv(data_file, index=False)
        
        # Calculate and save metrics
        r, p, mae, mae_std = evaluate_regression(outputs_df)
        results = {'seed': seed, 'r': r, 'p': p, 'mae': mae, 'mae_std': mae_std}
        pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, f'final_metrics_{cond}.csv'), index=False)
        
        # Log metrics
        for k, v in results.items():
            ex.log_scalar(f'final_prediction/{cond}/{k}', v)
        logger.info(f"Results for {cond}: r={r:.4f}, p={p:.4e}, mae={mae:.4f} ± {mae_std:.4f}.")
        
        # Create true vs predicted scatter plot
        title = f'{cond}: r={r:.4f}, p={p:.4e}, MAE={mae:.4f} ± {mae_std:.4f}'
        save_path = os.path.join(output_dir, f'true_vs_predicted_{cond}.png')
        true_vs_pred_scatter(outputs_df, title=title, save_path=save_path)
        image_files.append(save_path)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Plot VGAE reconstructions -------------------------------------------------
    # Get reconstructions for both conditions
    # For training condition
    adj_orig_rcn_train, x_orig_rcn_train, fold_assignments_train = get_test_reconstructions(
        [vgae], train_data, [np.arange(len(train_data))], mean_std=None)
    
    # For test condition
    adj_orig_rcn_test, x_orig_rcn_test, fold_assignments_test = get_test_reconstructions(
        [vgae], test_data, [np.arange(len(test_data))], mean_std=None)
    
    # Evaluate FC reconstructions for both conditions
    for cond, adj_orig_rcn in [(train_cond, adj_orig_rcn_train), (eval_cond, adj_orig_rcn_test)]:
        adj_orig_rcn['metrics'] = evaluate_fc_reconstructions(adj_orig_rcn)
        ex.log_scalar(f'final_reconstruction/{cond}/fc_corr', np.mean(adj_orig_rcn['metrics']['corr']))
        ex.log_scalar(f'final_reconstruction/{cond}/fc_mae', np.mean(adj_orig_rcn['metrics']['mae']))

    # Evaluate node feature reconstructions for both conditions
    for cond, x_orig_rcn in [(train_cond, x_orig_rcn_train), (eval_cond, x_orig_rcn_test)]:
        if x_orig_rcn:
            x_orig_rcn['metrics'] = evaluate_x_reconstructions(x_orig_rcn)
            for feature in x_orig_rcn['feature_names']:
                ex.log_scalar(f'final_reconstruction/{cond}/x_{feature}_corr', x_orig_rcn['metrics']['corr'][feature].mean())
                ex.log_scalar(f'final_reconstruction/{cond}/x_{feature}_mae', x_orig_rcn['metrics']['mae'][feature].mean())
    
    # Plot reconstructions for both conditions
    rsn_mapping, rsn_labels = get_rsn_mapping(data.atlas)
    vrange = None
    if _config['dataset']['edge_attrs'][0] == 'functional_connectivity':
        vrange = (-0.7, 0.7)
    
    # Plot training condition reconstructions
    save_path = os.path.join(output_dir, f'vgae_reconstructions_{train_cond}.png')
    conditions = None
    if train_cond == 'P':
        conditions = [1.0]*len(train_data)
    elif train_cond == 'E':
        conditions = [-1.0]*len(train_data)
    image_files += plot_vgae_reconstructions(adj_orig_rcn_train, 
                                           x_orig_rcn_train, 
                                           fold_assignments_train,
                                           conditions=conditions,
                                           rsn_mapping=rsn_mapping,
                                           rsn_labels=rsn_labels,
                                           atlas=data.atlas,
                                           vrange=vrange, 
                                           save_path=save_path)
    
    # Plot test condition reconstructions
    save_path = os.path.join(output_dir, f'vgae_reconstructions_{eval_cond}.png')
    conditions = None
    if eval_cond == 'E':
        conditions = [-1.0]*len(test_data)
    elif eval_cond == 'P':
        conditions = [1.0]*len(test_data)
    image_files += plot_vgae_reconstructions(adj_orig_rcn_test, 
                                           x_orig_rcn_test, 
                                           fold_assignments_test,
                                           conditions=conditions,
                                           rsn_mapping=rsn_mapping,
                                           rsn_labels=rsn_labels,
                                           atlas=data.atlas,
                                           vrange=vrange, 
                                           save_path=save_path)

    # Plot loss curves
    plot_loss_curves({0: vgae_train_loss}, {0: vgae_test_loss}, {0: []}, 
                     save_path=os.path.join(output_dir, f'vgae_loss_curves_{train_cond}to{eval_cond}.png'))
    plot_loss_curves({0: mlp_train_loss}, {0: mlp_test_loss}, {0: []}, 
                     save_path=os.path.join(output_dir, f'mlp_loss_curves_{train_cond}to{eval_cond}.png'))
    image_files += [os.path.join(output_dir, f'vgae_loss_curves_{train_cond}to{eval_cond}.png'), 
                   os.path.join(output_dir, f'mlp_loss_curves_{train_cond}to{eval_cond}.png')]

    # Log images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close()
