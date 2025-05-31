'''
Joint training of VGAE and Logistic Regression models for predicting treatment.
Both models are newly initialised, no pre-trained model loading.

Authors: Hanna M. Tolle
Date: 2025-04-18
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import numpy as np

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices
from utils.plotting import plot_vgae_reconstructions, plot_loss_curves
from preprocessing.metrics import get_rsn_mapping


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_classifier', ingredients=[data_ingredient, 
                                                         vgae_ingredient, 
                                                         mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg():
    # Experiment name and ID
    exname = 'train_classifier'
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
    num_epochs = 300      # Number of epochs to train.
    num_z_samples = 1     # 0 for training MLP on the means of VGAE latent variables.
    alpha = 0.5           # Loss = alpha*vgae_loss + (1-alpha)*mlp_loss

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # MLP has to be a LogisticRegressionMLP
    if 'mlp_model' in config:
        model_type = config['mlp_model'].get('model_type', 'LogisticRegressionMLP')
        if model_type != 'LogisticRegressionMLP':
            raise ValueError('MLP has to be a LogisticRegressionMLP.')
        config['mlp_model']['model_type'] = 'LogisticRegressionMLP'
    else:
        config['mlp_model'] = {'model_type': 'LogisticRegressionMLP'}

    # Dataset configuration checks 
    if 'dataset' in config:
        # drug_condition has to be None 
        # (because this fixes the treatment for all patients to a constant value)
        drug_condition = config['dataset'].get('drug_condition', None)
        if drug_condition is not None:
            raise ValueError('drug_condition has to be None for classifying treatment.')
        config['dataset']['drug_condition'] = None

        # Condition cannot be part of context attributes
        context_attrs = config['dataset'].get('context_attrs', [])
        if 'Condition' in context_attrs:
            raise ValueError('Condition has to be excluded from context attributes for classifying treatment.')
        config['dataset']['context_attrs'] = [attr for attr in context_attrs if attr != 'Condition']

        # Condition cannot be part of the graph attributes
        graph_attrs = config['dataset'].get('graph_attrs', [])
        if 'Condition' in graph_attrs:
            raise ValueError('Condition has to be excluded from graph attributes for classifying treatment.')
        config['dataset']['graph_attrs'] = [attr for attr in graph_attrs if attr != 'Condition']

    else:
        config['dataset'] = {'drug_condition': None, 
                             'context_attrs': [],
                             'graph_attrs': []}

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
    alpha = _config['alpha']
    num_folds = _config['dataset']['num_folds']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Get dataloaders
    data = load_data()
    train_loaders, val_loaders, test_loaders, test_indices, mean_std \
        = get_kfold_dataloaders(data, seed=seed)
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Train-test loop ------------------------------------------------------------
    start_time = time()

    best_outputs = init_outputs_dict(data)
    vgae_train_loss, vgae_test_loss, vgae_val_loss = {}, {}, {}
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}
    best_vgae_states, best_mlp_states = [], []

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        vgae_train_loss[k], vgae_test_loss[k], vgae_val_loss[k] = [], [], []
        
        # Initialise models and optimizer
        vgae = build_vgae().to(device)
        mlp = build_mlp(latent_dim=vgae.readout_dim).to(device)
        
        # Compute positive weight for this fold
        y_train = torch.cat([batch.y for batch in train_loaders[k]])
        mlp.set_pos_weight(y_train)

        # Initialise optimizer
        optimizer = get_optimizer(vgae, mlp)

        # Best validation loss and early stopping counter
        best_val_loss = float('inf')
        best_vgae_state = None
        best_mlp_state = None

        for epoch in range(_config['num_epochs']):
            # Train VGAE and MLP
            _ = train_vgae_mlp(vgae, mlp, train_loaders[k], optimizer, device)
            
            # Compute training losses
            vgae_train_loss_epoch, mlp_train_loss_epoch = test_vgae_mlp(vgae, mlp, train_loaders[k], device)
            vgae_train_loss[k].append(vgae_train_loss_epoch)
            mlp_train_loss[k].append(mlp_train_loss_epoch)
            
            # Test VGAE and MLP
            vgae_test_loss_epoch, mlp_test_loss_epoch = test_vgae_mlp(vgae, mlp, test_loaders[k], device)
            vgae_test_loss[k].append(vgae_test_loss_epoch)
            mlp_test_loss[k].append(mlp_test_loss_epoch)

            # Log training and test losses
            ex.log_scalar(f'training/fold{k}/epoch/vgae_loss', vgae_train_loss_epoch)
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/vgae_loss', vgae_test_loss_epoch)
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss_epoch)
            
            # Validate models, if applicable
            if len(val_loaders) > 0:
                vgae_val_loss_epoch, mlp_val_loss_epoch = test_vgae_mlp(vgae, mlp, val_loaders[k], device)
                vgae_val_loss[k].append(vgae_val_loss_epoch)
                mlp_val_loss[k].append(mlp_val_loss_epoch)

                # Log validation losses
                ex.log_scalar(f'validation/fold{k}/epoch/vgae_loss', vgae_val_loss_epoch)
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss_epoch)
                
                # Save the best model if validation loss is at its minimum
                total_val_loss = alpha*vgae_val_loss_epoch + (1-alpha)*mlp_val_loss_epoch
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
        outputs = get_mlp_outputs_nograd(mlp, test_loaders[k], device, 
                                         get_x=get_x_with_vgae, 
                                         vgae=vgae, num_z_samples=0)
        update_best_outputs(best_outputs, outputs)

    # Print training time
    end_time = time()
    logger.info(f"Joint training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save ouputs as csv file
    best_outputs = pd.DataFrame(best_outputs)
    best_outputs = add_drug_condition_to_outputs(best_outputs, _config['dataset']['study'])
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    best_outputs.to_csv(data_file, index=False)

    # Save test fold assignments
    test_indices_file = save_test_indices(test_indices, output_dir)

    # Save final prediction results
    # Convert predictions to binary (threshold at 0.5)
    y_true = best_outputs['label'].values
    y_pred = (best_outputs['prediction'].values > 0.5).astype(int)
    y_prob = best_outputs['prediction'].values  # Keep probabilities for ROC curve

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_prob)  # No need for multi_class parameter in binary case

    results = {
        'seed': seed,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Final results: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}")

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
    
    # Plot VGAE reconstructions
    rsn_mapping, rsn_labels = get_rsn_mapping(data.atlas)
    save_path = os.path.join(output_dir, 'vgae.png')
    vrange = None
    if _config['dataset']['edge_attrs'][0] == 'functional_connectivity':
        vrange = (-0.7, 0.7)
    image_files += plot_vgae_reconstructions(adj_orig_rcn, 
                                             x_orig_rcn, 
                                             fold_assignments, 
                                             conditions=None,
                                             rsn_mapping=rsn_mapping,
                                             rsn_labels=rsn_labels,
                                             atlas=data.atlas,
                                             vrange=vrange, 
                                             save_path=save_path)
    
    # Loss curves
    plot_loss_curves(vgae_train_loss, vgae_test_loss, vgae_val_loss, save_path=os.path.join(output_dir, 'vgae_loss_curves.png'))
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=os.path.join(output_dir, 'mlp_loss_curves.png'))
    image_files += [os.path.join(output_dir, 'vgae_loss_curves.png'), os.path.join(output_dir, 'mlp_loss_curves.png')]

    # Classification evaluation plots
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    image_files.append(os.path.join(output_dir, 'confusion_matrix.png'))

    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    image_files.append(os.path.join(output_dir, 'roc_curve.png'))

    # 3. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    image_files.append(os.path.join(output_dir, 'precision_recall_curve.png'))

    # Log images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)

    # Close all plots if not verbose
    if not verbose:
        plt.close()
