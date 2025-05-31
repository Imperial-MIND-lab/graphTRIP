'''
Trains a Logistic Regression model for predicting treatment
from the latent representations of a pre-trained (frozen) VGAE model.

Authors: Hanna M. Tolle
Date: 2025-04-19
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
from sklearn.metrics import precision_recall_curve, roc_curve
import seaborn as sns
import numpy as np

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, save_test_indices, check_weights_exist
from utils.plotting import plot_loss_curves
from utils.configs import load_ingredient_configs, match_ingredient_configs
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('train_rep_classifier', ingredients=[data_ingredient, 
                                              vgae_ingredient, 
                                              mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'train_rep_classifier'
    jobid = 0
    seed = 291
    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    # Logging and saving
    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    save_weights = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Pre-trained VGAE model
    weights_dir = 'outputs/runs/train_xlearner/best_model'
    weight_filenames = {'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
                        'test_fold_indices': ['test_fold_indices.csv']}

    # Training configurations
    lr = 0.001            # Learning rate.
    num_epochs = 300      # Number of epochs to train.
    num_z_samples = 1     # 0 for training MLP on the means of VGAE latent variables.

# Match configs function -------------------------------------------------------
def match_config(config: Dict) -> Dict:
    '''Matches the configs specific to this experiment.'''
    # Get weights_dir (must be in the config)
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    # Load the VGAE and dataset configs from weights_dir
    ingredients = ['vgae_model', 'dataset']     
    previous_config = load_ingredient_configs(weights_dir, ingredients)

    # Match all dataset and VGAE configs
    config_updates = copy.deepcopy(config)
    exceptions = ['target'] # predict the treatment, not the target of the pre-trained VGAE.
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Other config checks
    num_pretrained_models = previous_config['dataset']['num_folds']
    default_weight_filenames = {'vgae': [f'k{i}_vgae_weights.pth' for i in range(num_pretrained_models)], 
                                'test_fold_indices': ['test_fold_indices.csv']}
    weight_filenames = config.get('weight_filenames', default_weight_filenames)
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames

    # MLP has to be a LogisticRegressionMLP
    if 'mlp_model' in config_updates:
        model_type = config_updates['mlp_model'].get('model_type', 'LogisticRegressionMLP')
        if model_type != 'LogisticRegressionMLP':
            raise ValueError('MLP has to be a LogisticRegressionMLP.')
        config_updates['mlp_model']['model_type'] = 'LogisticRegressionMLP'
    else:
        config_updates['mlp_model'] = {'model_type': 'LogisticRegressionMLP'}

    return config_updates

# Captured functions -----------------------------------------------------------
@ex.capture
def get_optimizer(mlp, lr):
    '''Creates the optimizer for the joint training of VGAE and MLP.'''
    return torch.optim.Adam(list(mlp.parameters()), lr=lr)

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Unpack configs
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    num_folds = _config['dataset']['num_folds']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']
    num_z_samples = _config['num_z_samples']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []    

    # Load dataset
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load pretrained VGAE and testfold indices
    pretrained_vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device=device)
    test_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)
    train_loaders, val_loaders, test_loaders, _ = get_dataloaders_from_test_indices(data, test_indices, seed=seed)

    # Train-test loop ------------------------------------------------------------
    start_time = time()

    best_outputs = init_outputs_dict(data)
    mlp_train_loss, mlp_test_loss, mlp_val_loss = {}, {}, {}
    best_mlp_states = []

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):

        # Initialise losses
        mlp_train_loss[k], mlp_test_loss[k], mlp_val_loss[k] = [], [], []
        
        # Get pretrained VGAE and freeze it
        vgae = pretrained_vgaes[k].to(device)
        freeze_model(vgae)

        # Initialise classifier (LogisticRegressionMLP)
        mlp = build_mlp(latent_dim=vgae.readout_dim).to(device)
        
        # Compute positive weight for this fold
        y_train = torch.cat([batch.y for batch in train_loaders[k]])
        mlp.set_pos_weight(y_train)

        # Initialise optimizer
        optimizer = get_optimizer(mlp)

        # Best validation loss and early stopping counter
        best_val_loss = float('inf')
        best_mlp_state = None

        for epoch in range(_config['num_epochs']):
            # Train MLP
            _ = train_mlp(mlp, train_loaders[k], optimizer, device, 
                          get_x=get_x_with_vgae, vgae=vgae, num_z_samples=num_z_samples)
            
            # Compute training loss
            mlp_train_loss[k].append(test_mlp(mlp, train_loaders[k], device, 
                                              get_x=get_x_with_vgae, vgae=vgae, 
                                              num_z_samples=num_z_samples))
            
            # Test MLP
            mlp_test_loss[k].append(test_mlp(mlp, test_loaders[k], device, 
                                             get_x=get_x_with_vgae, vgae=vgae, 
                                             num_z_samples=num_z_samples))

            # Log training and test losses
            ex.log_scalar(f'training/fold{k}/epoch/mlp_loss', mlp_train_loss[k][-1])
            ex.log_scalar(f'test/fold{k}/epoch/mlp_loss', mlp_test_loss[k][-1])
            
            # Validate models, if applicable
            if len(val_loaders) > 0:
                mlp_val_loss[k].append(test_mlp(mlp, val_loaders[k], device, 
                                                 get_x=get_x_with_vgae, vgae=vgae, 
                                                 num_z_samples=num_z_samples))

                # Log validation losses
                ex.log_scalar(f'validation/fold{k}/epoch/mlp_loss', mlp_val_loss[k][-1])
                
                # Save the best model if validation loss is at its minimum
                if mlp_val_loss[k][-1] < best_val_loss:
                    best_val_loss = mlp_val_loss[k][-1]
                    best_mlp_state = copy.deepcopy(mlp.state_dict()) 

        # Load best model of this fold 
        if best_mlp_state is not None:
            mlp.load_state_dict(best_mlp_state)

        # Save model weights
        if save_weights:
            torch.save(mlp.state_dict(), os.path.join(output_dir, f'k{k}_mlp_weights.pth'))

        # Keep a list of model states
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
    test_indices_list = [np.where(test_indices == fold)[0] for fold in range(num_folds)]
    _ = save_test_indices(test_indices_list, output_dir)

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
    
    # Loss curves
    plot_loss_curves(mlp_train_loss, mlp_test_loss, mlp_val_loss, save_path=os.path.join(output_dir, 'mlp_loss_curves.png'))
    image_files += [os.path.join(output_dir, 'mlp_loss_curves.png')]

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
