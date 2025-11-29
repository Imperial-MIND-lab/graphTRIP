"""
Loads a pre-trained VGAE, obtains latent representations,
and trains a logistic regression model on the latent representations
for binary classification (propensity score estimation).

Author: Hanna Tolle
Date: 2025-11-19 
License: BSD-3-Clause
"""

import sys
sys.path.append('graphTRIP/')

# # non-interactive backend
# import matplotlib
# matplotlib.use('Agg') 

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
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve
import seaborn as sns
from typing import Dict

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.configs import load_ingredient_configs, match_ingredient_configs
from models.utils import freeze_model
from utils.plotting import plot_raincloud, PSILO, ESCIT


# Create experiment and logger -------------------------------------------------
ex = Experiment('estimate_propensity', ingredients=[data_ingredient, 
                                                    vgae_ingredient])
logger = get_logger()
ex.logger = logger

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

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    # Experiment name and ID
    exname = 'estimate_propensity'
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
    reinit_pooling = False   # Whether to reinitialise the VGAE pooling module
    logit_C = 1.0            # Inverse reg. strength for logit (smaller = stronger reg)
    n_pca_components = 0     # Performs PCA on Z if n_pca_components > 0
    n_permutations = 1000    # Number of permutations for AUC permutation test

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
    exceptions = ['graph_attrs', 'target']
    reinit_pooling = config.get('reinit_pooling', False)
    if reinit_pooling:
        exceptions.append('pooling_cfg')
    config_updates = match_ingredient_configs(config=config,
                                              previous_config=previous_config,
                                              ingredients=ingredients,
                                              exceptions=exceptions)

    # Assert that target is binary
    assert config_updates['dataset']['target'] == 'Condition_bin01', \
        f"Target must be 'Condition_bin01' for binary classification, got '{config_updates['dataset']['target']}'"

    # Other config checks
    num_pretrained_models = previous_config['dataset']['num_folds']
    default_weight_filenames = {'vgae': [f'k{i}_vgae_weights.pth' for i in range(num_pretrained_models)]}
    weight_filenames = config.get('weight_filenames', default_weight_filenames)
    check_weights_exist(weights_dir, weight_filenames)
    config_updates['weight_filenames'] = weight_filenames
    
    return config_updates

# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Set up environment --------------------------------------------------------
    output_dir = add_project_root(_config['output_dir'])
    verbose = _config['verbose']
    save_weights = _config['save_weights']
    seed = _config['seed']
    logit_C = _config['logit_C']
    n_pca_components = _config['n_pca_components']
    num_folds = _config['dataset']['num_folds']
    weights_dir = add_project_root(_config['weights_dir'])
    weight_filenames = _config['weight_filenames']
    reinit_pooling = _config['reinit_pooling']

    # Create output directories, fix seed
    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    image_files = []  

    # Load data
    data = load_data()
    device = torch.device(_config['device'])
    logger.info(f'Using device: {device}')

    # Load pretrained VGAEs
    if reinit_pooling:
        # Initializes VGAE with new pooling module and doesn't load the weights from the old model, so should be safe to do
        pretrained_vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device=device, exclude_module='pooling')
    else:
        pretrained_vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device=device)  

    # Get test fold indices
    test_indices = np.loadtxt(os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)

    # Train logistic regression models on VGAE latent representations ------------
    start_time = time()

    # Initialize outputs dictionary
    all_outputs = init_outputs_dict(data)
    logit_models = []
    pca_transformers = []
    pca_var_explained = []

    for k in tqdm(range(num_folds), desc='Folds', disable=not verbose):
        # Get pretrained VGAE for this fold
        vgae = pretrained_vgaes[k].to(device)
        vgae = freeze_model(vgae)
        
        # Get train and test data for this fold
        train_data = data[test_indices != k]
        test_data = data[test_indices == k]
        
        # Extract latent representations for train and test sets
        z_train, y_train, clinical_train, subject_train = extract_latent_representations(
            vgae, train_data, device)
        z_test, y_test, clinical_test, subject_test = extract_latent_representations(
            vgae, test_data, device)

        # We actually want to train on [z, clinical_data] -> the MLP input in train_jointly.py
        x_train = np.concatenate([z_train, clinical_train], axis=1)
        x_test = np.concatenate([z_test, clinical_test], axis=1)
        
        # Apply PCA if requested
        pca = None
        if n_pca_components > 0:
            pca = PCA(n_components=n_pca_components, random_state=seed)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
            logger.info(f"Fold {k}: Applied PCA, reduced from {vgae.readout_dim} to {n_pca_components} dimensions")
            var_expl_dict = {f'PC{i+1}': pca.explained_variance_ratio_[i] for i in range(n_pca_components)}
            pca_var_explained.append(var_expl_dict)
        pca_transformers.append(pca)
        
        # Train Logistic regression model
        logit_model = LogisticRegression(C=logit_C, random_state=seed, max_iter=1000)
        logit_model.fit(x_train, y_train)
        logit_models.append(logit_model)
        
        # Make predictions on test set (probabilities)
        y_prob_test = logit_model.predict_proba(x_test)[:, 1]  # Get probability of class 1
        
        # Store outputs
        outputs = {'prediction': y_prob_test, 
                   'label': y_test, 
                   'subject_id': subject_test,
                   'clinical_data': []}
        for sub in range(len(subject_test)):
            outputs['clinical_data'].append(tuple(clinical_test[sub, :]))
        update_best_outputs(all_outputs, outputs)
        
        # Evaluate on test set
        y_pred_test = (y_prob_test > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred_test, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average='binary', zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_prob_test)
        except ValueError:
            # roc_auc_score can fail if y_prob_test has only one class
            roc_auc = np.nan
            logger.warning(f"Fold {k}: roc_auc_score failed, setting to NaN")
            
        # Log metrics for this fold
        ex.log_scalar(f'fold{k}/accuracy', accuracy)
        ex.log_scalar(f'fold{k}/precision', precision)
        ex.log_scalar(f'fold{k}/recall', recall)
        ex.log_scalar(f'fold{k}/f1', f1)
        ex.log_scalar(f'fold{k}/roc_auc', roc_auc)
        
        if verbose:
            logger.info(f"Fold {k}: accuracy={accuracy:.4f}, precision={precision:.4f}, \
                         recall={recall:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}, \
                         logit_C={logit_C:.4f}, n_pca_components={n_pca_components}")
        
        # Save model weights if requested
        if save_weights:
            import pickle
            model_path = os.path.join(output_dir, f'k{k}_logit_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(logit_model, f)
            
            if pca is not None:
                pca_path = os.path.join(output_dir, f'k{k}_pca_transformer.pkl')
                with open(pca_path, 'wb') as f:
                    pickle.dump(pca, f)

    # Print training time
    end_time = time()
    logger.info(f"Training completed after {(end_time-start_time)/60:.2f} minutes.")

    # Save outputs as csv file
    all_outputs_df = pd.DataFrame(all_outputs)
    data_file = os.path.join(output_dir, 'prediction_results.csv')
    all_outputs_df.to_csv(data_file, index=False)

    if len(pca_var_explained) > 0:
        pca_var_explained_df = pd.DataFrame(pca_var_explained)
        pca_var_explained_df.to_csv(os.path.join(output_dir, 'pca_var_explained.csv'), index=False)

    # Evaluate overall results ----------------------------------------------------------
    y_true = all_outputs_df['label'].values
    y_prob = all_outputs_df['prediction'].values
    y_pred = (y_prob > 0.5).astype(int)
    
    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    results = {
        'seed': seed,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)

    # Log final metrics
    for k, v in results.items():
        ex.log_scalar(f'final_prediction/{k}', v)
    logger.info(f"Final results: accuracy={accuracy:.4f}, precision={precision:.4f}, \
                 recall={recall:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}, \
                 logit_C={logit_C:.4f}, n_pca_components={n_pca_components}")
    
    # Permutation test for ROC AUC --------------------------------------------
    n_permutations = _config['n_permutations']  
    rng = np.random.default_rng(seed)
    perm_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        # Shuffle labels (treatment assignment) to simulate random assignment
        y_perm = rng.permutation(y_true)
        try:
            perm_aucs[i] = roc_auc_score(y_perm, y_prob)
        except ValueError:
            # roc_auc_score can fail if y_perm has only one class
            perm_aucs[i] = np.nan

    # Drop any failed permutations (should be rare for balanced labels)
    perm_aucs = perm_aucs[~np.isnan(perm_aucs)]

    # One-sided p-value: probability AUC_null >= AUC_observed
    p_value = (np.sum(perm_aucs >= roc_auc) + 1) / (len(perm_aucs) + 1)

    # Save permutation AUCs
    perm_auc_path = os.path.join(output_dir, 'perm_aucs.csv')
    pd.DataFrame({'perm_auc': perm_aucs}).to_csv(perm_auc_path, index=False)

    # Log p-value
    logger.info(f"Permutation test: observed AUC = {roc_auc:.4f}, "
                f"null mean = {np.mean(perm_aucs):.4f}, "
                f"null sd = {np.std(perm_aucs):.4f}, "
                f"p = {p_value:.4g}, n_perm = {len(perm_aucs)}")

    # Histogram of null AUCs with observed AUC marked
    plt.figure(figsize=(8, 6))
    sns.histplot(perm_aucs, bins=50, kde=True)
    plt.axvline(roc_auc, linestyle='--', linewidth=2,
                color='red', label=f'Observed AUC = {roc_auc:.3f}')
    plt.xlabel('AUC under label permutation')
    plt.ylabel('Frequency')
    plt.title(
        f'Permutation test for propensity AUC\n'
        f'Observed = {roc_auc:.3f}, '
        f'Null mean = {np.mean(perm_aucs):.3f}, '
        f'SD = {np.std(perm_aucs):.3f}, '
        f'p = {p_value:.3g}, '
        f'n_perm = {len(perm_aucs)}'
    )
    plt.legend(loc='upper right')
    plt.tight_layout()
    perm_hist_path = os.path.join(output_dir, 'auc_permutation_hist.png')
    plt.savefig(perm_hist_path)
    image_files.append(perm_hist_path)

    # Classification evaluation plots --------------------------------------------
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

    # 4. Raincloud plot with propensity distributions
    palette = {'escitalopram': ESCIT, 'psilocybin': PSILO}
    symbols = {'escitalopram': 'o', 'psilocybin': 'd'}
    save_path = os.path.join(output_dir, 'propensity_raincloud.png')
    distributions = {
        'escitalopram': all_outputs_df[all_outputs_df['label'] == 0]['prediction'],
        'psilocybin': all_outputs_df[all_outputs_df['label'] == 1]['prediction']}
    plot_raincloud(distributions, 
                    palette=palette, 
                    vline=0.5,
                    box_alpha=0.5,
                    figsize=(4, 5),
                    save_path=save_path)

    # Close all plots
    plt.close('all')

    # Log all images
    for img in image_files:
        if img is not None:
            ex.add_artifact(filename=img)
