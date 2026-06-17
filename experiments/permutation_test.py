"""
Permutation test for model significance.

Uses frozen VGAE weights from the real CV run, shuffles QIDS labels across
subjects, retrains only the MLP on shuffled labels using the same 7-fold splits,
and repeats to build a null distribution of r, R², MAE, MSE, RMSE.

Pre-computes mu / logvar / eval-embeddings once per fold (outside the permutation
loop) to avoid running the Graphormer encoder 4M+ times.

Author: Hanna Tolle
Date: 2026-06-17
License: BSD-3-Clause
"""

import sys
sys.path.append('graphTRIP/')

from sacred import Experiment
from experiments.ingredients.data_ingredient import *
from experiments.ingredients.vgae_ingredient import *
from experiments.ingredients.mlp_ingredient import *

import os
import copy
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from time import time
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.files import add_project_root
from utils.helpers import fix_random_seed, get_logger, check_weights_exist
from utils.configs import load_ingredient_configs, match_ingredient_configs
from models.utils import freeze_model


# Create experiment and logger -------------------------------------------------
ex = Experiment('permutation_test', ingredients=[data_ingredient,
                                                  vgae_ingredient,
                                                  mlp_ingredient])
logger = get_logger()
ex.logger = logger

# Define configurations --------------------------------------------------------
@ex.config
def cfg(dataset):
    exname = 'permutation_test'
    jobid = 0
    seed = 0

    num_permutations = 100   # permutations handled by this Sacred run / HPC job
    perm_offset = jobid * num_permutations  # starting index in the global null distribution

    run_name = f'{exname}_job{jobid}_seed{seed}'
    output_dir = os.path.join('outputs', 'runs', run_name)

    verbose = False
    ex.logger.setLevel(logging.INFO if verbose else logging.ERROR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory with pre-trained VGAE weights (same as weights_dir in retrain_mlp)
    weights_dir = os.path.join('outputs', 'weights')
    weight_filenames = {
        'vgae': [f'k{k}_vgae_weights.pth' for k in range(dataset['num_folds'])],
        'test_fold_indices': ['test_fold_indices.csv'],
    }

    mlp_num_epochs = 300   # epochs per permuted MLP; reduce for speed (e.g. 100)
    mlp_lr = 0.001


# Match configs function -------------------------------------------------------
def match_config(config):
    '''Matches VGAE and dataset configs against the pre-trained weights directory.'''
    assert 'weights_dir' in config, "weights_dir must be specified in config."
    weights_dir = add_project_root(config['weights_dir'])

    ingredients = ['vgae_model', 'dataset']
    previous_config = load_ingredient_configs(weights_dir, ingredients)

    exceptions = ['target', 'graph_attrs', 'graph_attrs_to_standardise', 'num_graph_attr']
    config_updates = match_ingredient_configs(
        config=config,
        previous_config=previous_config,
        ingredients=ingredients,
        exceptions=exceptions,
    )

    if 'standardise_x' in previous_config.get('dataset', {}):
        assert not config_updates['dataset']['standardise_x'], \
            "Standardisation of x in pretrained models is not supported."

    return config_updates


# Main function ----------------------------------------------------------------
@ex.automain
def run(_config):

    # Setup -------------------------------------------------------------------
    output_dir   = add_project_root(_config['output_dir'])
    seed         = _config['seed']
    device       = torch.device(_config['device'])
    num_folds    = _config['dataset']['num_folds']
    batch_size   = _config['dataset']['batch_size']
    weights_dir  = add_project_root(_config['weights_dir'])
    weight_filenames  = _config['weight_filenames']
    num_permutations  = _config['num_permutations']
    perm_offset       = _config['perm_offset']
    mlp_num_epochs    = _config['mlp_num_epochs']
    mlp_lr            = _config['mlp_lr']

    os.makedirs(output_dir, exist_ok=True)
    fix_random_seed(seed)
    check_weights_exist(weights_dir, weight_filenames)
    logger.info(f'Using device: {device}')

    # Load data and fold splits ------------------------------------------------
    data = load_data()
    n_total = len(data)

    test_indices = np.loadtxt(
        os.path.join(weights_dir, weight_filenames['test_fold_indices'][0]), dtype=int)
    train_loaders, val_loaders, test_loaders, mean_std = \
        get_dataloaders_from_test_indices(data, test_indices, seed=seed)

    # Ground-truth y in dataset order (not standardised — only node/graph attrs are)
    y_global = torch.stack([data[i].y for i in range(n_total)]).squeeze().float()  # [n_total]

    # Load pretrained VGAEs (frozen) ------------------------------------------
    vgaes = load_trained_vgaes(weights_dir, weight_filenames['vgae'], device=device)
    for vgae in vgaes:
        vgae.eval()
        freeze_model(vgae)

    readout_dim = vgaes[0].readout_dim

    # Pre-computation (one VGAE forward pass per fold) ------------------------
    # For each fold k: cache mu, logvar, readout(mu) for ALL n_total subjects
    # in global dataset order, using fold k's VGAE and fold-specific transforms.
    logger.info("Pre-computing VGAE encodings per fold...")
    start_precomp = time()

    latent_dim = _config['vgae_model']['params']['latent_dim']
    N_nodes = data[0].num_nodes  # fixed for schaefer atlas

    precomputed = {}  # k → dict of pre-computed tensors (global dataset order)

    for k in range(num_folds):
        vgae_k = vgaes[k].to(device)

        # Global dataset indices for this fold's train / test subjects
        train_idx_k = np.where(test_indices != k)[0]  # sorted ascending
        test_idx_k  = np.where(test_indices == k)[0]

        # Allocate storage in global order
        mu_k      = torch.zeros(n_total, N_nodes, latent_dim)
        logvar_k  = torch.zeros(n_total, N_nodes, latent_dim)
        context_k = None  # filled on first iteration
        x_eval_k  = torch.zeros(n_total, readout_dim + len(_config['dataset']['graph_attrs']))

        for subset, global_idx in [
            (train_loaders[k].dataset, train_idx_k),
            (test_loaders[k].dataset,  test_idx_k),
        ]:
            # Single pass over this subset in fixed order (shuffle=False)
            loader = PyGDataLoader(subset, batch_size=len(subset), shuffle=False)
            batch  = next(iter(loader)).to(device)
            n_subj = batch.num_graphs

            assert batch.num_nodes == n_subj * N_nodes, \
                f"Expected {N_nodes} nodes per subject, got {batch.num_nodes // n_subj}"

            with torch.no_grad():
                out     = vgae_k(batch)
                ctx     = get_context(batch)               # [n_subj * N_nodes, ctx_dim]
                h_eval  = vgae_k.readout(out.mu, ctx, batch.batch)  # [n_subj, readout_dim]
                x_eval  = torch.cat([h_eval, batch.graph_attr], dim=1)  # [n_subj, readout_dim+n_clin]

            ctx_dim = ctx.shape[1]
            if context_k is None:
                context_k = torch.zeros(n_total, N_nodes, ctx_dim)

            mu_k[global_idx]      = out.mu.view(n_subj, N_nodes, latent_dim).cpu()
            logvar_k[global_idx]  = out.logvar.view(n_subj, N_nodes, latent_dim).cpu()
            context_k[global_idx] = ctx.view(n_subj, N_nodes, ctx_dim).cpu()
            x_eval_k[global_idx]  = x_eval.cpu()

        if context_k is None:
            context_k = torch.zeros(n_total, N_nodes, 0)

        precomputed[k] = {
            'mu':      mu_k,       # [n_total, N_nodes, latent_dim]
            'logvar':  logvar_k,   # [n_total, N_nodes, latent_dim]
            'context': context_k,  # [n_total, N_nodes, ctx_dim]
            'x_eval':  x_eval_k,  # [n_total, readout_dim + n_clinical]
            'train_idx': train_idx_k,
            'test_idx':  test_idx_k,
        }

    elapsed_precomp = time() - start_precomp
    logger.info(f"Pre-computation done in {elapsed_precomp:.1f}s.")

    # Permutation loop --------------------------------------------------------
    rng = np.random.default_rng(seed + perm_offset)
    null_rows = []
    ctx_dim = precomputed[0]['context'].shape[2]

    for perm_i in tqdm(range(num_permutations), desc='Permutations'):
        global_perm = rng.permutation(n_total)
        # subject at position i gets the label originally belonging to position global_perm[i]
        y_perm = y_global[global_perm]  # [n_total]

        all_preds = []
        all_true  = []

        for k in range(num_folds):
            vgae_k      = vgaes[k].to(device)
            pc          = precomputed[k]
            train_idx_k = pc['train_idx']
            test_idx_k  = pc['test_idx']
            n_train     = len(train_idx_k)

            # Permuted labels for this fold
            y_train_perm = y_perm[train_idx_k].to(device)  # [n_train]
            y_test_perm  = y_perm[test_idx_k]               # [n_test]  (stays on CPU)

            # Pre-computed train tensors (kept on CPU, moved to device per mini-batch)
            mu_train      = pc['mu'][train_idx_k]       # [n_train, N_nodes, latent_dim]
            logvar_train  = pc['logvar'][train_idx_k]
            context_train = pc['context'][train_idx_k]  # [n_train, N_nodes, ctx_dim]
            clinical_train = pc['x_eval'][train_idx_k, readout_dim:]  # [n_train, n_clinical]

            # Pre-computed eval inputs for test subjects
            x_eval_test = pc['x_eval'][test_idx_k].to(device)  # [n_test, readout_dim+n_clinical]

            # Fresh MLP
            mlp       = build_mlp(latent_dim=readout_dim).to(device)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=mlp_lr)

            # -- Training loop with stochastic z (replicates num_z_samples=1) --
            for epoch in range(mlp_num_epochs):
                mlp.train()
                perm_order = torch.randperm(n_train)
                for b_start in range(0, n_train, batch_size):
                    b_idx = perm_order[b_start: b_start + batch_size]
                    n_b   = len(b_idx)

                    # Move mini-batch to device
                    mu_b      = mu_train[b_idx].view(-1, latent_dim).to(device)
                    logvar_b  = logvar_train[b_idx].view(-1, latent_dim).to(device)
                    # ctx_dim==0 means no context; view(-1, 0) is ambiguous for zero-element tensors
                    if ctx_dim == 0:
                        context_b = torch.zeros(n_b * N_nodes, 0, device=device)
                    else:
                        context_b = context_train[b_idx].view(-1, ctx_dim).to(device)
                    clinical_b = clinical_train[b_idx].to(device)
                    y_b       = y_train_perm[b_idx].view(-1, 1)

                    # Batch index for pooling: each of the n_b subjects has N_nodes nodes
                    batch_idx_b = torch.arange(n_b, device=device).repeat_interleave(N_nodes)

                    # Cheap stochastic z sample (reparameterize, no Graphormer call)
                    with torch.no_grad():
                        z_b = vgae_k.reparameterize(mu_b, logvar_b)
                        h_b = vgae_k.readout(z_b, context_b, batch_idx_b)  # [n_b, readout_dim]

                    x_b = torch.cat([h_b, clinical_b], dim=1)  # [n_b, readout_dim+n_clinical]

                    optimizer.zero_grad()
                    y_pred_b = mlp(x_b)
                    loss = mlp.loss(y_pred_b, y_b)
                    loss.backward()
                    optimizer.step()

            # -- Evaluation on pre-computed mu-based embeddings ----------------
            mlp.eval()
            with torch.no_grad():
                y_pred_test = mlp(x_eval_test).squeeze().cpu()

            all_preds.append(y_pred_test)
            all_true.append(y_test_perm.cpu())

        # Aggregate across folds and compute null metrics
        y_pred_all = torch.cat(all_preds).numpy()
        y_true_all = torch.cat(all_true).numpy()

        r, _  = pearsonr(y_true_all, y_pred_all)
        r2    = r2_score(y_true_all, y_pred_all)
        mae   = mean_absolute_error(y_true_all, y_pred_all)
        mse   = mean_squared_error(y_true_all, y_pred_all)
        rmse  = float(np.sqrt(mse))

        null_rows.append({
            'perm_idx': int(perm_offset + perm_i),
            'r': float(r), 'r2': float(r2),
            'mae': float(mae), 'mse': float(mse), 'rmse': rmse,
        })

        ex.log_scalar('null/r',   r)
        ex.log_scalar('null/r2',  r2)
        ex.log_scalar('null/mae', mae)

    # Save null distribution --------------------------------------------------
    null_df = pd.DataFrame(null_rows)
    csv_path = os.path.join(output_dir, 'null_distribution.csv')
    null_df.to_csv(csv_path, index=False)
    ex.add_artifact(csv_path)

    logger.info(
        f"Null distribution ({len(null_rows)} permutations) saved to {csv_path}.\n"
        f"  r range: [{null_df['r'].min():.4f}, {null_df['r'].max():.4f}]  "
        f"  mean r = {null_df['r'].mean():.4f}"
    )
