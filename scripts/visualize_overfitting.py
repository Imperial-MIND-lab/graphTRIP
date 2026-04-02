"""
Visualizes overfitting behaviour for a single-fold (num_folds_to_run=1) train_jointly run.
Produces a single figure with two panels: MLP loss curves (left) and true-vs-predicted
scatter coloured by train/test split (right).

Usage (from project root):
    python scripts/visualize_overfitting.py <run_dir> [--output_dir <dir>] [--fmt <ext>]

Example:
    python scripts/visualize_overfitting.py outputs/flatvae_mlp/job_0/seed_1 \
        --output_dir outputs/overfitting/flatvae --fmt png

Author: Hanna M. Tolle
Date: 2026-03-29
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')

import sys
import os
# Ensure project root is on the path regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from utils.configs import load_ingredient_configs, fill_missing_configs
from models.utils import init_model
from experiments.ingredients.data_ingredient import load_dataset_from_configs
from experiments.ingredients.mlp_ingredient import get_x_with_vgae, get_mlp_outputs_nograd


# ── Figure constants ──────────────────────────────────────────────────────────

FIGSIZE = (6.5, 2.75)            # (width, height) of the combined figure in inches
SUBPLOT_TITLE_FONTSIZE = 10  # font size for panel titles ("MLP loss", "MLP Predictions")
AXIS_LABEL_FONTSIZE = 9      # font size for axis labels and legend text


# ── Model loading helpers ─────────────────────────────────────────────────────

def _init_vgae(config: dict):
    combined_params = {
        'params': config['params'],
        'node_emb_model_cfg': config['node_emb_model_cfg'],
        'pooling_cfg': config['pooling_cfg'],
        'encoder_cfg': config['encoder_cfg'],
        'node_decoder_cfg': config['node_decoder_cfg'],
        'edge_decoder_cfg': config['edge_decoder_cfg'],
    }
    if config['model_type'] != 'GraphLevelVGAE':
        combined_params['edge_idx_decoder_cfg'] = config['edge_idx_decoder_cfg']
    return init_model(config['model_type'], combined_params)


def _load_vgae(config: dict, weights_path: str, device):
    vgae = _init_vgae(config)
    vgae.load_state_dict(torch.load(weights_path, map_location=device))
    vgae.to(device)
    return vgae


def _init_mlp(config: dict, latent_dim: int):
    params = copy.deepcopy(config.get('params', {}))
    extra_dim = config.get('extra_dim', None) or len(config['dataset']['graph_attrs'])
    hidden_dim = params.get('hidden_dim', None) or max(latent_dim, extra_dim)
    params['input_dim'] = latent_dim + extra_dim
    params['hidden_dim'] = hidden_dim
    return init_model(config['model_type'], params)


def _load_mlp(config: dict, latent_dim: int, weights_path: str, device):
    mlp = _init_mlp(config, latent_dim)
    mlp.load_state_dict(torch.load(weights_path, map_location=device))
    mlp.to(device)
    return mlp


def _load_dataset(config: dict):
    import importlib
    data_ingredient_mod = importlib.import_module('experiments.ingredients.data_ingredient')
    default_config = data_ingredient_mod.data_cfg()
    config = fill_missing_configs(config, reference_config=default_config)
    return load_dataset_from_configs(config)


# ── Utilities ────────────────────────────────────────────────────────────────

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_overfitting_figure(train_mlp_loss, test_mlp_loss,
                             train_df, test_df,
                             target_name: str,
                             save_path: str,
                             log_every: int = 1):
    """Single figure with two panels: MLP loss curves (left) and scatter (right)."""
    _, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    # ── Left: MLP loss curves ─────────────────────────────────────────────────
    ax = axes[0]
    epochs = range(0, len(train_mlp_loss), log_every)
    ax.plot(epochs, train_mlp_loss[::log_every], color='blue', label='train', linewidth=2)
    ax.plot(epochs, test_mlp_loss[::log_every],  color='red',  label='test', linewidth=2)
    ax.set_title('MLP loss', fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_xlabel('Epoch',   fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Loss',    fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True)

    # ── Right: true-vs-predicted scatter ─────────────────────────────────────
    ax = axes[1]
    ax.scatter(train_df['label'], train_df['prediction'],
               color='blue', edgecolors='blue', alpha=0.7, label='train', marker='o', s=15)
    ax.scatter(test_df['label'], test_df['prediction'],
               color='red',  edgecolors='red',  alpha=0.7, label='test',  marker='o', s=15)

    all_vals = pd.concat([train_df[['label', 'prediction']], test_df[['label', 'prediction']]])
    lo = min(all_vals.min()) - 2
    hi = max(all_vals.max()) + 2
    ax.plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.6)

    # Regression lines + r annotations
    for df, color in [(train_df, 'blue'), (test_df, 'red')]:
        m, b = np.polyfit(df['label'], df['prediction'], 1)
        x_line = np.array([lo, hi])
        ax.plot(x_line, m * x_line + b, '-', color=color, alpha=0.5, linewidth=1.5)
        r, _ = pearsonr(df['label'], df['prediction'])
        # Place annotation near the regression line midpoint
        x_mid = (lo + hi) / 2
        y_mid = m * x_mid + b
        ax.text(x_mid, y_mid, f'r={r:.3f}', color=color,
                fontsize=AXIS_LABEL_FONTSIZE, ha='center', va='bottom')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'True {target_name.split("_")[0]}',      fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f'Predicted {target_name.split("_")[0]}', fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('MLP Predictions', fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(run_dir: str, output_dir: str, fmt: str, log_every: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. Load config ────────────────────────────────────────────────────────
    print("Loading configs...")
    config = load_ingredient_configs(run_dir, ingredients=['dataset', 'vgae_model', 'mlp_model'])
    target_name = config['dataset'].get('target', 'target')

    # ── 2. Load models and dataset ────────────────────────────────────────────
    print("Loading dataset...")
    data = _load_dataset(config['dataset'])

    print("Loading VGAE...")
    vgae = _load_vgae(config['vgae_model'],
                      os.path.join(run_dir, 'k0_vgae_weights.pth'),
                      device)

    print("Loading MLP...")
    mlp = _load_mlp(config['mlp_model'], vgae.readout_dim,
                    os.path.join(run_dir, 'k0_mlp_weights.pth'),
                    device)

    # ── 3. Load fold assignments ──────────────────────────────────────────────
    fold_assignments = np.loadtxt(os.path.join(run_dir, 'test_fold_indices.csv'), dtype=int)
    test_indices  = np.where(fold_assignments == 0)[0]
    train_indices = np.where(fold_assignments != 0)[0]
    print(f"Subjects — train: {len(train_indices)}, test: {len(test_indices)}")

    # ── 4. Load MLP loss curves ───────────────────────────────────────────────
    print("Loading metrics...")
    with open(os.path.join(run_dir, 'metrics.json')) as f:
        metrics = json.load(f)
    train_mlp_loss = np.array(metrics['training/fold0/epoch/mlp_loss']['values'])
    test_mlp_loss  = np.array(metrics['test/fold0/epoch/mlp_loss']['values'])

    # ── 5. Predictions ────────────────────────────────────────────────────────
    # Test predictions already saved by train_jointly
    test_df = pd.read_csv(os.path.join(run_dir, 'prediction_results.csv'))

    # Train predictions: forward pass through VGAE + MLP
    print("Computing train predictions...")
    train_dataset = data[train_indices]
    train_loader  = DataLoader(train_dataset, batch_size=len(train_indices), shuffle=False)
    train_outputs = get_mlp_outputs_nograd(mlp, train_loader, device,
                                           get_x=get_x_with_vgae,
                                           vgae=vgae, num_z_samples=0)
    train_df = pd.DataFrame({
        'label':      train_outputs['label'],
        'prediction': train_outputs['prediction'],
        'subject_id': train_outputs['subject_id'],
    })

    # ── 6. Save stats CSV ─────────────────────────────────────────────────────
    train_corr, train_p = pearsonr(train_df['label'], train_df['prediction'])
    test_corr,  test_p  = pearsonr(test_df['label'],  test_df['prediction'])
    mlp_params  = _count_params(mlp)
    vae_params  = _count_params(vgae)
    stats_df = pd.DataFrame([{
        'mlp_param_count':   mlp_params,
        'vae_param_count':   vae_params,
        'total_param_count': mlp_params + vae_params,
        'train_corr': train_corr,
        'train_p':    train_p,
        'test_corr':  test_corr,
        'test_p':     test_p,
    }])
    stats_df.to_csv(os.path.join(output_dir, 'stats.csv'), index=False)

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, f'overfitting.{fmt}')
    print(f"Saving figure to {save_path}...")
    _plot_overfitting_figure(
        train_mlp_loss, test_mlp_loss,
        train_df, test_df,
        target_name=target_name,
        save_path=save_path,
        log_every=log_every,
    )
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize overfitting for a single-fold train_jointly run.')
    parser.add_argument('run_dir', type=str,
                        help='Directory with run outputs (e.g. outputs/flatvae_mlp/job_0/seed_1)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save the figure (default: <run_dir>/overfitting_plots/)')
    parser.add_argument('--fmt', type=str, default='png',
                        help='Plot file format (default: png)')
    parser.add_argument('--log_every', type=int, default=1,
                        help='Plot every N-th epoch in loss curves (default: 1)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.run_dir, 'overfitting_plots')
    main(args.run_dir, output_dir, args.fmt, log_every=args.log_every)
