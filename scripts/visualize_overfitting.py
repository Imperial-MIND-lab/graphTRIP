"""
Compares overfitting behaviour between two models (e.g. FlatVAE baseline vs graphTRIP),
each run with multiple seeds.

Produces a single figure with three panels:
  - True-vs-predicted scatter for model 1 (left)
  - True-vs-predicted scatter for model 2 (center)
  - Split violin comparing the train-test Pearson r gap (right):
      y-axis: train_r - test_r (per seed)
      x-axis: three condition subsets — Both, Escitalopram (Condition=-1), Psilocybin (Condition=1)
      A red asterisk (*) is shown if a Wilcoxon signed-rank test (or Mann-Whitney U when
      seed counts differ) finds p < 0.05 between the two models.

Scatter predictions are averaged per subject across seeds and folds.

Usage (from project root):
    python scripts/visualize_overfitting.py \
        --model1_dir outputs/flatvae_mlp \
        --model2_dir outputs/graphtrip \
        [--model1_name baseline] [--model2_name graphTRIP] \
        [--output_dir outputs/comparison] [--fmt png]

Author: Hanna M. Tolle
Date: 2026-04-12
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')

import sys
import os
# Ensure project root is on the path regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu
from torch_geometric.loader import DataLoader

from utils.configs import load_ingredient_configs, fill_missing_configs
from models.utils import init_model
from experiments.ingredients.data_ingredient import load_dataset_from_configs
from experiments.ingredients.mlp_ingredient import get_x_with_vgae, get_mlp_outputs_nograd


# ── Figure constants ──────────────────────────────────────────────────────────

FIGSIZE = (9.5, 3.0)         # (width, height) of the combined figure in inches
SUBPLOT_TITLE_FONTSIZE = 10  # font size for panel titles
AXIS_LABEL_FONTSIZE = 9      # font size for axis labels and legend text

# ── Color palette ─────────────────────────────────────────────────────────────

COLOR_TRAIN     = '#377eb8'  # blue  — train scatter points
COLOR_TEST      = '#e41a1c'  # red   — test scatter points
COLOR_MODEL1    = '#4daf4a'  # green — violin for model 1
COLOR_MODEL2    = '#984ea3'  # purple — violin for model 2
VIOLIN_ALPHA    = 0.6        # transparency of violin bodies


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


# ── Utilities ─────────────────────────────────────────────────────────────────

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _detect_num_folds(run_dir: str) -> int:
    """Count trained folds by checking for k{i}_mlp_weights.pth files."""
    k = 0
    while os.path.exists(os.path.join(run_dir, f'k{k}_mlp_weights.pth')):
        k += 1
    if k == 0:
        raise FileNotFoundError(f"No fold weight files found in {run_dir}")
    return k


def _find_seed_dirs(parent_dir: str) -> list:
    """Find all seed_* subdirectories in parent_dir, sorted."""
    pattern = os.path.join(parent_dir, 'seed_*')
    dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not dirs:
        raise FileNotFoundError(f"No seed_* subdirectories found in {parent_dir}")
    return dirs


# ── Correlation helpers ───────────────────────────────────────────────────────

def _corr_diff(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Compute train_r - test_r across all subjects.

    Returns np.nan if either split has fewer than 3 samples.
    """
    if len(train_df) < 3 or len(test_df) < 3:
        return np.nan
    train_r, _ = pearsonr(train_df['label'], train_df['prediction'])
    test_r,  _ = pearsonr(test_df['label'],  test_df['prediction'])
    return train_r - test_r


# ── Per-run data collection ───────────────────────────────────────────────────

def collect_run_data(run_dir: str, data, config: dict, device) -> dict:
    """Collect predictions for one seed directory.

    Returns a dict with:
        train_df:   pd.DataFrame  subject_id, label, prediction, Condition
                    (fold-mean predictions per subject)
        test_df:    pd.DataFrame  subject_id, label, prediction, Condition
                    (from prediction_results.csv — one row per subject)
        mlp_params: int
        vae_params: int
    """
    num_folds = _detect_num_folds(run_dir)
    print(f"  {run_dir}: {num_folds} fold(s)")

    # Load fold models
    vgaes, mlps = [], []
    for k in range(num_folds):
        vgae = _load_vgae(config['vgae_model'],
                          os.path.join(run_dir, f'k{k}_vgae_weights.pth'),
                          device)
        mlp  = _load_mlp(config['mlp_model'], vgae.readout_dim,
                         os.path.join(run_dir, f'k{k}_mlp_weights.pth'),
                         device)
        vgaes.append(vgae)
        mlps.append(mlp)

    # Load fold assignments for this seed
    fold_assignments = np.loadtxt(
        os.path.join(run_dir, 'test_fold_indices.csv'), dtype=int)

    # Test predictions (covers all subjects via k-fold — one test fold each)
    test_df = pd.read_csv(os.path.join(run_dir, 'prediction_results.csv'))

    # Build Condition lookup from test_df (all subjects appear exactly once)
    condition_map = dict(zip(test_df['subject_id'], test_df['Condition']))

    # Train predictions: average per subject over folds where they were in train
    pred_accum  = defaultdict(list)
    label_store = {}
    for k in range(num_folds):
        train_indices = np.where(fold_assignments != k)[0]
        n_train = len(train_indices)
        train_loader = DataLoader(data[train_indices], batch_size=n_train, shuffle=False)
        outputs = get_mlp_outputs_nograd(mlps[k], train_loader, device,
                                         get_x=get_x_with_vgae,
                                         vgae=vgaes[k], num_z_samples=0)
        for sid, pred, lbl in zip(outputs['subject_id'],
                                  outputs['prediction'],
                                  outputs['label']):
            pred_accum[sid].append(pred)
            label_store[sid] = lbl

    train_df = pd.DataFrame({
        'subject_id': list(pred_accum.keys()),
        'label':      [label_store[s] for s in pred_accum],
        'prediction': [np.mean(preds) for preds in pred_accum.values()],
        'Condition':  [condition_map.get(s, np.nan) for s in pred_accum],
    })

    return {
        'train_df':   train_df,
        'test_df':    test_df,
        'mlp_params': _count_params(mlps[0]),
        'vae_params': _count_params(vgaes[0]),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_scatter_panel(ax, train_df: pd.DataFrame, test_df: pd.DataFrame,
                        target_name: str, model_name: str):
    """True-vs-predicted scatter on ax, coloured by train/test split."""
    ax.scatter(train_df['label'], train_df['prediction'],
               color=COLOR_TRAIN, edgecolors=COLOR_TRAIN, alpha=0.7, label='train', marker='o', s=15)
    ax.scatter(test_df['label'], test_df['prediction'],
               color=COLOR_TEST,  edgecolors=COLOR_TEST,  alpha=0.7, label='test',  marker='o', s=15)

    all_vals = pd.concat([train_df[['label', 'prediction']], test_df[['label', 'prediction']]])
    lo = min(all_vals.min()) - 2
    hi = max(all_vals.max()) + 2
    ax.plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.6)

    for df, color in [(train_df, COLOR_TRAIN), (test_df, COLOR_TEST)]:
        m, b = np.polyfit(df['label'], df['prediction'], 1)
        x_line = np.array([lo, hi])
        ax.plot(x_line, m * x_line + b, '-', color=color, alpha=0.5, linewidth=1.5)
        r, _ = pearsonr(df['label'], df['prediction'])
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
    ax.set_title(model_name, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.grid(True)


def _plot_violin_panel(ax, seed_results1: list, seed_results2: list,
                       model1_name: str, model2_name: str):
    """Two violins of per-seed (train_r - test_r), one per model."""
    colors = [COLOR_MODEL1, COLOR_MODEL2]
    model_names = [model1_name, model2_name]
    all_seed_results = [seed_results1, seed_results2]

    # Collect per-seed values and draw violins via matplotlib
    data_per_model = []
    for seed_results in all_seed_results:
        vals = [_corr_diff(r['train_df'], r['test_df']) for r in seed_results]
        data_per_model.append([v for v in vals if not np.isnan(v)])

    for x_pos, (vals, color) in enumerate(zip(data_per_model, colors)):
        if len(vals) < 2:
            continue
        parts = ax.violinplot(vals, positions=[x_pos], widths=0.6,
                              showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(VIOLIN_ALPHA)
        parts['cmedians'].set_color(color)
        parts['cmedians'].set_linewidth(2)

    # Scatter points on top (one per seed, with jitter)
    rng = np.random.default_rng(seed=0)
    for x_pos, (seed_results, color) in enumerate(zip(all_seed_results, colors)):
        for r in seed_results:
            val = _corr_diff(r['train_df'], r['test_df'])
            if np.isnan(val):
                continue
            jitter = rng.uniform(-0.06, 0.06)
            ax.scatter(x_pos + jitter, val,
                       color=color, edgecolors='black',
                       s=20, zorder=5, alpha=0.85, linewidths=0.5)

    # Non-parametric test — red asterisk if p < 0.05
    d1 = np.array(data_per_model[0])
    d2 = np.array(data_per_model[1])
    p = 1.0
    if len(d1) >= 3 and len(d2) >= 3:
        if len(d1) == len(d2):
            try:
                _, p = wilcoxon(d1, d2, alternative='two-sided')
            except ValueError:
                _, p = mannwhitneyu(d1, d2, alternative='two-sided')
        else:
            _, p = mannwhitneyu(d1, d2, alternative='two-sided')

    all_vals = d1.tolist() + d2.tolist()
    if all_vals:
        y_top = max(all_vals)
        y_bot = min(all_vals)
        y_range = y_top - y_bot
        ax.set_ylim(y_bot - 0.05 * y_range, y_top + 0.15 * y_range)

    if p < 0.05 and all_vals:
        asterisk_y = y_top + 0.04 * y_range
        # x=0.5 in axes-fraction coordinates, y in data coordinates
        ax.text(0.5, asterisk_y, '*', color='red',
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                transform=ax.get_yaxis_transform())

    # Legend patches
    patches = [mpatches.Patch(facecolor=c, alpha=VIOLIN_ALPHA, label=n)
               for c, n in zip(colors, model_names)]
    ax.legend(handles=patches, fontsize=AXIS_LABEL_FONTSIZE)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(model_names, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel('',               fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Train r - Test r', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Train-Test Gap',  fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, axis='y', alpha=0.5)


def _plot_comparison_figure(agg_train_df1, agg_test_df1,
                             agg_train_df2, agg_test_df2,
                             seed_results1, seed_results2,
                             model1_name: str, model2_name: str,
                             target_name: str, save_path: str):
    """Three-panel figure: scatter model1 | scatter model2 | split violin."""
    _, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    _plot_scatter_panel(axes[0], agg_train_df1, agg_test_df1, target_name, model1_name)
    _plot_scatter_panel(axes[1], agg_train_df2, agg_test_df2, target_name, model2_name)
    _plot_violin_panel(axes[2], seed_results1, seed_results2, model1_name, model2_name)

    plt.tight_layout()
    plt.savefig(save_path, format=save_path.split('.')[-1])
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def _aggregate_scatter(seed_results: list):
    """Average per-subject predictions and labels across seeds."""
    all_train = pd.concat([r['train_df'] for r in seed_results], ignore_index=True)
    agg_train = (all_train.groupby('subject_id')
                 .mean(numeric_only=True)
                 .reset_index())

    all_test = pd.concat([r['test_df'] for r in seed_results], ignore_index=True)
    agg_test = (all_test.groupby('subject_id')
                .mean(numeric_only=True)
                .reset_index())

    return agg_train, agg_test


def main(model1_dir: str, model2_dir: str,
         model1_name: str, model2_name: str,
         output_dir: str, fmt: str):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find seed directories
    seed_dirs1 = _find_seed_dirs(model1_dir)
    seed_dirs2 = _find_seed_dirs(model2_dir)
    print(f"{model1_name}: {len(seed_dirs1)} seed dir(s) in {model1_dir}")
    print(f"{model2_name}: {len(seed_dirs2)} seed dir(s) in {model2_dir}")

    # Load configs (one per model, from their respective first seed)
    print("Loading configs...")
    config1 = load_ingredient_configs(seed_dirs1[0],
                                      ingredients=['dataset', 'vgae_model', 'mlp_model'])
    config2 = load_ingredient_configs(seed_dirs2[0],
                                      ingredients=['dataset', 'vgae_model', 'mlp_model'])
    target_name = config1['dataset'].get('target', 'target')

    # Load datasets
    print("Loading datasets...")
    data1 = _load_dataset(config1['dataset'])
    data2 = _load_dataset(config2['dataset'])

    # Collect per-seed data
    print(f"Collecting data for {model1_name}...")
    seed_results1 = [collect_run_data(sd, data1, config1, device) for sd in seed_dirs1]

    print(f"Collecting data for {model2_name}...")
    seed_results2 = [collect_run_data(sd, data2, config2, device) for sd in seed_dirs2]

    # Aggregate scatter predictions (per-subject mean across seeds)
    agg_train1, agg_test1 = _aggregate_scatter(seed_results1)
    agg_train2, agg_test2 = _aggregate_scatter(seed_results2)

    # ── Stats CSV ─────────────────────────────────────────────────────────────
    rows = []
    for model_name, seed_results in [(model1_name, seed_results1),
                                     (model2_name, seed_results2)]:
        vals = [_corr_diff(r['train_df'], r['test_df']) for r in seed_results]
        vals = [v for v in vals if not np.isnan(v)]
        n = len(vals)
        rows.append({
            'model':           model_name,
            'num_seeds':       len(seed_results),
            'mlp_params':      seed_results[0]['mlp_params'],
            'vae_params':      seed_results[0]['vae_params'],
            'mean_corr_diff':  float(np.mean(vals)) if n > 0 else np.nan,
            'se_corr_diff':    float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        })

    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'stats.csv'), index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, f'comparison.{fmt}')
    print(f"Saving figure to {save_path}...")
    _plot_comparison_figure(
        agg_train1, agg_test1,
        agg_train2, agg_test2,
        seed_results1, seed_results2,
        model1_name=model1_name,
        model2_name=model2_name,
        target_name=target_name,
        save_path=save_path,
    )
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare overfitting between two multi-seed model runs.')

    parser.add_argument('--model1_dir', type=str, required=True,
                        help='Parent directory (with seed_* subdirs) for model 1')
    parser.add_argument('--model2_dir', type=str, required=True,
                        help='Parent directory (with seed_* subdirs) for model 2')
    parser.add_argument('--model1_name', type=str, default='baseline',
                        help='Display name for model 1 (default: baseline)')
    parser.add_argument('--model2_name', type=str, default='graphTRIP',
                        help='Display name for model 2 (default: graphTRIP)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save outputs (default: <model1_dir>/comparison/)')
    parser.add_argument('--fmt', type=str, default='png',
                        help='Plot file format (default: png)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model1_dir, 'comparison')
    main(args.model1_dir, args.model2_dir,
         args.model1_name, args.model2_name,
         output_dir, args.fmt)
