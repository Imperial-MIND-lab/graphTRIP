"""
Supplementary: overfitting of graphTRIP compared with a non-graph baseline (FlatVAE + MLP).

Panels, in one figure:
- true-vs-predicted scatter of the baseline, training and test predictions
- the same for graphTRIP
- per-seed gap between training and test correlation, one violin per model, with a
  red asterisk where the two models differ (Wilcoxon signed-rank, or Mann-Whitney U
  when the seed counts differ)

Training predictions are averaged over the folds in which a subject was a training
subject; test predictions are the out-of-fold ones. Scatter points average both across
seeds.

Author: Hanna M. Tolle
Date: 2026-04-12
License: BSD 3-Clause
"""

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu
from torch_geometric.loader import DataLoader

from utils.configs import load_ingredient_configs
from experiments.ingredients.mlp_ingredient import get_x_with_vgae, get_mlp_outputs_nograd

from figure_making.loaders import get_device, load_dataset, load_vgaes, load_mlps
from figure_making.paths import output_dir, require, MissingInput
from figure_making.registry import register


# (display name, results directory parts) of the two models, in panel order
MODELS = [
    ('Non-graph baseline', ('flatvae_mlp',)),
    ('graphTRIP', ('graphtrip', 'weights')),
]

FIGSIZE = (9.5, 3.0)
SUBPLOT_TITLE_FONTSIZE = 10
AXIS_LABEL_FONTSIZE = 9

COLOR_TRAIN = '#377eb8'
COLOR_TEST = '#e41a1c'
COLOR_MODELS = ['#4daf4a', '#984ea3']
VIOLIN_ALPHA = 0.6


# Collecting the runs ---------------------------------------------------------------

def _seed_dirs(base_dir):
    '''The seed_* subdirectories of one model, sorted.'''
    dirs = sorted(d for d in glob.glob(os.path.join(base_dir, 'seed_*')) if os.path.isdir(d))
    if not dirs:
        raise MissingInput(os.path.join(base_dir, 'seed_*'))
    return dirs


def _num_folds(run_dir):
    '''Number of trained folds, from the k{i}_mlp_weights.pth files.'''
    k = 0
    while os.path.exists(os.path.join(run_dir, f'k{k}_mlp_weights.pth')):
        k += 1
    if k == 0:
        raise MissingInput(os.path.join(run_dir, 'k0_mlp_weights.pth'))
    return k


def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_run_data(run_dir, data, config, device):
    '''
    Training and test predictions of one seed.

    Returns a dict with train_df and test_df (subject_id, label, prediction, Condition),
    and the parameter counts of the MLP and the VAE.
    '''
    num_folds = _num_folds(run_dir)
    vgaes = load_vgaes(config['vgae_model'], run_dir,
                       [f'k{k}_vgae_weights.pth' for k in range(num_folds)])
    vgaes = [vgae.to(device) for vgae in vgaes]
    mlps = [mlp.to(device) for mlp in
            load_mlps(config['mlp_model'], vgaes[0].readout_dim, run_dir,
                      [f'k{k}_mlp_weights.pth' for k in range(num_folds)])]

    fold_assignments = np.loadtxt(os.path.join(run_dir, 'test_fold_indices.csv'), dtype=int)

    # Out-of-fold predictions: every subject is in exactly one test fold
    test_df = pd.read_csv(os.path.join(run_dir, 'prediction_results.csv'))
    condition_map = dict(zip(test_df['subject_id'], test_df['Condition']))

    # Training predictions, averaged over the folds a subject was trained on
    pred_accum, label_store = defaultdict(list), {}
    for k in range(num_folds):
        train_indices = np.where(fold_assignments != k)[0]
        loader = DataLoader(data[train_indices], batch_size=len(train_indices), shuffle=False)
        outputs = get_mlp_outputs_nograd(mlps[k], loader, device, get_x=get_x_with_vgae,
                                         vgae=vgaes[k], num_z_samples=0)
        for sid, pred, lbl in zip(outputs['subject_id'], outputs['prediction'],
                                  outputs['label']):
            pred_accum[sid].append(pred)
            label_store[sid] = lbl

    train_df = pd.DataFrame({
        'subject_id': list(pred_accum.keys()),
        'label': [label_store[s] for s in pred_accum],
        'prediction': [np.mean(preds) for preds in pred_accum.values()],
        'Condition': [condition_map.get(s, np.nan) for s in pred_accum],
    })

    return {'train_df': train_df, 'test_df': test_df,
            'mlp_params': _count_params(mlps[0]), 'vae_params': _count_params(vgaes[0])}


def _aggregate(seed_results):
    '''Per-subject mean of the training and of the test predictions across seeds.'''
    agg = []
    for split in ('train_df', 'test_df'):
        frame = pd.concat([r[split] for r in seed_results], ignore_index=True)
        agg.append(frame.groupby('subject_id').mean(numeric_only=True).reset_index())
    return agg


# Statistics -------------------------------------------------------------------------

def _corr_gap(train_df, test_df):
    '''Training r minus test r; nan if either split has fewer than 3 subjects.'''
    if len(train_df) < 3 or len(test_df) < 3:
        return np.nan
    return (pearsonr(train_df['label'], train_df['prediction'])[0]
            - pearsonr(test_df['label'], test_df['prediction'])[0])


def _seed_gaps(seed_results):
    gaps = [_corr_gap(r['train_df'], r['test_df']) for r in seed_results]
    return [g for g in gaps if not np.isnan(g)]


def _gap_test(gaps1, gaps2):
    '''
    Two-sided test that the two models' gaps differ, as (p, test name).

    Paired over seeds when both models have the same number of seeds; p is 1.0 with
    fewer than three seeds.
    '''
    d1, d2 = np.array(gaps1), np.array(gaps2)
    if len(d1) < 3 or len(d2) < 3:
        return 1.0, 'none (too few seeds)'
    if len(d1) == len(d2):
        try:
            return wilcoxon(d1, d2, alternative='two-sided')[1], 'Wilcoxon signed-rank'
        except ValueError:
            pass
    return mannwhitneyu(d1, d2, alternative='two-sided')[1], 'Mann-Whitney U'


# Panels -----------------------------------------------------------------------------

def _scatter_panel(ax, train_df, test_df, target_name, model_name):
    '''True-vs-predicted scatter, coloured by training/test split.'''
    ax.scatter(train_df['label'], train_df['prediction'], color=COLOR_TRAIN,
               edgecolors=COLOR_TRAIN, alpha=0.7, label='train', marker='o', s=15)
    ax.scatter(test_df['label'], test_df['prediction'], color=COLOR_TEST,
               edgecolors=COLOR_TEST, alpha=0.7, label='test', marker='o', s=15)

    all_vals = pd.concat([train_df[['label', 'prediction']],
                          test_df[['label', 'prediction']]])
    lo = min(all_vals.min()) - 2
    hi = max(all_vals.max()) + 2
    ax.plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.6)

    for df, color in [(train_df, COLOR_TRAIN), (test_df, COLOR_TEST)]:
        m, b = np.polyfit(df['label'], df['prediction'], 1)
        x_line = np.array([lo, hi])
        ax.plot(x_line, m * x_line + b, '-', color=color, alpha=0.5, linewidth=1.5)
        r, _ = pearsonr(df['label'], df['prediction'])
        x_mid = (lo + hi) / 2
        ax.text(x_mid, m * x_mid + b, f'r={r:.3f}', color=color,
                fontsize=AXIS_LABEL_FONTSIZE, ha='center', va='bottom')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'True {target_name.split("_")[0]}', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f'Predicted {target_name.split("_")[0]}', fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(model_name, fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.grid(True)


def _gap_panel(ax, gaps, names, p):
    '''One violin of per-seed gaps per model, the seeds on top, and the test result.'''
    for x_pos, (vals, color) in enumerate(zip(gaps, COLOR_MODELS)):
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

    # Local generator, so the jitter does not depend on which targets ran before
    rng = np.random.default_rng(seed=0)
    for x_pos, (vals, color) in enumerate(zip(gaps, COLOR_MODELS)):
        for val in vals:
            ax.scatter(x_pos + rng.uniform(-0.06, 0.06), val, color=color,
                       edgecolors='black', s=20, zorder=5, alpha=0.85, linewidths=0.5)

    all_vals = [v for vals in gaps for v in vals]
    if all_vals:
        y_top, y_bot = max(all_vals), min(all_vals)
        y_range = y_top - y_bot
        ax.set_ylim(y_bot - 0.05 * y_range, y_top + 0.15 * y_range)
        if p < 0.05:
            ax.text(0.5, y_top + 0.04 * y_range, '*', color='red', ha='center',
                    va='bottom', fontsize=13, fontweight='bold',
                    transform=ax.get_yaxis_transform())

    ax.legend(handles=[mpatches.Patch(facecolor=c, alpha=VIOLIN_ALPHA, label=n)
                       for c, n in zip(COLOR_MODELS, names)],
              fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Train r - Test r', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Train-Test Gap', fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, axis='y', alpha=0.5)


@register('overfitting', group='supp', subdir='SUPPLEMENTARY/overfitting')
def overfitting(ctx, out):
    device = get_device()
    names, seed_results, target_name = [], [], None

    for name, parts in MODELS:
        seed_dirs = _seed_dirs(require(output_dir(*parts)))
        config = load_ingredient_configs(seed_dirs[0],
                                         ingredients=['dataset', 'vgae_model', 'mlp_model'])
        target_name = target_name or config['dataset'].get('target', 'target')
        data = load_dataset(config['dataset'])
        names.append(name)
        seed_results.append([collect_run_data(d, data, config, device) for d in seed_dirs])

    gaps = [_seed_gaps(results) for results in seed_results]
    p, test = _gap_test(*gaps)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for ax, name, results in zip(axes[:2], names, seed_results):
        train_df, test_df = _aggregate(results)
        _scatter_panel(ax, train_df, test_df, target_name, name)
    _gap_panel(axes[2], gaps, names, p)
    fig.tight_layout()
    save_path = out.fig('overfitting_comparison')
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)

    rows = [{'model': name, 'num_seeds': len(results),
             'mlp_params': results[0]['mlp_params'], 'vae_params': results[0]['vae_params'],
             'mean_corr_diff': float(np.mean(g)) if g else np.nan,
             'se_corr_diff': float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1
             else np.nan}
            for name, results, g in zip(names, seed_results, gaps)]
    table = pd.DataFrame(rows)
    out.table('overfitting_stats', table)
    out.log_df('Train-test correlation gap per model', table)
    out.log(f'{names[0]} vs {names[1]}: p = {p:.4g} (two-sided {test})')
