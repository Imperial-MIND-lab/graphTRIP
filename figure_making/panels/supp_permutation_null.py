"""
Supplementary: the permutation null distribution of prediction performance.

Every model is retrained on outcomes permuted across the whole cohort, 100 permutations x
10 training seeds each (scripts/permutation_null.py, scripts/train_selser.py). Permuting the
raw outcome before any pipeline step makes this a null for the entire pipeline -- splitting,
scaling, VGAE fitting and prediction -- rather than for the final correlation alone. All
models share the same 100 perm_seeds, so their nulls are paired.

Two levels are read off the same runs:

    ensemble level  the 10 same-permutation models' predictions are averaged before the
                    metric is computed, matching how the reported metric is built. One
                    draw per permutation, so a hundred draws.
    seed level      one value per run, so a thousand. Its unit of analysis is the training
                    seed rather than the subject, which makes it a statement about
                    leakage and not about generalisation.

The panels answer two questions. Whether the null is centred on zero, which is what a
pipeline free of leakage gives; and whether its width matches the parametric r = 0 null
that p-values on a correlation assume.

The graphTRIP models share a grid per panel, so their nulls can be read against each
other. SELSER is a different model family fitted by a different script, so it gets one
overview figure carrying all three panels instead of a column in each grid.

For graphTRIP the same null weights are also evaluated zero-shot on Schaefer 200, AAL and
psilodep1, giving permutation nulls for the transfer claims at no extra training cost.

Author: Hanna M. Tolle
Date: 2026-08-26
License: BSD 3-Clause
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import beta as beta_fn

from utils.helpers import aggregate_prediction_results, summarise_seed_metrics
from utils.plotting import NEUTRAL, NEUTRAL2, PSILO, ESCIT

from figure_making.paths import output_dir, perm_dirs, MissingInput
from figure_making.registry import register


# (label, null run tree, empirical run tree), shown together in one grid per panel.
MODELS = [
    ('graphTRIP', ('graphtrip', 'permutation_null'), ('graphtrip', 'weights')),
    ('Medusa-graphTRIP', ('medusa_graphtrip', 'permutation_null'),
     ('medusa_graphtrip', 'weights')),
    ('graphTRIP (FC + REACT)',
     ('ablation', 'feature_ablation', 'no_clinical_features', 'permutation_null'),
     ('ablation', 'feature_ablation', 'no_clinical_features')),
    ('graphTRIP (BDI)', ('graphtrip_bdi', 'permutation_null'), ('graphtrip_bdi', 'weights')),
    ('Clinical-only MLP',
     ('ablation', 'feature_ablation', 'control_mlp_raw', 'permutation_null'),
     ('ablation', 'feature_ablation', 'control_mlp_raw')),
]

# Same tuple shape, but each of these gets its own overview figure instead.
STANDALONE_MODELS = [
    ('SELSER', ('selser', 'permutation_null'), ('selser', 'selser')),
]

# Metrics to report, and whether a larger value is the better one.
METRICS = [('r', True), ('r2', True), ('mae', False), ('rmse', False)]

# Zero-shot analyses of graphTRIP's null weights: (label, null subdir, observed tree,
# prediction file, metrics). No retraining, so these ride along with the graphTRIP null.
#
# The atlas transfers stay on the same cohort and outcome, so every metric is comparable
# with the in-atlas result. psilodep1 is a different cohort, treatment and target, so
# predictions are not expected to fall on the identity line and only the correlation is
# interpretable.
TRANSFERS = [
    ('graphTRIP to Schaefer 200', ('transfer_atlas', 'schaefer200'),
     ('graphtrip', 'transfer_atlas', 'schaefer200'), 'initial_prediction_results.csv',
     METRICS),
    ('graphTRIP to AAL', ('transfer_atlas', 'aal'),
     ('graphtrip', 'transfer_atlas', 'aal'), 'initial_prediction_results.csv',
     METRICS),
    ('graphTRIP zero-shot on psilodep1', ('psilodep1',),
     ('validation', 'evaluate_graphtrip'),
     'initial_prediction_results_mean_vote_harmonised.csv',
     [('r', True)]),
]

TRANSFER_SOURCE = ('graphtrip', 'permutation_null')

# The metric the null histogram is drawn for.
HEADLINE = 'r'

# Panels are laid out in a grid, because the models do not fit in one row.
NCOLS = 3

# The observed value is marked in dark red, so that it reads against the grey/cyan nulls.
DARK_RED = '#AA0000'


# Metrics ----------------------------------------------------------------------------

def prediction_metrics(y_true, y_pred):
    '''
    Prediction metrics of one set of predictions.

    Deliberately the same definitions as summarise_seed_metrics, which supplies the
    seed-level values: the two levels have to be directly comparable.
    '''
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r, p_value = stats.pearsonr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    mse = np.mean((y_true - y_pred)**2)
    return {'r': r, 'p_value': p_value, 'r2': 1 - ss_res/ss_tot,
            'mae': np.mean(np.abs(y_true - y_pred)), 'mse': mse, 'rmse': np.sqrt(mse)}


def seed_number(seed_labels):
    '''Turns the 'seed_3' subdirectory names of summarise_seed_metrics into integers.'''
    return seed_labels.str.replace('seed_', '', regex=False).astype(int)


# Collecting the runs ----------------------------------------------------------------

def ensemble_predictions(run_dir, prediction_file='prediction_results.csv'):
    '''
    Mean-across-seed predictions of one directory of seed_* runs.

    aggregate_prediction_results caches its result under the directory, which is also where
    the empirical tree keeps its aggregate.
    '''
    return aggregate_prediction_results(
        results_file=os.path.join(run_dir, prediction_file))


def collect_empirical(parts):
    '''Observed ensemble metrics and the observed seed-level values.'''
    base = output_dir(*parts)
    agg = ensemble_predictions(base)
    seeds = summarise_seed_metrics(base_dir=base)
    seeds['seed'] = seed_number(seeds['seed'])
    return prediction_metrics(agg['label'], agg['prediction']), seeds, agg


def collect_null(parts, true_labels, n_seeds):
    '''
    Both levels of the null, plus the true-label probe.

    The probe correlates each null ensemble's predictions with the *unpermuted* outcome.
    A pipeline that leaked the outcome would recover it even when trained on permuted
    labels, so this is a sharper leakage test than the permuted-label correlation.

    Permutations with fewer than n_seeds (i.e. incomplete) runs are dropped.
    '''
    base = output_dir(*parts)
    dirs = perm_dirs(base)
    if not dirs:
        raise FileNotFoundError(f'No perm_* directories in {base}')

    ensemble, seed_level, incomplete = [], [], []
    for perm_dir in dirs:
        perm_seed = int(os.path.basename(perm_dir).split('_')[-1])

        found = len(glob.glob(os.path.join(perm_dir, 'seed_*', 'prediction_results.csv')))
        if found != n_seeds:
            incomplete.append((perm_seed, found))
            continue

        agg = ensemble_predictions(perm_dir).sort_values('subject_id')
        row = {'perm_seed': perm_seed, **prediction_metrics(agg['label'], agg['prediction'])}
        row['r_vs_true'] = stats.pearsonr(true_labels, agg['prediction'].values)[0]
        ensemble.append(row)

        seeds = summarise_seed_metrics(base_dir=perm_dir)
        seeds['seed'] = seed_number(seeds['seed'])
        seeds.insert(0, 'perm_seed', perm_seed)
        seed_level.append(seeds)

    if not ensemble:
        raise FileNotFoundError(f'No complete {n_seeds}-seed permutations in {base}')

    return (pd.DataFrame(ensemble), pd.concat(seed_level, ignore_index=True), incomplete)


def collect_transfer(null_subdir, observed_parts, prediction_file):
    '''
    Ensemble-level null draws and the observed value of one zero-shot analysis.

    The null runs sit inside each permutation's directory, so one draw per permutation is
    the mean prediction of that permutation's ten transferred models -- the same
    construction as the observed value.
    '''
    observed_dir = output_dir(*observed_parts)
    if not os.path.exists(observed_dir):
        return None, None
    obs_agg = ensemble_predictions(observed_dir, prediction_file)
    observed = prediction_metrics(obs_agg['label'], obs_agg['prediction'])

    draws = []
    for perm_dir in perm_dirs(output_dir(*TRANSFER_SOURCE)):
        run_dir = os.path.join(perm_dir, *null_subdir)
        if not glob.glob(os.path.join(run_dir, 'seed_*', prediction_file)):
            continue
        agg = ensemble_predictions(run_dir, prediction_file).sort_values('subject_id')
        draws.append({'perm_seed': int(os.path.basename(perm_dir).split('_')[-1]),
                      **prediction_metrics(agg['label'], agg['prediction'])})

    return observed, (pd.DataFrame(draws) if draws else None)


# Statistics -------------------------------------------------------------------------

def null_stats(observed, null, greater_is_better):
    '''
    Where the observed value falls in the null.

    The rank p is the result: it is exact, makes no distributional assumption, and floors
    at 1/(N+1), which is 0.0099 for the 100 permutations run here. The z-based p
    extrapolates from a Gaussian fit to the draws and is reported only as a reference,
    since it can quote values far below anything the draws actually resolve.
    '''
    null = np.asarray(null, dtype=float)
    n_draws = len(null)
    mu, sd = null.mean(), null.std(ddof=1)

    exceed = (null >= observed).sum() if greater_is_better else (null <= observed).sum()
    z = (observed - mu)/sd if greater_is_better else (mu - observed)/sd
    return {'observed': observed, 'n_draws': n_draws,
            'rank_p': (1 + exceed)/(1 + n_draws), 'rank_p_floor': 1/(1 + n_draws),
            'null_mean': mu, 'null_sd': sd, 'null_min': null.min(), 'null_max': null.max(),
            'z': z, 'z_p': 2*stats.norm.sf(abs(z))}


def parametric_null_sd(n_subjects):
    '''SD of the r = 0 null the manuscript's p-values assume, for n independent subjects.'''
    return 1/np.sqrt(n_subjects - 2)


def parametric_null_density(r_grid, n_subjects):
    '''Density of Pearson r under H0 for n independent observations.'''
    return (1 - r_grid**2)**((n_subjects - 4)/2)/beta_fn(0.5, (n_subjects - 2)/2)


def parametric_p(r, n_subjects):
    '''The two-sided p the manuscript quotes for a correlation of r.'''
    t = r*np.sqrt(n_subjects - 2)/np.sqrt(1 - r**2)
    return 2*stats.t.sf(abs(t), n_subjects - 2)


# Panels -----------------------------------------------------------------------------

R_GRID = np.linspace(-0.99, 0.99, 400)


def _model_axes(n_models, width=4.2, height=3.4):
    '''A grid of axes, one per model, with the unused cells removed.'''
    ncols = min(NCOLS, n_models)
    nrows = int(np.ceil(n_models/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width*ncols, height*nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_models:]:
        ax.remove()
    return fig, axes[:n_models]


def _save(fig, out, name):
    save_path = out.fig(name)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def draw_histogram(ax, c, title=None):
    '''Null draws with the observed value marked: the headline panel.'''
    null = c['ensemble'][HEADLINE].values
    observed = c['observed'][HEADLINE]
    s = c['stats'][HEADLINE]

    density = stats.norm.pdf(R_GRID, null.mean(), null.std(ddof=1))
    ax.fill_between(R_GRID, density, color=NEUTRAL2, alpha=0.25, linewidth=0)
    ax.plot(R_GRID, density, color=NEUTRAL2, linewidth=1.6, label='permutation null')
    ax.plot(null, np.zeros_like(null), '|', color=NEUTRAL2, markersize=10,
            markeredgewidth=1.2)
    ax.axvline(0, color=NEUTRAL2, linewidth=0.8, zorder=0)
    ax.axvline(observed, color=DARK_RED, linestyle='--', linewidth=2,
               label=f'observed r = {observed:.3f}')
    ax.set_xlabel('Ensemble r under label permutation')
    ax.set_ylabel('Density')
    ax.set_title(title if title is not None else
                 f"{c['label']}\nnull {null.mean():+.3f} $\\pm$ {null.std(ddof=1):.3f}, "
                 f"rank p = {s['rank_p']:.3f} ({s['n_draws']} draws)", fontsize=10)
    ax.legend(loc='upper left', fontsize=8, frameon=False)


def draw_vs_parametric(ax, c, title=None):
    '''
    The empirical null against the parametric r = 0 null.

    The panel that shows why the quoted p-values had to be replaced: a permutation null
    narrower than the parametric one would have made them conservative, a wider one makes
    them optimistic.
    '''
    null = c['ensemble'][HEADLINE].values
    observed = c['observed'][HEADLINE]
    mu, sd = null.mean(), null.std(ddof=1)
    n_subjects = c['n_subjects']

    ax.plot(R_GRID, parametric_null_density(R_GRID, n_subjects), color=NEUTRAL2,
            linewidth=1.6, label=f'parametric, SD = {parametric_null_sd(n_subjects):.3f}')
    ax.plot(R_GRID, stats.norm.pdf(R_GRID, mu, sd), color=ESCIT, linewidth=1.6,
            label=f'permutation, SD = {sd:.3f}')
    ax.plot(null, np.zeros_like(null), '|', color=ESCIT, markersize=10,
            markeredgewidth=1.2)
    ax.axvline(observed, color=DARK_RED, linestyle='--', linewidth=2,
               label=f'observed r = {observed:.3f}')
    ax.set_xlabel('r')
    ax.set_ylabel('Density')
    ax.set_title(title if title is not None else c['label'], fontsize=10)
    ax.legend(loc='upper left', fontsize=8, frameon=False)


def draw_seed_strip(ax, c, rng, title=None):
    '''Every null run against every empirical run, one point per trained model.'''
    groups = [(f"null\n({len(c['seed_level'])} runs)", c['seed_level'][HEADLINE].values,
               NEUTRAL),
              (f"observed\n({len(c['empirical_seeds'])} seeds)",
               c['empirical_seeds'][HEADLINE].values, PSILO)]
    for y, (label, values, colour) in enumerate(groups):
        ax.scatter(values, y + rng.uniform(-0.12, 0.12, len(values)), s=18,
                   color=colour, edgecolor=NEUTRAL2, linewidth=0.4, alpha=0.9)
    ax.axvline(0, color=NEUTRAL2, linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    ax.set_xlabel('Seed-level r')
    ax.set_title(title if title is not None else c['label'], fontsize=10)


def null_histogram(collected, out, name='permutation_null_histogram'):
    '''One histogram per model, shared grid.'''
    fig, axes = _model_axes(len(collected))
    for ax, c in zip(axes, collected):
        draw_histogram(ax, c)
    _save(fig, out, name)


def null_vs_parametric(collected, out, name='permutation_null_vs_parametric'):
    '''One empirical-vs-parametric comparison per model, shared grid.'''
    fig, axes = _model_axes(len(collected))
    for ax, c in zip(axes, collected):
        draw_vs_parametric(ax, c)
    _save(fig, out, name)


def seed_level_strip(collected, out, rng, name='permutation_null_seed_level'):
    '''One seed-level strip per model, shared grid.'''
    fig, axes = _model_axes(len(collected), height=3.0)
    for ax, c in zip(axes, collected):
        draw_seed_strip(ax, c, rng)
    _save(fig, out, name)


def model_overview(c, out, rng):
    '''
    All three panels of one model in a single figure.

    Used for models that are not comparable enough with the graphTRIP variants to share a
    grid with them, but still need the same three views.
    '''
    s = c['stats'][HEADLINE]
    null = c['ensemble'][HEADLINE].values

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    draw_histogram(axes[0], c, title='Permutation null')
    draw_vs_parametric(axes[1], c, title='Permutation vs parametric null')
    draw_seed_strip(axes[2], c, rng, title='Seed level')
    fig.suptitle(f"{c['label']} (n = {c['n_subjects']}): observed r = "
                 f"{c['observed'][HEADLINE]:.3f}, null {null.mean():+.3f} $\\pm$ "
                 f"{null.std(ddof=1):.3f}, rank p = {s['rank_p']:.3f} "
                 f"({s['n_draws']} draws)", fontsize=11)

    _save(fig, out, f"permutation_null_{c['label'].lower().replace(' ', '_')}")


# Target -----------------------------------------------------------------------------

def gather_models(models, out):
    '''
    Everything the panels need, for whichever models have a null tree on disk.

    Models are skipped individually rather than failing the target, so the panel keeps
    working while the remaining permutation arrays are still running.
    '''
    collected, missing, partial = [], [], []
    for label, null_parts, empirical_parts in models:
        try:
            observed, empirical_seeds, empirical_agg = collect_empirical(empirical_parts)
            true_labels = empirical_agg.sort_values('subject_id')['label'].values
            n_seeds = len(empirical_seeds)
            ensemble, seed_level, incomplete = collect_null(
                null_parts, true_labels, n_seeds)
        except (MissingInput, FileNotFoundError, ValueError) as error:
            missing.append(f'{label} ({error})')
            continue
        if incomplete:
            partial.append(f'{label}: ' + ', '.join(
                f'perm_{p} ({n}/{n_seeds} seeds)' for p, n in incomplete))

        collected.append({
            'label': label,
            'observed': observed,
            'ensemble': ensemble,
            'seed_level': seed_level,
            'empirical_seeds': empirical_seeds,
            'stats': {metric: null_stats(observed[metric], ensemble[metric], greater)
                      for metric, greater in METRICS},
            'seed_stats': {metric: null_stats(empirical_seeds[metric].mean(),
                                              seed_level[metric], greater)
                           for metric, greater in METRICS},
            'n_subjects': len(empirical_agg)})

    if missing:
        out.log(f'No permutation null for: {"; ".join(missing)}.')
        out.log()
    if partial:
        out.log('Permutations dropped for having an incomplete seed set -- a draw built '
                'from fewer seeds carries more seed noise and is not the same statistic:')
        for line in partial:
            out.log(f'  {line}')
        out.log()
    return collected


def gather_transfers(out):
    '''Ensemble-level nulls of graphTRIP's zero-shot analyses, where they have been run.'''
    rows, missing = [], []
    for label, null_subdir, observed_parts, prediction_file, metrics in TRANSFERS:
        try:
            observed, draws = collect_transfer(null_subdir, observed_parts, prediction_file)
        except (MissingInput, FileNotFoundError, ValueError) as error:
            missing.append(f'{label} ({error})')
            continue
        if observed is None or draws is None:
            missing.append(label)
            continue
        for metric, greater in metrics:
            rows.append({'analysis': label, 'metric': metric,
                         **null_stats(observed[metric], draws[metric], greater)})

    if missing:
        out.log(f'No zero-shot permutation null for: {"; ".join(missing)}.')
        out.log()
    return pd.DataFrame(rows)


@register('permutation_null', group='supp', subdir='SUPPLEMENTARY/permutation_null')
def permutation_null(ctx, out):
    '''
    Empirical null distributions of prediction performance, for every permuted model.

    Reports whether the pipeline leaks (is the null centred on zero?) and how the
    permutation null compares with the parametric r = 0 null.
    '''
    collected = gather_models(MODELS, out)
    standalone = gather_models(STANDALONE_MODELS, out)
    if not collected and not standalone:
        raise MissingInput(output_dir(*MODELS[0][1]))

    # Panels: a shared grid for the graphTRIP variants, one overview figure per model
    # that does not belong in that grid.
    if collected:
        null_histogram(collected, out)
        null_vs_parametric(collected, out)
        seed_level_strip(collected, out, ctx.rng)
    for c in standalone:
        model_overview(c, out, ctx.rng)

    # Tables: the draws themselves, and the statistics computed from them. Every model
    # is tabulated, whether or not it shares the grid.
    everything = collected + standalone
    draws, seeds, summary = [], [], []
    for c in everything:
        draws.append(c['ensemble'].assign(model=c['label']))
        seeds.append(c['seed_level'].assign(model=c['label']))
        for metric, _ in METRICS:
            summary.append({'model': c['label'], 'metric': metric,
                            **c['stats'][metric],
                            'seed_level_null_mean': c['seed_stats'][metric]['null_mean'],
                            'seed_level_null_sd': c['seed_stats'][metric]['null_sd']})
    summary = pd.DataFrame(summary)
    out.table('permutation_null_ensemble_draws', pd.concat(draws, ignore_index=True))
    out.table('permutation_null_seed_level', pd.concat(seeds, ignore_index=True))
    out.table('permutation_null_stats', summary)

    transfer = gather_transfers(out)
    if not transfer.empty:
        out.table('permutation_null_transfer_stats', transfer)

    # Report
    out.log(f'Permutation null for {len(everything)} model(s).')
    out.log(f'Ensemble level: {everything[0]["ensemble"].shape[0]} draws, each the mean '
            f'prediction of the {len(everything[0]["empirical_seeds"])} models sharing one '
            f'permutation. Seed level: {len(everything[0]["seed_level"])} runs.')
    out.log()

    for c in everything:
        s = c['stats'][HEADLINE]
        seed_s = c['seed_stats'][HEADLINE]
        n_subjects = c['n_subjects']
        sd_parametric = parametric_null_sd(n_subjects)
        separated = (c['empirical_seeds'][HEADLINE].min() > c['seed_level'][HEADLINE].max())

        out.log(f"--- {c['label']} (n = {n_subjects}) ---")
        out.log(f"  observed ensemble r = {s['observed']:.3f}")
        out.log(f"  null r              = {s['null_mean']:+.3f} +/- {s['null_sd']:.3f} "
                f"[{s['null_min']:+.3f}, {s['null_max']:+.3f}] over {s['n_draws']} draws")
        out.log(f"  rank p              = {s['rank_p']:.4f} (floor {s['rank_p_floor']:.4f})")
        out.log(f"  z-based p           = {s['z_p']:.2e} (z = {s['z']:.2f}, Gaussian fit, "
                f"reference only)")
        out.log(f"  parametric p        = {parametric_p(s['observed'], n_subjects):.2e} "
                f"(r = 0, n = {n_subjects})")
        out.log(f"  every observed seed above every null run: {separated}")
        out.log(f"  leakage probe, null vs true outcome: "
                f"{c['ensemble']['r_vs_true'].mean():+.3f} +/- "
                f"{c['ensemble']['r_vs_true'].std(ddof=1):.3f}")
        out.log(f"  CENTRE  null mean <= 0: {s['null_mean'] <= 0} "
                f"(seed level {seed_s['null_mean']:+.3f})")
        out.log(f"  SCALE   null SD {s['null_sd']:.3f} vs parametric {sd_parametric:.3f}: "
                f"{'narrower, the parametric p was conservative' if s['null_sd'] <= sd_parametric else 'WIDER, the parametric p was optimistic'}")
        out.log()

    out.log_df('All metrics', summary.round(4))
    if not transfer.empty:
        out.log_df('Zero-shot transfers', transfer.round(4))
