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

The panels answer three questions. Whether the null is centred on zero, which is what a
pipeline free of leakage gives; whether its width matches the parametric r = 0 null that
p-values on a correlation assume; and whether the standardised effect depends on how many
seeds are ensembled, which is what licenses the reported 10-seed statistic.

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

from figure_making.paths import output_dir, MissingInput
from figure_making.registry import register


# (label, null run tree, empirical run tree)
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

# Ensemble sizes for the invariance check, and how much resampling it uses.
ENSEMBLE_SIZES = [1, 2, 3, 4, 5, 7, 10]
INVARIANCE_DRAWS = 200
INVARIANCE_REPEATS = 20

# Panels are laid out in a grid, because six models do not fit in one row.
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


def perm_dirs(base):
    '''The perm_* subdirectories of a null tree, in permutation order.'''
    dirs = glob.glob(os.path.join(base, 'perm_*'))
    return sorted(dirs, key=lambda d: int(os.path.basename(d).split('_')[-1]))


def seed_predictions(run_dir):
    '''Per-seed prediction vectors of one directory, aligned on subject_id.'''
    frames = []
    for path in sorted(glob.glob(os.path.join(run_dir, 'seed_*', 'prediction_results.csv'))):
        frames.append(pd.read_csv(path).sort_values('subject_id'))
    if not frames:
        return None, None
    labels = frames[0]['label'].values
    return np.array([f['prediction'].values for f in frames]), labels


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


def standardised_effect(observed, null, greater_is_better):
    '''Signed z of an observed value against a null, oriented so larger is better.'''
    null = np.asarray(null, dtype=float)
    mu, sd = null.mean(), null.std(ddof=1)
    if sd == 0:
        return np.nan
    return (observed - mu)/sd if greater_is_better else (mu - observed)/sd


def ensemble_size_effects(empirical_preds, empirical_labels, null_runs, rng):
    '''
    Standardised effect as a function of the number of seeds averaged.

    Averaging predictions across seeds removes seed noise from the true signal and from
    each permutation's spurious signal alike, so it inflates the observed metric and the
    null spread together. If the ratio is flat, the reported ten-seed statistic is not an
    artefact of ensembling. Observed and null are always computed at the same ensemble
    size, since comparing across sizes would not be a valid test.
    '''
    n_seeds = min([len(empirical_preds)] + [len(preds) for preds, _ in null_runs])
    rows = []
    for k in [k for k in ENSEMBLE_SIZES if k <= n_seeds]:
        observed = [prediction_metrics(
            empirical_labels, empirical_preds[rng.choice(n_seeds, k, replace=False)].mean(0))
            for _ in range(INVARIANCE_DRAWS)]
        null = []
        for _ in range(INVARIANCE_REPEATS):
            for preds, labels in null_runs:
                idx = rng.choice(len(preds), k, replace=False)
                null.append(prediction_metrics(labels, preds[idx].mean(0)))
        observed, null = pd.DataFrame(observed), pd.DataFrame(null)

        row = {'k_seeds': k}
        for metric, greater in METRICS:
            row[f'z_{metric}'] = standardised_effect(
                observed[metric].mean(), null[metric], greater)
            row[f'observed_{metric}'] = observed[metric].mean()
            row[f'null_sd_{metric}'] = null[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


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


def null_histogram(collected, out, name='permutation_null_histogram'):
    '''Null draws with the observed value marked: the headline panel.'''
    r_grid = np.linspace(-0.99, 0.99, 400)
    fig, axes = _model_axes(len(collected))
    for ax, c in zip(axes, collected):
        null = c['ensemble'][HEADLINE].values
        observed = c['observed'][HEADLINE]
        s = c['stats'][HEADLINE]

        density = stats.norm.pdf(r_grid, null.mean(), null.std(ddof=1))
        ax.fill_between(r_grid, density, color=NEUTRAL2, alpha=0.25, linewidth=0)
        ax.plot(r_grid, density, color=NEUTRAL2, linewidth=1.6, label='permutation null')
        ax.plot(null, np.zeros_like(null), '|', color=NEUTRAL2, markersize=10,
                markeredgewidth=1.2)
        ax.axvline(0, color=NEUTRAL2, linewidth=0.8, zorder=0)
        ax.axvline(observed, color=DARK_RED, linestyle='--', linewidth=2,
                   label=f'observed r = {observed:.3f}')
        ax.set_xlabel('Ensemble r under label permutation')
        ax.set_ylabel('Density')
        ax.set_title(f"{c['label']}\nnull {null.mean():+.3f} $\\pm$ {null.std(ddof=1):.3f}, "
                     f"rank p = {s['rank_p']:.3f} ({s['n_draws']} draws)", fontsize=10)
        ax.legend(loc='upper left', fontsize=8, frameon=False)

    _save(fig, out, name)


def null_vs_parametric(collected, out, name='permutation_null_vs_parametric'):
    '''
    The empirical null against the parametric r = 0 null.

    The panel that shows why the quoted p-values had to be replaced: a permutation null
    narrower than the parametric one would have made them conservative, a wider one makes
    them optimistic.
    '''
    r_grid = np.linspace(-0.99, 0.99, 400)
    fig, axes = _model_axes(len(collected))
    for ax, c in zip(axes, collected):
        null = c['ensemble'][HEADLINE].values
        observed = c['observed'][HEADLINE]
        mu, sd = null.mean(), null.std(ddof=1)
        n_subjects = c['n_subjects']

        ax.plot(r_grid, parametric_null_density(r_grid, n_subjects), color=NEUTRAL2,
                linewidth=1.6, label=f'parametric, SD = {parametric_null_sd(n_subjects):.3f}')
        ax.plot(r_grid, stats.norm.pdf(r_grid, mu, sd), color=ESCIT, linewidth=1.6,
                label=f'permutation, SD = {sd:.3f}')
        ax.plot(null, np.zeros_like(null), '|', color=ESCIT, markersize=10,
                markeredgewidth=1.2)
        ax.axvline(observed, color=DARK_RED, linestyle='--', linewidth=2,
                   label=f'observed r = {observed:.3f}')
        ax.set_xlabel('r')
        ax.set_ylabel('Density')
        ax.set_title(c['label'], fontsize=10)
        ax.legend(loc='upper left', fontsize=8, frameon=False)

    _save(fig, out, name)


def seed_level_strip(collected, out, rng, name='permutation_null_seed_level'):
    '''Every null run against every empirical run, one point per trained model.'''
    fig, axes = _model_axes(len(collected), height=3.0)
    for ax, c in zip(axes, collected):
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
        ax.set_title(c['label'], fontsize=10)

    _save(fig, out, name)


def true_label_probe(collected, out, name='permutation_null_true_label_probe'):
    '''
    Null models against the outcome they were never shown.

    Leakage of the true outcome into training would push these correlations above zero,
    whatever labels the model was handed.
    '''
    fig, ax = plt.subplots(figsize=(1.0 + 1.1*len(collected), 3.4), constrained_layout=True)
    for x, c in enumerate(collected):
        values = c['ensemble']['r_vs_true'].values
        ax.scatter(np.full(len(values), x), values, s=22, color=NEUTRAL,
                   edgecolor=NEUTRAL2, linewidth=0.4)
        ax.hlines(values.mean(), x - 0.2, x + 0.2, color=PSILO, linewidth=2)
    ax.axhline(0, color=NEUTRAL2, linewidth=0.8, zorder=0)
    ax.set_xticks(range(len(collected)))
    ax.set_xticklabels([c['label'] for c in collected], fontsize=8, rotation=30,
                       ha='right')
    ax.set_xlim(-0.5, len(collected) - 0.5)
    ax.set_ylabel('r of null predictions with the true outcome')

    _save(fig, out, name)


def ensemble_size_panel(invariance, out, name='permutation_null_ensemble_size'):
    '''Standardised effect against ensemble size, one axis per model.'''
    fig, axes = _model_axes(len(invariance), height=3.0)
    colours = {'r': PSILO, 'r2': ESCIT, 'mae': NEUTRAL2}
    for ax, (label, table) in zip(axes, invariance):
        for metric in ['r', 'r2', 'mae']:
            ax.plot(table['k_seeds'], table[f'z_{metric}'], 'o-', color=colours[metric],
                    linewidth=1.6, markersize=4, label=f'z({metric})')
        ax.set_xlabel('Seeds averaged per ensemble')
        ax.set_ylabel('Standardised effect')
        ax.set_ylim(bottom=0)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, frameon=False)

    _save(fig, out, name)


# Target -----------------------------------------------------------------------------

def gather_models(out, rng):
    '''
    Everything the panels need, for whichever models have a null tree on disk.

    Models are skipped individually rather than failing the target, so the panel keeps
    working while the remaining permutation arrays are still running.
    '''
    collected, invariance, missing, partial = [], [], [], []
    for label, null_parts, empirical_parts in MODELS:
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

        empirical_preds, empirical_labels = seed_predictions(output_dir(*empirical_parts))
        null_runs = [run for run in (seed_predictions(d)
                                     for d in perm_dirs(output_dir(*null_parts)))
                     if run[0] is not None and len(run[0]) == n_seeds]
        if empirical_preds is not None and len(empirical_preds) > 1 and null_runs:
            invariance.append((label, ensemble_size_effects(
                empirical_preds, empirical_labels, null_runs, rng)))

    if missing:
        out.log(f'No permutation null for: {"; ".join(missing)}.')
        out.log()
    if partial:
        out.log('Permutations dropped for having an incomplete seed set -- a draw built '
                'from fewer seeds carries more seed noise and is not the same statistic:')
        for line in partial:
            out.log(f'  {line}')
        out.log()
    return collected, invariance


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

    Reports whether the pipeline leaks (is the null centred on zero?), how the permutation
    null compares with the parametric r = 0 null, and whether the standardised effect
    depends on the number of seeds ensembled.
    '''
    collected, invariance = gather_models(out, ctx.rng)
    if not collected:
        raise MissingInput(output_dir(*MODELS[0][1]))

    # Panels
    null_histogram(collected, out)
    null_vs_parametric(collected, out)
    seed_level_strip(collected, out, ctx.rng)
    true_label_probe(collected, out)
    if invariance:
        ensemble_size_panel(invariance, out)

    # Tables: the draws themselves, and the statistics computed from them
    draws, seeds, summary = [], [], []
    for c in collected:
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
    for label, table in invariance:
        out.table(f'permutation_null_ensemble_size_{label.split()[0].lower()}', table)

    transfer = gather_transfers(out)
    if not transfer.empty:
        out.table('permutation_null_transfer_stats', transfer)

    # Report
    out.log(f'Permutation null for {len(collected)} model(s).')
    out.log(f'Ensemble level: {collected[0]["ensemble"].shape[0]} draws, each the mean '
            f'prediction of the {len(collected[0]["empirical_seeds"])} models sharing one '
            f'permutation. Seed level: {len(collected[0]["seed_level"])} runs.')
    out.log()

    for c in collected:
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
    for label, table in invariance:
        out.log_df(f'Ensemble size, {label}',
                   table[['k_seeds'] + [f'z_{m}' for m, _ in METRICS]].round(3))
    if not transfer.empty:
        out.log_df('Zero-shot transfers', transfer.round(4))
