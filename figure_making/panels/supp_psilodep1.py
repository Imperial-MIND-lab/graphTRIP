"""
Supplementary: graphTRIP and the feature ablations transferred to psilodep1 without
fine-tuning, and the graphTRIP counterfactual estimates.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import glob
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
from statsmodels.stats.multitest import fdrcorrection

from utils.annotations import load_annotations
from utils.helpers import aggregate_prediction_results
from utils.plotting import regression_scatter, true_vs_pred_scatter
from utils.statsalg import calculate_cohens_d

from figure_making.common import collect_seed_metrics, raincloud_of_model_r
from figure_making.paths import output_dir, require, MissingInput
from figure_making.registry import register


PSILODEP1_NUM_SUBS = 16

# The outcome the transfer experiments evaluated, and the later outcome the second target
# re-scores the same predictions against.
PRIMARY_TARGET = 'QIDS_1week'
ALTERNATIVE_TARGET = 'QIDS_3months'

# (filename suffix, panel label) of the two input mappings, in panel order.
CONDITIONS = [('', 'No harmonisation'), ('_harmonised', 'Harmonised')]

GRAPHTRIP_PARTS = ('validation', 'evaluate_graphtrip')

# Labels match those of the psilodep2 feature ablation panel 
MODELS = [
    ('graphtrip', 'graphTRIP'),
    ('no_node_features', 'graphTRIP, Trained without REACT Node Features'),
    ('no_clinical_features', 'graphTRIP, Trained without Clinical Features'),
    ('no_react_no_clinical', 'graphTRIP, Trained without REACT Node or Clinical Features'),
    ('control_mlp', 'MLP, Trained on Clinical Data'),
    ('linreg_on_clinical_data', 'OLS Regression, Trained on Clinical Data'),
]

# Results directory of each model
RESULTS_PARTS = {'graphtrip': GRAPHTRIP_PARTS,
                 'control_mlp': ('validation', 'feature_ablation', 'control_mlp_raw')}

# graphTRIP is tested against the two clinical-only models
REFERENCE_MODEL = 'graphtrip'
BENCHMARK_MODELS = ['linreg_on_clinical_data', 'control_mlp']


def _results_dir(model):
    return output_dir(*RESULTS_PARTS.get(model, ('validation', 'feature_ablation', model)))


def _available_suffix(results_dir, suffix):
    '''
    The requested input mapping if it was evaluated for this model, otherwise the
    no-harmonisation one.

    A model whose only clinical input is Condition has nothing to harmonise, so the
    harmonised condition was never evaluated for it. Harmonisation is the identity map
    there, so its no-harmonisation results are also its harmonised results, and it belongs
    in the harmonised comparison rather than missing from it.
    '''
    if not suffix:
        return suffix
    if glob.glob(os.path.join(results_dir, 'seed_*', f'initial_metrics_mean_vote{suffix}.csv')):
        return suffix
    return ''


# Outcomes ---------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _psilodep1_column(column):
    '''
    A psilodep1 annotation column indexed by subject_id.
    '''
    annotations = load_annotations(study='psilodep1')
    return pd.Series(annotations[column].values, index=annotations['Patient'].values - 1)


def _rescore(results, target):
    '''
    Replaces the outcome the predictions are scored against.

    Subjects whose outcome is missing are dropped, so the returned frame can be shorter than
    the evaluated cohort.
    '''
    evaluated = _psilodep1_column(PRIMARY_TARGET).reindex(results['subject_id']).values
    if not np.allclose(results['label'].values, evaluated, equal_nan=True):
        raise ValueError('subject_id does not index the psilodep1 annotations as Patient - 1, '
                         f'so the {target} values cannot be matched to the predictions.')

    rescored = results.copy()
    rescored['label'] = _psilodep1_column(target).reindex(results['subject_id']).values
    return rescored.dropna(subset=['label'])


def _condition_results(results_dir, suffix, target):
    '''
    Mean-vote predictions of one input mapping, averaged across seeds, or None if that
    mapping was not evaluated.
    '''
    filename = f'initial_prediction_results_mean_vote{suffix}.csv'
    if not glob.glob(os.path.join(results_dir, 'seed_*', filename)):
        return None
    results = aggregate_prediction_results(results_file=os.path.join(results_dir, filename))
    return results if target == PRIMARY_TARGET else _rescore(results, target)


# Panels -----------------------------------------------------------------------------------

def condition_pair_figure(results_dir, out, name, title, target):
    '''
    Plots the zero-shot predictions of one model under each input mapping, side by side.
    '''
    panels = [(label, results) for suffix, label in CONDITIONS
              if (results := _condition_results(results_dir, suffix, target)) is not None]
    if not panels:
        raise MissingInput(os.path.join(results_dir, 'seed_*',
                                        'initial_prediction_results_mean_vote.csv'))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, (label, results) in zip(axes[0], panels):
        regression_scatter(results, ax=ax, title=f'{title}\n{label}', yerr='prediction_sem')
        if target != PRIMARY_TARGET:
            ax.set_xlabel(f'label ({target})')

    fig.tight_layout()
    save_path = out.fig(name)
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)

    for label, results in panels:
        r, p = pearsonr(results['label'], results['prediction'])
        out.log(f'{title} ({label}): r={r:.4f}, p={p:.4e}, n={len(results)}')


def _seed_r(models, suffix, target):
    '''
    Returns a {model: r} dataframe indexed by seed, so that models stay seed-aligned.

    One seed is one set of fold models, trained on the same split of psilodep2 and evaluated
    on the same psilodep1 patients.
    '''
    if target == PRIMARY_TARGET:
        specs = [(model, directory,
                  f'initial_metrics_mean_vote{_available_suffix(directory, suffix)}.csv')
                 for model, directory in models]
        metrics = collect_seed_metrics(specs, skip_missing=True)
        return metrics.pivot(index='seed', columns='model', values='r')

    columns = {}
    for model, directory in models:
        filename = ('initial_prediction_results_mean_vote'
                    f'{_available_suffix(directory, suffix)}.csv')
        per_seed = {}
        for path in sorted(glob.glob(os.path.join(directory, 'seed_*', filename))):
            seed = int(os.path.basename(os.path.dirname(path)).split('_')[-1])
            rescored = _rescore(pd.read_csv(path), target)
            per_seed[seed] = pearsonr(rescored['label'], rescored['prediction'])[0]
        if per_seed:
            columns[model] = pd.Series(per_seed)

    if not columns:
        raise FileNotFoundError(f'No zero-shot predictions to score against {target}.')
    return pd.DataFrame(columns)


def _seed_raincloud(out, name, suffix, models, target, num_subs):
    '''
    Seed sensitivity of the zero-shot correlation for every model, under one input mapping.
    '''
    identity = [model for model, directory in models
                if suffix and not _available_suffix(directory, suffix)]
    if identity:
        out.log(f"Harmonisation is the identity map for {', '.join(identity)}, which take "
                'no baseline severity scores as inputs; their unharmonised results are '
                'shown instead.')
    try:
        r = _seed_r(models, suffix, target)
    except FileNotFoundError:
        out.log(f'No zero-shot metrics found for any model; skipping {name}.')
        return

    distributions = {model: r[model].dropna().tolist()
                     for model, _ in models if model in r.columns}
    raincloud_of_model_r(distributions, out, name, num_subs=num_subs, offset=2,
                         figsize=(6, 3))


def comparison_tests(out, target):
    '''
    Paired t-tests of the per-seed zero-shot correlation: graphTRIP against each
    clinical-only benchmark, under each input mapping.

    The effect size is Cohen's d_z, the standardised mean of the paired differences, which
    is the effect size the paired t statistic corresponds to. p-values are FDR corrected
    across every test in the table.
    '''
    models = [(model, _results_dir(model)) for model in [REFERENCE_MODEL] + BENCHMARK_MODELS]
    rows = []
    for suffix, condition in CONDITIONS:
        r = _seed_r(models, suffix, target)

        for benchmark in BENCHMARK_MODELS:
            paired = r[[REFERENCE_MODEL, benchmark]].dropna()
            reference, other = paired[REFERENCE_MODEL], paired[benchmark]
            difference = reference - other
            t_stat, p_value = ttest_rel(reference, other)

            rows.append({
                'target': target,
                'condition': condition,
                'comparison': f'{REFERENCE_MODEL} - {benchmark}',
                'n_seeds': len(paired),
                f'{REFERENCE_MODEL}_mean_r': reference.mean(),
                f'{REFERENCE_MODEL}_median_r': reference.median(),
                'benchmark_mean_r': other.mean(),
                'benchmark_median_r': other.median(),
                'mean_difference': difference.mean(),
                'median_difference': difference.median(),
                'sd_difference': difference.std(ddof=1),
                't': t_stat,
                'p': p_value,
                'cohen_dz': calculate_cohens_d(difference.values)})

    table = pd.DataFrame(rows)
    table['p_fdr'] = fdrcorrection(table['p'].values)[1]
    table['n_fdr_tests'] = len(table)

    out.log('=== graphTRIP versus the clinical-only models (paired t-tests across seeds) ===')
    out.log('Positive difference: graphTRIP correlates more strongly than the benchmark.')
    out.log_df('Zero-shot performance comparisons', table)
    out.table('graphtrip_vs_clinical_only_tests', table)
    return table


def zeroshot_on_psilodep1(out, target):
    '''
    Every model of the feature ablation design, transferred zero-shot onto psilodep1 and
    scored against one outcome.
    '''
    out.log(f'Outcome: {target}.')
    if target != PRIMARY_TARGET:
        out.log('The predictions are the ones evaluated against '
                f'{PRIMARY_TARGET}: a zero-shot prediction uses no labels, so scoring it '
                'against another outcome needs no new model run.')

    missing, num_subs = [], PSILODEP1_NUM_SUBS
    for model, title in MODELS:
        results_dir = _results_dir(model)
        if not os.path.exists(results_dir):
            missing.append(model)
            continue
        panel_title = title if target == PRIMARY_TARGET else f'{title}\n{target}'
        condition_pair_figure(results_dir, out, f'{model}_psilodep1', panel_title, target)
        if model == REFERENCE_MODEL:
            num_subs = len(_condition_results(results_dir, '', target))

    if len(missing) == len(MODELS):
        raise MissingInput(output_dir('validation'))
    if missing:
        out.log(f"No zero-shot results for: {', '.join(missing)}.")

    # Seed sensitivity ---------------------------------------------------------------------
    models = [(model, _results_dir(model)) for model, _ in MODELS]

    out.log(f'=== Zero-shot correlation across seeds (no harmonisation), n={num_subs} ===')
    _seed_raincloud(out, 'raincloud_psilodep1', '', models, target, num_subs)

    out.log(f'=== Zero-shot correlation across seeds (harmonised), n={num_subs} ===')
    _seed_raincloud(out, 'raincloud_psilodep1_harmonised', '_harmonised', models, target,
                    num_subs)

    # Does the brain add anything over the clinical scores? ---------------------------------
    comparison_tests(out, target)


@register('feature_ablations_on_psilodep1', group='supp',
          subdir='SUPPLEMENTARY/feature_ablations_on_psilodep1')
def feature_ablations_on_psilodep1(ctx, out):
    '''graphTRIP and every feature ablation, transferred zero-shot onto psilodep1.'''
    zeroshot_on_psilodep1(out, PRIMARY_TARGET)


@register('feature_ablations_on_psilodep1_3months', group='supp',
          subdir='SUPPLEMENTARY/feature_ablations_on_psilodep1_3months')
def feature_ablations_on_psilodep1_3months(ctx, out):
    '''
    The same models and the same predictions, scored against the 3-month outcome.
    '''
    zeroshot_on_psilodep1(out, ALTERNATIVE_TARGET)


@register('graphtrip_counterfactuals', group='supp',
          subdir='SUPPLEMENTARY/graphtrip_counterfactuals')
def graphtrip_counterfactuals(ctx, out):
    psilo_dir = require(output_dir('graphtrip', 'predictions_psilocybin'))
    escit_dir = require(output_dir('graphtrip', 'predictions_escitalopram'))

    psilo_results = aggregate_prediction_results(
        results_file=os.path.join(psilo_dir, 'initial_prediction_results.csv'))
    escit_results = aggregate_prediction_results(
        results_file=os.path.join(escit_dir, 'initial_prediction_results.csv'))

    combined = psilo_results.rename(columns={'prediction': 'psilocybin_prediction'})
    combined = combined.merge(escit_results[['subject_id', 'prediction']],
                              on='subject_id', how='left')
    combined = combined.rename(columns={'prediction': 'escitalopram_prediction'})
    combined = combined.sort_values(by='subject_id')
    combined['Condition'] = ctx.conditions

    true_vs_pred_scatter(combined,
                         save_path=out.fig('escitalopram_vs_psilocybin_predictions'),
                         ycol='escitalopram_prediction',
                         xcol='psilocybin_prediction')
