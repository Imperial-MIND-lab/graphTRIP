"""
Supplementary: the distribution of subject-level absolute prediction errors.

The prediction scatters of Fig. 2a and Fig. 5 summarise accuracy as a single MAE, which
says nothing about how the error is spread over patients. The same mean-across-seed
predictions are shown here as the distribution of |y - y_hat| across the psilodep2
patients, for graphTRIP and Medusa-graphTRIP.

Author: Hanna M. Tolle
Date: 2026-09-02
License: BSD 3-Clause
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils.helpers import aggregate_prediction_results
from utils.plotting import NEUTRAL, NEUTRAL2

from figure_making.panels.supp_misc import model_target
from figure_making.paths import output_dir, require
from figure_making.registry import register


# The two models whose predictions the main text plots, in Fig. 2a and Fig. 5.
MODELS = [
    ('graphTRIP', ('graphtrip', 'weights')),
    ('Medusa-graphTRIP', ('medusa_graphtrip', 'weights')),
]

# One bin per point of the outcome scale.
BIN_WIDTH = 1.0

# QIDS severity bands are five points wide, so an error of five points or less keeps a
# patient in the predicted band or the one next to it.
ERROR_THRESHOLDS = [2.5, 5.0, 10.0]

# The MAE is marked in dark red, so that it reads against the grey bars.
DARK_RED = '#AA0000'


def collect_errors(name, parts):
    '''Subject-level absolute errors of one model's mean-across-seed predictions.'''
    base_dir = require(output_dir(*parts))
    results = aggregate_prediction_results(
        results_file=os.path.join(base_dir, 'prediction_results.csv'))

    errors = results.copy()
    errors['model'] = name
    errors['abs_error'] = (errors['label'] - errors['prediction']).abs()
    return errors, model_target(base_dir)


def summarise(errors, name, target):
    '''
    The error distribution as one row.

    The baseline is the MAE of always predicting the cohort mean outcome, which is what
    the spread of the errors has to be read against.
    '''
    abs_error = errors['abs_error']
    labels = errors['label']

    summary = {'model': name, 'target': target, 'n': len(errors),
               'mae': abs_error.mean(), 'sd': abs_error.std(ddof=1),
               'sem': abs_error.sem(), 'rmse': np.sqrt((abs_error ** 2).mean()),
               'median': abs_error.median(),
               'q25': abs_error.quantile(0.25), 'q75': abs_error.quantile(0.75),
               'p90': abs_error.quantile(0.9),
               'min': abs_error.min(), 'max': abs_error.max(),
               'skew': stats.skew(abs_error),
               'baseline_mae': (labels - labels.mean()).abs().mean()}

    for threshold in ERROR_THRESHOLDS:
        summary[f'percent_within_{threshold:g}'] = (abs_error <= threshold).mean() * 100
    return summary


def outcome_unit(target):
    '''The scale an outcome is measured on, e.g. "QIDS" for QIDS_Final_Integration.'''
    return target.split('_')[0] if target else 'outcome'


def error_histograms(collected, out, name='absolute_error_histograms'):
    '''One histogram per model, on shared bins so that the two are comparable.'''
    max_error = max(c['errors']['abs_error'].max() for c in collected)
    edges = np.arange(0, np.ceil(max_error / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH, BIN_WIDTH)

    fig, axes = plt.subplots(1, len(collected), figsize=(4.6 * len(collected), 3.4),
                             sharex=True, sharey=True, constrained_layout=True,
                             squeeze=False)
    for ax, c in zip(axes.ravel(), collected):
        s = c['summary']
        ax.hist(c['errors']['abs_error'], bins=edges,
                color=NEUTRAL, edgecolor=NEUTRAL2, linewidth=0.8)
        ax.axvline(s['baseline_mae'], color=NEUTRAL2, linestyle=':', linewidth=1.6,
                   label=f"cohort-mean baseline = {s['baseline_mae']:.2f}")
        ax.axvline(s['mae'], color=DARK_RED, linestyle='--', linewidth=2,
                   label=f"MAE = {s['mae']:.2f}")
        ax.set_xlabel(f"Absolute prediction error ({outcome_unit(s['target'])} points)")
        ax.set_ylabel('Patients')
        # Both panels carry the y label, so both need their counts readable; sharey
        # would otherwise leave the second one labelled but unticked.
        ax.tick_params(labelleft=True)
        ax.set_title(f"{s['model']} (n = {s['n']})\n"
                     f"median {s['median']:.2f} "
                     f"[IQR {s['q25']:.2f}-{s['q75']:.2f}], max {s['max']:.2f}",
                     fontsize=10)
        ax.legend(loc='upper right', fontsize=8, frameon=False)

    save_path = out.fig(name)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


@register('error_distribution', group='supp', subdir='SUPPLEMENTARY/error_distribution')
def error_distribution(ctx, out):
    '''
    Histograms of the subject-level absolute prediction errors, per model.
    '''
    collected = []
    for name, parts in MODELS:
        errors, target = collect_errors(name, parts)
        collected.append({'name': name, 'errors': errors,
                          'summary': summarise(errors, name, target)})

    error_histograms(collected, out)

    # Tables: the per-patient errors, and the distributions they summarise to
    columns = ['model', 'subject_id', 'Condition', 'label', 'prediction',
               'prediction_sem', 'abs_error']
    per_patient = pd.concat([c['errors'][columns] for c in collected], ignore_index=True)
    summary = pd.DataFrame([c['summary'] for c in collected])
    out.table('absolute_errors_per_patient', per_patient)
    out.table('absolute_error_summary', summary)

    # Report
    out.log('=== Distribution of absolute prediction errors ===')
    out.log_df('Summary', summary)

    for c in collected:
        s = c['summary']
        within = ', '.join(f"<= {t:g}: {s[f'percent_within_{t:g}']:.1f}%"
                           for t in ERROR_THRESHOLDS)
        out.log(f"--- {s['model']} (n = {s['n']}, target {s['target']}) ---")
        out.log(f"  MAE      = {s['mae']:.2f} +/- {s['sem']:.2f} (SD {s['sd']:.2f}), "
                f"RMSE = {s['rmse']:.2f}")
        out.log(f"  median   = {s['median']:.2f} "
                f"[IQR {s['q25']:.2f}-{s['q75']:.2f}], range "
                f"{s['min']:.2f}-{s['max']:.2f}, skew {s['skew']:+.2f}")
        out.log(f"  baseline = {s['baseline_mae']:.2f} (always predicting the cohort mean)")
        out.log(f"  patients within {within}")
        out.log()

    # The two models predict the same patients, so their errors are paired
    if len(collected) == 2:
        first, second = (c['errors'].sort_values('subject_id') for c in collected)
        difference = first['abs_error'].values - second['abs_error'].values
        w, p = stats.wilcoxon(difference)
        out.log(f"{collected[0]['name']} minus {collected[1]['name']}: "
                f"median paired difference {np.median(difference):+.2f}, "
                f"Wilcoxon W = {w:.1f}, p = {p:.3f}")
