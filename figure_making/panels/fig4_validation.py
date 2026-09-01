"""
Fig. 4: Generalisation to an independent dataset (psilodep1).

graphTRIP is applied zero-shot: no weight is updated and no psilodep1 outcome is used.

Panels:
- b. baseline severity of the two cohorts, which the model is asked to extrapolate across
- c. zero-shot prediction of graphTRIP
- d. reconstruction performance of the graphTRIP VGAE, not fine-tuned, tested against
     the primary-dataset reconstructions of Fig. 2d
- e. r and partial r, for the full model, the imaging-only ablation and the clinical-only
     benchmark. One bar panel removes baseline QIDS from both sides, a second removes
     baseline QIDS and BDI, the two severity scores graphTRIP receives

Panels c and e are drawn without clinical harmonisation for the main figure, and again
with it as separate files.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.loader import DataLoader

from experiments.ingredients.data_ingredient import (
    HARMONISED, NO_HARMONISATION, build_transfer_inputs)
from utils.configs import load_configs_from_json, load_ingredient_configs
from utils.helpers import aggregate_prediction_results
from utils.plotting import NEUTRAL, NEUTRAL2, PSILO, regression_scatter
from utils.statsalg import compare_reconstruction_performance

from figure_making.common import (
    ABLATION_NAMES, attach_annotations, baseline_severity_panels, partial_correlation,
    plot_correlation_boxplot, study_annotations)
from figure_making.loaders import get_device, load_dataset, load_mlps, load_vgaes
from figure_making.paths import output_dir, perm_dirs, require
from figure_making.registry import register


# Independent fold assignments drawn for the ensemble-matched reconstruction control
N_MATCHED_DRAWS = 10

# (label, filename suffix) of the two input mappings. The first is the main figure's.
CONDITIONS = [('no harmonisation', ''), ('harmonised', '_harmonised')]

# Zero-shot predictions of one model, averaged over its fold models.
PREDICTIONS_FILE = 'initial_prediction_results_mean_vote'

GRAPHTRIP = 'Full graphTRIP'
NO_CLINICAL = 'graphTRIP without clinical data'
CLINICAL_ONLY = 'Clinical-only MLP'

# (results directory parts, label) of every model in the summary table. control_mlp_raw is
# the clinical-only MLP retrained for the transfer; the stale control_mlp/ tree is not read.
MODELS = [
    (('validation', 'evaluate_graphtrip'), GRAPHTRIP),
    (('validation', 'feature_ablation', 'no_node_features'),
     ABLATION_NAMES['no_node_features']),
    (('validation', 'feature_ablation', 'no_clinical_features'), NO_CLINICAL),
    (('validation', 'feature_ablation', 'no_react_no_clinical'),
     ABLATION_NAMES['no_react_no_clinical']),
    (('validation', 'feature_ablation', 'control_mlp_raw'), CLINICAL_ONLY),
]

# Subset shown in the bar panel; the benchmark is pinned to the bottom.
PANEL_MODELS = [GRAPHTRIP, NO_CLINICAL, CLINICAL_ONLY]

# Null models exist for graphTRIP only, so its p-values are NaN for every other model.
NULL_MODEL = GRAPHTRIP
NULL_PARTS = ('graphtrip', 'permutation_null')

# Predictors compared in the sensitivity tables. QIDS_Before is the score itself, used as
# a zero-parameter predictor, so it has no partial correlation given itself.
SENSITIVITY_MODELS = [GRAPHTRIP, CLINICAL_ONLY]
BASELINE_PREDICTOR = 'QIDS_Before'

# The baseline severity scores graphTRIP reads, attached to every model's predictions.
BASELINE_COVARIATES = ('QIDS_Before', 'BDI_Before')

# Covariate sets removed from both sides of the correlation, one bar panel each.
PARTIAL_SPECS = [
    {'suffix': '_qids', 'covariates': ['QIDS_Before'],
     'legend': 'partial r | QIDS_Before'},
    {'suffix': '_qids_bdi', 'covariates': list(BASELINE_COVARIATES),
     'legend': 'partial r | QIDS_Before, BDI_Before'},
]
PARTIAL_COLUMNS = [f"partial_r{spec['suffix']}" for spec in PARTIAL_SPECS]

# Statistics the correlation table reports, one row each.
STATISTICS = ('r', *PARTIAL_COLUMNS, *[f'r_with_{c}' for c in BASELINE_COVARIATES])

# Outcome the models were trained on, and the arm they were trained on.
TRAINING_TARGET = 'QIDS_Final_Integration'
PSILOCYBIN_CONDITION = 1

# Columns shared by the two reconstruction ensembles.
COMPARISON_COLUMNS = ['ensemble', 'feature', 'n_draws', 'n_primary', 'mean_primary',
                      'n_validation', 'mean_validation', 'mean_difference', 'cohens_d',
                      'cohens_d_ci_low', 'cohens_d_ci_high', 'p_uncorrected', 'p_fdr',
                      'frac_significant_fdr']

SCATTER_XLABEL = 'True QIDS, 1 week post 10+25-mg psilocybin for TRD'
SCATTER_YLABEL = 'Predicted QIDS, 3 weeks post 2x25-mg psilocybin for MDD'

# Permutation importance: the MLP input blocks whose contribution is measured, and how many
# times each is shuffled across patients within a training seed.
LATENT_BLOCK = 'Brain latents (z)'
IMPORTANCE_ATTRS = ['QIDS_Before', 'BDI_Before']
N_IMPORTANCE_PERMUTATIONS = 100

BAR_COLOR_FULL = NEUTRAL
BAR_COLOR_PARTIAL = PSILO
BAR_HEIGHT = 0.68
NESTED_FRACTION = 0.42


def matched_fold_assignment(num_subs, num_folds, num_seeds, rng):
    '''
    Draws a random k-fold assignment for a dataset that no model was trained on.

    Each psilodep2 subject is reconstructed only by the one VGAE per seed that held them
    out, so their reconstruction is an average over num_seeds models. Reconstructing
    psilodep1 with every VGAE averages num_seeds * num_folds models instead, and the
    extra averaging cancels sampling noise. Spreading the psilodep1 subjects over the
    folds of each seed restores the matched ensemble size.

    Returns:
    -------
        dict: {seed key: array of length num_subs holding the fold each subject is
               assigned to}, in the format get_mean_test_reconstructions expects.
    '''
    if num_subs < num_folds:
        raise ValueError(f'Cannot spread {num_subs} subjects over {num_folds} folds: '
                         'every fold needs at least one subject.')
    # Balanced fold sizes, then shuffled independently for each seed
    folds = np.tile(np.arange(num_folds), int(np.ceil(num_subs / num_folds)))[:num_subs]
    return {f'seed_{seed}': rng.permutation(folds) for seed in range(num_seeds)}


def summarise_matched_draws(per_draw):
    '''
    Condenses the per-draw comparisons of the ensemble-matched control into one row per
    feature, taking the median over draws. Each draw is a different arbitrary assignment
    of val subjects to folds.
    '''
    df = pd.concat(per_draw, ignore_index=True)
    summary = df.groupby('feature', sort=False).agg(
        n_draws=('feature', 'size'),
        n_primary=('n_primary', 'first'),
        mean_primary=('mean_primary', 'first'),
        n_validation=('n_validation', 'first'),
        mean_validation=('mean_validation', 'median'),
        mean_difference=('mean_difference', 'median'),
        cohens_d=('cohens_d', 'median'),
        cohens_d_ci_low=('cohens_d_ci_low', 'median'),
        cohens_d_ci_high=('cohens_d_ci_high', 'median'),
        p_uncorrected=('p_uncorrected', 'median'),
        p_fdr=('p_fdr', 'median'),
        frac_significant_fdr=('significant_fdr', 'mean'))
    return summary.reset_index()


def stack_reconstruction_tests(all_folds, matched):
    '''
    The two ensembles in one frame, one row per feature and ensemble.

    all_folds rows are single tests, so n_draws is 1 and frac_significant_fdr is 0 or 1;
    ensemble_matched rows are medians over the draws.
    '''
    all_folds = all_folds.assign(
        ensemble='all_folds', n_draws=1,
        frac_significant_fdr=all_folds['significant_fdr'].astype(float))
    matched = matched.assign(ensemble='ensemble_matched')
    return pd.concat([all_folds[COMPARISON_COLUMNS], matched[COMPARISON_COLUMNS]],
                     ignore_index=True)


def reconstruction_comparison(ctx, out, psilodep1_data):
    '''
    Tests whether VGAE reconstruction quality differs between the primary dataset
    (Fig. 2d) and the independent validation dataset (Fig. 4d).

    Only the correlations are tested, under two ensembles: all_folds is what panel d
    plots, ensemble_matched equalises the number of VGAEs averaged per subject.
    '''
    _, primary_x = ctx.core_reconstructions
    _, psilodep1_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, None)

    num_folds = len(ctx.vgaes_dict['seed_0'])
    out.log('=== Primary vs validation reconstruction ===')
    out.log(f'all_folds: psilodep2 averages {ctx.num_seeds} VGAEs per subject (the '
            f'held-out fold of each seed), psilodep1 averages '
            f'{ctx.num_seeds * num_folds} (every fold of every seed).')
    out.log(f'ensemble_matched: each psilodep1 subject is reconstructed by one randomly '
            f'drawn fold per seed instead of all {num_folds}, so that subjects of both '
            f'datasets average {ctx.num_seeds} VGAEs; {N_MATCHED_DRAWS} draws.')

    all_folds = compare_reconstruction_performance(primary_x['metrics'],
                                                   psilodep1_x['metrics'])['corr']

    rng = np.random.default_rng(ctx.cfg.seed)
    per_draw = []
    for draw in range(N_MATCHED_DRAWS):
        assignment = matched_fold_assignment(len(psilodep1_data), num_folds,
                                             ctx.num_seeds, rng)
        _, matched_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, assignment)
        per_draw.append(compare_reconstruction_performance(
            primary_x['metrics'], matched_x['metrics'])['corr'].assign(draw=draw))

    table = stack_reconstruction_tests(all_folds, summarise_matched_draws(per_draw))
    out.table('d_reconstruction_tests', table)
    out.log_df('Primary vs validation reconstruction (correlation)', table)

    return psilodep1_x


# Zero-shot predictions -------------------------------------------------------------------

def load_zeroshot_results(base_dir, suffix):
    '''
    Zero-shot predictions of one model under one input mapping, with the baseline scores.

    Models trained without clinical inputs read no severity score, so harmonisation is the
    identity for them and only the unharmonised file was written; the harmonised condition
    falls back to it. Their prediction CSVs also carry no baseline scores, which the partial
    correlations need, so they are joined on from the psilodep1 annotations.
    '''
    path = os.path.join(base_dir, f'{PREDICTIONS_FILE}{suffix}.csv')
    if not os.path.exists(path):
        path = require(os.path.join(base_dir, f'{PREDICTIONS_FILE}.csv'))
    return attach_annotations(pd.read_csv(path), study='psilodep1',
                              columns=BASELINE_COVARIATES)


def correlation_row(results, ycol='prediction'):
    '''
    The correlations one bar summarises.

    r is the raw agreement with the outcome; each partial_r is what survives after one
    covariate set is removed from both sides; r_with_<score> says how much of the
    prediction is that baseline score. A partial correlation is undefined when the
    predictor is one of its own covariates.
    '''
    row = {'n': len(results),
           'r': results[ycol].corr(results['label']),
           **{f'r_with_{c}': results[ycol].corr(results[c])
              for c in BASELINE_COVARIATES}}
    for column, spec in zip(PARTIAL_COLUMNS, PARTIAL_SPECS):
        row[column] = (np.nan if ycol in spec['covariates'] else
                       partial_correlation(results, ycol, 'label', spec['covariates'])[0])
    return row


def seed_rows(base_dir, suffix):
    '''The same correlations computed within each training seed, one row per seed.'''
    rows = []
    # Filtered to directories: the cached seed_metrics_*.csv summaries sit alongside them.
    for seed_dir in sorted(d for d in glob.glob(os.path.join(base_dir, 'seed_*'))
                           if os.path.isdir(d)):
        path = os.path.join(seed_dir, f'{PREDICTIONS_FILE}{suffix}.csv')
        if not os.path.exists(path):
            path = os.path.join(seed_dir, f'{PREDICTIONS_FILE}.csv')
        if not os.path.exists(path):
            continue
        results = attach_annotations(pd.read_csv(path), study='psilodep1',
                                     columns=BASELINE_COVARIATES)
        rows.append(correlation_row(results))
    return pd.DataFrame(rows)


def null_draws(reference, suffix, num_seeds):
    '''
    Correlations of the null models, one draw per permutation.

    scripts/permutation_null.py retrains graphTRIP on permuted psilodep2 outcomes and
    transfers each null model to psilodep1, so a draw is what this pipeline produces when
    nothing was learned from the primary cohort. Each draw averages one permutation's fold
    models exactly as the observed value does, and is scored against the true psilodep1
    outcomes joined on by subject_id. Permutations with an incomplete seed set are dropped.

    Returns None when the null tree is absent, so the panels still build without it.
    '''
    base = output_dir(*NULL_PARTS)
    if not os.path.exists(base):
        return None

    truth = reference.set_index('subject_id')[
        ['label', *BASELINE_COVARIATES]].sort_index()
    draws = []
    for perm_dir in perm_dirs(base):
        paths = sorted(glob.glob(os.path.join(perm_dir, 'psilodep1', 'seed_*',
                                              f'{PREDICTIONS_FILE}{suffix}.csv')))
        if len(paths) != num_seeds:
            continue
        predictions = pd.concat(
            [pd.read_csv(p).set_index('subject_id')['prediction'] for p in paths],
            axis=1).mean(axis=1)
        draws.append(correlation_row(
            truth.assign(prediction=predictions.reindex(truth.index)).reset_index()))
    return pd.DataFrame(draws) if draws else None


def null_pvalues(draws, row, statistics=STATISTICS):
    '''Two-sided rank p of each observed correlation against the null draws.'''
    if draws is None:
        return {f'{statistic}_p': np.nan for statistic in statistics}
    return {f'{statistic}_p':
            (1 + (draws[statistic].abs() >= abs(row[statistic])).sum()) / (1 + len(draws))
            for statistic in statistics}


def correlation_table(out, num_seeds):
    '''
    One row per model, input mapping and statistic: the ensemble value, its p-value
    against the null models, and the spread over the training seeds.

    Returns:
    -------
        tuple: (table, {condition label: frame}) where each frame holds the ensemble
               values of one condition, ordered by r with the benchmark pinned last.
    '''
    rows, panels = [], {}

    for condition, suffix in CONDITIONS:
        reference = load_zeroshot_results(
            require(output_dir('validation', 'evaluate_graphtrip')), suffix)
        draws = null_draws(reference, suffix, num_seeds)
        if draws is None:
            out.log(f'Null models not found under {output_dir(*NULL_PARTS)}; '
                    f'those p-values are left NaN.')

        ensemble = []
        for parts, label in MODELS:
            results = load_zeroshot_results(require(output_dir(*parts)), suffix)
            row = correlation_row(results)
            per_seed = seed_rows(output_dir(*parts), suffix)
            pvalues = null_pvalues(draws if label == NULL_MODEL else None, row)

            ensemble.append({'model': label, **{k: row[k] for k in STATISTICS}})

            for statistic in STATISTICS:
                values = per_seed[statistic]
                rows.append({'condition': condition, 'model': label,
                             'statistic': statistic, 'n': row['n'],
                             'ensemble': row[statistic],
                             'p_null': pvalues[f'{statistic}_p'],
                             'n_seeds': len(per_seed),
                             'seed_mean': values.mean(),
                             'seed_sd': values.std(ddof=1),
                             'seed_min': values.min(), 'seed_max': values.max()})

        ensemble = pd.DataFrame(ensemble)
        panels[condition] = reorder(ensemble, order_by_r(ensemble)['model'].tolist())

    return pd.DataFrame(rows), panels


def order_by_r(frame):
    '''Highest r on top, with the clinical-only benchmark pinned to the bottom.'''
    pinned = frame[frame['model'] == CLINICAL_ONLY]
    rest = frame[frame['model'] != CLINICAL_ONLY].sort_values('r', ascending=False)
    return pd.concat([rest, pinned], ignore_index=True)


def reorder(frame, order):
    '''Puts frame's rows in the given model order.'''
    return frame.set_index('model').loc[order].reset_index()


def subset(frame, models):
    '''Keeps the given models, in the order the frame already has.'''
    return frame[frame['model'].isin(models)].reset_index(drop=True)


def zeroshot_scatter(results, out, name, pvalue=None):
    '''
    Zero-shot predictions against the observed outcome, with a fitted regression line.

    No identity line: the two cohorts differ in treatment protocol, patient population
    and outcome timepoint, so the predictions are not expected to be calibrated to the
    validation scale, only to covary with it. The axes are square but their limits are
    independent, for the same reason.
    '''
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    regression_scatter(results, xcol='label', ycol='prediction', yerr='prediction_sem',
                       show_ci=False, ax=ax)

    ax.set_box_aspect(1)
    ax.set_xlabel(SCATTER_XLABEL, fontsize=9)
    ax.set_ylabel(SCATTER_YLABEL, fontsize=9)

    title = f"r = {results['prediction'].corr(results['label']):.4f}"
    if pvalue is not None and not np.isnan(pvalue):
        title += f', p = {pvalue:.4f}'
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    save_path = out.fig(name)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def correlation_bar_panel(out, frame, name, partial_col, partial_label):
    '''
    One horizontal bar per model: r as the full bar, partial r nested inside it.

    The nested bar is drawn narrower and in front rather than stacked, because partial r
    is not a component of r. 
    '''
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    positions = np.arange(len(frame))[::-1]

    ax.barh(positions, frame['r'], height=BAR_HEIGHT, color=BAR_COLOR_FULL,
            edgecolor=NEUTRAL2, linewidth=0.6, zorder=2,
            label='r (prediction, QIDS_1week)')
    ax.barh(positions, frame[partial_col], height=BAR_HEIGHT * NESTED_FRACTION,
            color=BAR_COLOR_PARTIAL, zorder=3, label=partial_label)

    ax.axvline(0, color=NEUTRAL2, linewidth=0.8, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(frame['model'])
    ax.set_xlabel('correlation with QIDS_1week')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize='small', loc='lower center',
              bbox_to_anchor=(0.5, -0.55), ncol=2)

    fig.tight_layout()
    save_path = out.fig(name)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def spread_row(values, label, reference_sd):
    '''Spread of one score or prediction, and its SD relative to reference_sd.'''
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    sd = values.std(ddof=1)
    return {'series': label, 'n': len(values), 'mean': values.mean(), 'sd': sd,
            'min': values.min(), 'max': values.max(),
            'range': values.max() - values.min(), 'sd_ratio': sd / reference_sd}


def prediction_spread_table(out):
    '''
    Spread of the training outcome, the validation outcome and the zero-shot predictions.

    sd_ratio is relative to the observed psilodep1 outcome, so it says how much of the
    outcome's spread the predictions reproduce.
    '''
    base_dir = require(output_dir('validation', 'evaluate_graphtrip'))
    predictions = {condition: load_zeroshot_results(base_dir, suffix)
                   for condition, suffix in CONDITIONS}
    observed = predictions[CONDITIONS[0][0]]['label']
    reference_sd = observed.std(ddof=1)

    training = study_annotations('psilodep2')
    training = training[training['Condition_numeric'] == PSILOCYBIN_CONDITION]

    rows = [spread_row(training[TRAINING_TARGET],
                       f'psilodep2 {TRAINING_TARGET} (psilocybin arm)', reference_sd),
            spread_row(observed, 'psilodep1 QIDS_1week (observed)', reference_sd)]
    rows += [spread_row(frame['prediction'],
                        f'graphTRIP zero-shot prediction ({condition})', reference_sd)
             for condition, frame in predictions.items()]

    table = pd.DataFrame(rows)
    out.table('c_prediction_spread', table)
    out.log_df('Spread of the training outcome, the validation outcome and the '
               'zero-shot predictions', table)
    return table


# Permutation importance of the MLP inputs ------------------------------------------------

def node_context(batch):
    '''Replicates data_ingredient.get_context outside a sacred run.'''
    if not hasattr(batch, 'context_attr') or batch.context_attr.shape[1] == 0:
        return torch.empty((batch.num_nodes, 0), dtype=torch.float32, device=batch.x.device)
    return batch.context_attr[batch.batch]


def load_seed_ensemble(seed_dir, data, batch, device):
    '''
    The fold models of one training seed, with the MLP input matrix each of them receives.

    The VGAE never reads graph_attr, so each readout is computed once and reused for both
    input mappings; only the head's forward pass is repeated per permutation.

    Returns:
    -------
        tuple: (heads, {condition label: list of [n_subjects, latent_dim + n_attrs] arrays,
                one per fold model}, latent_dim, graph_attrs)
    '''
    config = load_configs_from_json(os.path.join(seed_dir, 'config.json'))
    weights_dir = config['weights_dir']
    filenames = config['weight_filenames']
    graph_attrs = config['dataset']['graph_attrs']
    source_config = load_configs_from_json(os.path.join(weights_dir, 'config.json'))

    transfer = build_transfer_inputs(
        data=data, weights_dir=weights_dir, num_models=len(filenames['mlp']),
        graph_attrs=graph_attrs, harmonise=config['harmonise_graph_attrs'],
        source_standardised_attrs=config['source_standardised_attrs'],
        source_dataset_config=source_config['dataset'])

    readouts = []
    for vgae in load_vgaes(config['vgae_model'], weights_dir, filenames['vgae']):
        vgae.to(device).eval()
        with torch.no_grad():
            out = vgae(batch)
            readouts.append(vgae.readout(out.mu, node_context(batch),
                                         batch.batch).cpu().numpy())
    latent_dim = readouts[0].shape[1]

    heads = [head.to(device).eval() for head in
             load_mlps(config['mlp_model'], latent_dim, weights_dir, filenames['mlp'])]

    # A model with nothing to harmonise only carries the unharmonised mapping.
    matrices = {}
    for condition, _ in CONDITIONS:
        key = NO_HARMONISATION if condition == CONDITIONS[0][0] else HARMONISED
        clinical = transfer['inputs'].get(key, transfer['inputs'][NO_HARMONISATION])
        matrices[condition] = [np.concatenate([readout, attrs], axis=1)
                               for readout, attrs in zip(readouts, clinical)]

    return heads, matrices, latent_dim, graph_attrs


def ensemble_r(heads, matrices, labels, columns=None, order=None, device=None):
    '''
    r of the mean-voted ensemble prediction, optionally with one block of input columns
    shuffled across patients.

    The same permutation is applied to every fold model. Permuting each independently
    would let the mean vote average the perturbation away.
    '''
    predictions = np.zeros((len(labels), len(heads)))
    for i, (head, matrix) in enumerate(zip(heads, matrices)):
        x = matrix
        if columns is not None:
            x = matrix.copy()
            x[:, columns] = matrix[np.ix_(order, columns)]
        with torch.no_grad():
            predictions[:, i] = head(
                torch.tensor(x, dtype=torch.float32, device=device)
            ).squeeze(-1).cpu().numpy()
    return stats.pearsonr(predictions.mean(axis=1), labels)[0]


def seed_importance(heads, matrices, labels, blocks, rng, device):
    '''Intact r of one seed's ensemble, and the mean drop in r per input block.'''
    baseline = ensemble_r(heads, matrices, labels, device=device)

    rows = []
    for name, columns in blocks:
        permuted = np.array([
            ensemble_r(heads, matrices, labels, columns,
                       rng.permutation(len(labels)), device)
            for _ in range(N_IMPORTANCE_PERMUTATIONS)])
        rows.append({'feature': name, 'baseline_r': baseline,
                     'drop_r': baseline - permuted.mean()})
    return rows


def summarise_importance(per_seed):
    '''
    One row per input mapping and block: the drop in r across training seeds, its SEM, a
    one-sample t-test against zero, and the drop as a percentage of the intact r.
    '''
    rows = []
    for (condition, feature), group in per_seed.groupby(['condition', 'feature'],
                                                        sort=False):
        drops = group['drop_r'].to_numpy(dtype=float)
        baseline = group['baseline_r'].mean()
        rows.append({'condition': condition, 'feature': feature, 'n_seeds': len(drops),
                     'baseline_r': baseline, 'drop_r': drops.mean(),
                     'drop_r_sem': stats.sem(drops),
                     'p': stats.ttest_1samp(drops, 0).pvalue,
                     'percent_of_baseline': 100 * drops.mean() / baseline})
    return pd.DataFrame(rows)


def importance_analysis(out, results_base_dir, data, num_seeds, seed, device=None):
    '''
    Permutation importance of graphTRIP's inputs under both input mappings, computed
    within each training seed and aggregated across them.
    '''
    device = device or get_device()
    batch = next(iter(DataLoader(data, batch_size=len(data), shuffle=False))).to(device)
    labels = batch.y.squeeze(-1).cpu().numpy()

    rng = np.random.default_rng(seed)
    rows = []
    for seed_index in range(num_seeds):
        heads, matrices, latent_dim, graph_attrs = load_seed_ensemble(
            os.path.join(results_base_dir, f'seed_{seed_index}'), data, batch, device)

        blocks = [(LATENT_BLOCK, np.arange(latent_dim))]
        blocks += [(attr, np.array([latent_dim + graph_attrs.index(attr)]))
                   for attr in IMPORTANCE_ATTRS]

        for condition, _ in CONDITIONS:
            rows += [{'condition': condition, 'seed': seed_index, **row}
                     for row in seed_importance(heads, matrices[condition], labels,
                                                blocks, rng, device)]

    table = summarise_importance(pd.DataFrame(rows))
    out.table('e_permutation_importance', table)
    out.log_df(f'Permutation importance of the MLP inputs '
               f'({N_IMPORTANCE_PERMUTATIONS} shuffles per seed)', table)
    return table


# Sensitivity of the reported correlations ------------------------------------------------

def jackknife_range(results, ycol):
    '''
    Recomputes r and every partial r on each leave-one-patient-out subset.

    Returns:
    -------
        dict: min, max and sign count of each statistic over the n subsets.
    '''
    values = {statistic: [] for statistic in ('r', *PARTIAL_COLUMNS)}
    for drop in range(len(results)):
        subset = results.drop(results.index[drop])
        row = correlation_row(subset, ycol=ycol)
        for statistic in values:
            values[statistic].append(row[statistic])

    summary = {'n_subsets': len(results)}
    for statistic, series in values.items():
        series = np.asarray(series, dtype=float)
        if np.isnan(series).all():
            summary.update({f'{statistic}_jk_min': np.nan, f'{statistic}_jk_max': np.nan,
                            f'{statistic}_jk_positive': np.nan})
            continue
        summary.update({f'{statistic}_jk_min': series.min(),
                        f'{statistic}_jk_max': series.max(),
                        f'{statistic}_jk_positive': int((series > 0).sum())})
    return summary


def rank_statistics(results, ycol):
    '''Spearman rho of the predictor with the outcome, raw and given each covariate set.'''
    rho, pval = stats.spearmanr(results[ycol], results['label'])
    row = {'spearman_rho': rho, 'spearman_p': pval}
    for spec in PARTIAL_SPECS:
        column = f"partial_spearman_rho{spec['suffix']}"
        if ycol in spec['covariates']:
            row[column] = np.nan
            continue
        ranked = pd.DataFrame({c: results[c].rank()
                               for c in (ycol, 'label', *spec['covariates'])})
        row[column] = partial_correlation(ranked, ycol, 'label', spec['covariates'])[0]
    return row


def sensitivity_table(out):
    '''
    Leave-one-patient-out ranges and rank correlations for the reported predictors.

    Covers the two models in the bar panel plus QIDS_Before used directly as a
    zero-parameter predictor, which is condition-independent and therefore reported once.
    '''
    directories = {label: output_dir(*parts) for parts, label in MODELS}
    rows = []

    for condition, suffix in CONDITIONS:
        for label in SENSITIVITY_MODELS:
            results = load_zeroshot_results(require(directories[label]), suffix)
            rows.append({'model': label, 'condition': condition,
                         **correlation_row(results),
                         **jackknife_range(results, 'prediction'),
                         **rank_statistics(results, 'prediction')})

    results = load_zeroshot_results(
        require(output_dir('validation', 'evaluate_graphtrip')), CONDITIONS[0][1])
    rows.append({'model': BASELINE_PREDICTOR, 'condition': 'n/a',
                 **correlation_row(results, ycol='QIDS_Before'),
                 **jackknife_range(results, 'QIDS_Before'),
                 **rank_statistics(results, 'QIDS_Before')})

    table = pd.DataFrame(rows)
    out.table('e_sensitivity', table)
    out.log_df('Leave-one-patient-out ranges and rank correlations', table)
    return table


@register('fig4', group='main', subdir='Fig.4')
def fig4_validation(ctx, out):
    results_base_dir = require(output_dir('validation', 'evaluate_graphtrip'))
    main_condition = CONDITIONS[0][0]

    # b. Baseline severity of the two cohorts --------------------------------------------
    baseline_tests = baseline_severity_panels(
        out, 'b_baseline_severity', np.random.default_rng(ctx.cfg.seed))
    out.table('b_baseline_severity_tests', baseline_tests)
    out.log_df('Baseline severity, psilodep2 psilocybin arm versus psilodep1 '
               "(Welch's t-tests, uncorrected)", baseline_tests)

    # d. Reconstruction performance on the validation dataset ----------------------------
    psilodep1_config = load_ingredient_configs(os.path.join(results_base_dir, 'seed_0'),
                                               ingredients=['dataset'])
    psilodep1_data = load_dataset(psilodep1_config['dataset'])

    # All psilodep1 patients were treated with psilocybin
    psilodep1_conditions = np.ones(len(psilodep1_data))

    # Every VGAE of every seed and fold reconstructs every patient, then averages, and
    # the result is tested against the primary dataset of Fig. 2d
    psilodep1_x = reconstruction_comparison(ctx, out, psilodep1_data)

    plot_correlation_boxplot(out, psilodep1_x, psilodep1_conditions,
                             'd_reconstruction_corrs')

    # c, e. Zero-shot prediction and the input-domain contrast ---------------------------
    table, panels = correlation_table(out, ctx.num_seeds)
    out.table('e_correlations', table)
    out.log_df('Zero-shot correlations by model, input mapping and statistic', table)

    r_pvalues = table[(table['model'] == GRAPHTRIP)
                      & (table['statistic'] == 'r')].set_index('condition')['p_null']
    for condition, suffix in CONDITIONS:
        tag = '' if condition == main_condition else '_harmonised'
        zeroshot_scatter(
            aggregate_prediction_results(results_file=os.path.join(
                results_base_dir, f'{PREDICTIONS_FILE}{suffix}.csv')),
            out, f'c_true_vs_pred{tag}', r_pvalues[condition])
        frame = subset(panels[condition], PANEL_MODELS)
        for column, spec in zip(PARTIAL_COLUMNS, PARTIAL_SPECS):
            correlation_bar_panel(out, frame, f"e_bars{spec['suffix']}{tag}",
                                  partial_col=column, partial_label=spec['legend'])

    prediction_spread_table(out)
    importance_analysis(out, results_base_dir, psilodep1_data, ctx.num_seeds,
                        ctx.cfg.seed, ctx.device)
    sensitivity_table(out)
