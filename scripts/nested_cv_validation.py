"""
Validates the GRAIL biomarker selection on patients it never saw.

Two statistics, both pooled over all 42 out-of-fold patients:

  ridge        the selected biomarkers are fitted on the inner patients and predict the
               held-out ones. Calibrated against random candidate sets of the same size.
  directional  each model's training alignments against the held-out correlations of the
               same biomarkers.

Dependencies:
- outputs/graphtrip/nested_cv/outer_*/grail/seed_*/sub_*/
- outputs/graphtrip/nested_cv/outer_*/weights/seed_*/nested_split.csv
- outputs/graphtrip/test_biomarkers/feature_values.csv

Outputs:
- outputs/graphtrip/nested_cv/analysis/

Author: Hanna M. Tolle
Date: 2026-09-12
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binom, pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection

from utils.files import add_project_root
from utils.statsalg import get_fold_performance, compute_permutation_stats


NESTED_DIR = 'outputs/graphtrip/nested_cv/'
BIOMARKER_VALUES = 'outputs/graphtrip/test_biomarkers/feature_values.csv'

# The published graphTRIP set: the biomarkers whose category graphTRIP alone defines. Used
# as a reference row only -- it was chosen on the whole cohort, so it is not out of fold.
PUBLISHED_SET = ['x5-HTT_corr_NAT', 'x5-HTT_mean_DMN', 'x5-HT2A_mean_SMN',
                 'x5-HTT_corr_VAChT', 'fc_mean_VIS', 'modularity_rsn',
                 'fc_mean_SMN', 'fc_corr_5-HT2A']

ALPHA_PER_RUN = 0.05    # the per-model FDR the spin test used, i.e. the vote's null rate
ALPHA = 0.05            # significance of the aggregated, per-patient FDR
MIN_RHO = 0.0           # inner fold models below this are droppedrun
RIDGE_ALPHAS = np.logspace(-6, 6, 13)


# Loading the nested GRAIL tree ------------------------------------------------------------

def _outer_dirs(nested_dir):
    '''The outer fold directories that have GRAIL results, by outer fold index.'''
    dirs = {}
    for path in sorted(glob.glob(os.path.join(nested_dir, 'outer_*'))):
        if glob.glob(os.path.join(path, 'grail', 'seed_*', 'sub_*')):
            dirs[int(os.path.basename(path).split('_')[1])] = path
    if not dirs:
        raise FileNotFoundError(f'No GRAIL results under {nested_dir}/outer_*/grail/')
    return dirs


def load_outer_fold(outer_dir, refresh=False):
    '''
    One outer fold's GRAIL results.

    Returns a dict with alignments and spin-test selections as [seed, fold, subject,
    biomarker] over the full cohort, the inner fold models' test rho as [seed, fold], the
    biomarker names, and the nested split. Cached, because the tree is ~10k small CSVs.
    '''
    cache = os.path.join(outer_dir, 'grail', 'cache.npz')
    split = pd.read_csv(os.path.join(outer_dir, 'weights', 'seed_0', 'nested_split.csv'))

    if os.path.exists(cache) and not refresh:
        stored = np.load(cache, allow_pickle=True)
        return {'align': stored['align'], 'sel': stored['sel'], 'rho': stored['rho'],
                'feat': list(stored['feat']), 'seeds': list(stored['seeds']),
                'split': split}

    seed_dirs = sorted(glob.glob(os.path.join(outer_dir, 'grail', 'seed_*')),
                       key=lambda d: int(os.path.basename(d).split('_')[1]))
    seeds = [int(os.path.basename(d).split('_')[1]) for d in seed_dirs]
    n_sub = len(glob.glob(os.path.join(seed_dirs[0], 'sub_*')))
    n_fold = len(glob.glob(os.path.join(seed_dirs[0], 'sub_0', 'k*_mean_alignments.csv')))

    feat, align, sel, rho = None, [], [], []
    for seed, seed_dir in zip(seeds, seed_dirs):
        # The fold models' performance on their own inner test folds
        weights_dir = os.path.join(outer_dir, 'weights', f'seed_{seed}')
        performance = get_fold_performance(weights_dir).sort_values('fold')
        rho.append(performance['rho'].values)

        seed_align, seed_sel = [], []
        for k in range(n_fold):
            fold_align, fold_sel = [], []
            for sub in range(n_sub):
                sub_dir = os.path.join(seed_dir, f'sub_{sub}')
                means = pd.read_csv(os.path.join(sub_dir, f'k{k}_mean_alignments.csv'))
                if feat is None:
                    feat = list(means.columns)
                fold_align.append(means[feat].values[0])
                fold_sel.append(pd.read_csv(
                    os.path.join(sub_dir, f'k{k}_selected_features.csv'))[feat].values[0])
            seed_align.append(fold_align)
            seed_sel.append(fold_sel)
        align.append(seed_align)
        sel.append(seed_sel)

    result = {'align': np.array(align, dtype=float), 'sel': np.array(sel, dtype=float),
              'rho': np.array(rho, dtype=float), 'feat': feat, 'seeds': seeds,
              'split': split}
    np.savez_compressed(cache, align=result['align'], sel=result['sel'],
                        rho=result['rho'], feat=np.array(feat), seeds=np.array(seeds))
    return result


# The selection chain, on the inner patients only -------------------------------------------

def select_biomarkers(fold, rule='half', top_n=10, min_rho=MIN_RHO):
    '''
    Reruns the published graphTRIP selection on one outer fold's inner patients.

    The chain is the same as scripts/posthoc.py: drop fold models that failed to predict,
    count how many models called each (patient, biomarker) significant in the spin test,
    turn the counts into a binomial p, FDR-correct across biomarkers within a patient, and
    take the direction from the performance-weighted mean alignment. What differs is that
    only inner patients and inner models contribute, and that there is no Medusa tree, so
    a biomarker's category is just the sign it is given.

    rule 'half' is the published rule of load_biomarker_categories: keep a biomarker that is
    n.s. in at most int(0.5 * n) inner patients, counting significance in either direction.
    'top_n' takes the n most often significant, which always returns a set.
    '''
    split = fold['split']
    inner = split[split['role'] == 'inner']['position'].values
    feat = fold['feat']

    # Fold models that predicted their own inner test fold
    keep = fold['rho'] > min_rho
    if not keep.any():
        raise ValueError('No inner fold model passes the rho filter.')
    align = fold['align'][keep]            # [model, subject, biomarker]
    sel = fold['sel'][keep]
    rho = fold['rho'][keep]
    n_models = len(rho)

    # The binomial vote, per inner patient, then FDR across biomarkers within that patient
    votes = sel[:, inner, :].sum(axis=0)                       # [inner patient, biomarker]
    pvals = binom.sf(votes - 1, n_models, ALPHA_PER_RUN)
    qvals = np.array([fdrcorrection(row, alpha=ALPHA)[1] for row in pvals])
    significant = qvals < ALPHA

    # The direction comes from the performance-weighted mean alignment
    weights = rho / (rho.sum() + 1e-6)
    weighted_mean = np.tensordot(weights, align[:, inner, :], axes=(0, 0))
    indicator = np.sign(weighted_mean) * significant            # in {-1, 0, +1}

    # How often each biomarker is called, and in which direction
    frac_significant = significant.mean(axis=0)
    n_positive = (indicator > 0).sum(axis=0)
    n_negative = (indicator < 0).sum(axis=0)
    majority_sign = np.where(n_positive >= n_negative, 1, -1)
    frac_majority = np.maximum(n_positive, n_negative)/len(inner)

    table = pd.DataFrame({'biomarker': feat,
                          'frac_significant': frac_significant,
                          'frac_majority_sign': frac_majority,
                          'majority_sign': majority_sign,
                          'mean_weighted_alignment': weighted_mean.mean(axis=0),
                          'n_inner_patients': len(inner),
                          'n_models': n_models})

    if rule == 'half':
        # The published rule: n.s. in at most int(0.5 * n) patients, in either direction
        n_not_significant = len(inner) - significant.sum(axis=0)
        chosen = n_not_significant <= int(0.5*len(inner))
    elif rule == 'top_n':
        order = (table.assign(abs_alignment=table['mean_weighted_alignment'].abs())
                 .sort_values(['frac_significant', 'abs_alignment'],
                              ascending=[False, False]).index[:top_n])
        chosen = table.index.isin(order)
    else:
        raise ValueError(f'Unknown selection rule: {rule}')

    table['selected'] = chosen
    return table


# Statistic A: ridge prediction of the held-out patients -------------------------------------

def _ridge_predict(values, biomarkers, inner, outer):
    '''Fits a ridge on the inner patients and predicts the held-out ones.'''
    X = values[biomarkers].values
    y = values['y'].values.astype(float)
    scaler = StandardScaler().fit(X[inner])
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(scaler.transform(X[inner]), y[inner])
    return model.predict(scaler.transform(X[outer]))


def _pooled_r(values, sets_by_fold, splits):
    '''Pools the held-out predictions of every outer fold and correlates them with outcome.'''
    prediction = np.full(len(values), np.nan)
    for outer_fold, biomarkers in sets_by_fold.items():
        inner, outer = splits[outer_fold]
        prediction[outer] = _ridge_predict(values, biomarkers, inner, outer)
    observed = ~np.isnan(prediction)
    y = values['y'].values.astype(float)
    return prediction, pearsonr(y[observed], prediction[observed])[0], observed


def ridge_statistic(values, feat, sets_by_fold, splits, n_null=1000, seed=0):
    '''
    Do the selected biomarkers predict the patients they were not selected on?

    The null draws random candidate sets of the same size.
    '''
    prediction, observed_r, mask = _pooled_r(values, sets_by_fold, splits)
    y = values['y'].values.astype(float)
    rho_observed = spearmanr(y[mask], prediction[mask])[0]
    common = {'spearman': rho_observed, 'n_patients': int(mask.sum()),
              'set_sizes': ','.join(str(len(s)) for s in sets_by_fold.values())}

    # A set that is already every candidate has nothing to resample against
    if all(len(biomarkers) == len(feat) for biomarkers in sets_by_fold.values()):
        return ({'observed': observed_r, 'p_value': np.nan, 'null_mean': np.nan,
                 'null_std': np.nan, 'z_score': np.nan, **common}, prediction,
                np.array([]))

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        draw = {fold: list(rng.choice(feat, size=len(biomarkers), replace=False))
                for fold, biomarkers in sets_by_fold.items()}
        null.append(_pooled_r(values, draw, splits)[1])
    null = np.array(null)

    stats = compute_permutation_stats(observed_r, null, alternative='greater')
    stats.update(common)
    return stats, prediction, null


# Statistic B: the learned direction against the held-out correlations ------------------------

def _corr_cols(X, y):
    '''Pearson correlation of every column of X with y.'''
    Xz = (X - X.mean(0))/(X.std(0) + 1e-12)
    yz = (y - y.mean())/(y.std() + 1e-12)
    return Xz.T @ yz/len(y)


def directional_statistic(fold, values, feat, selected, outer_fold, min_rho=MIN_RHO):
    '''
    Per inner fold model: does its training alignment point the way the held-out patients do?

    signed_r is the mean of sign(alignment) x held-out correlation, which the exploratory
    analysis found more stable than the Pearson version for small biomarker sets;
    The held-out correlations are the same for every model of this outer fold; only the alignments differ.
    '''
    split = fold['split']
    inner = split[split['role'] == 'inner']['position'].values
    outer = split[split['role'] == 'outer']['position'].values
    inner_fold = split.set_index('position')['inner_fold']

    X = values[feat].values
    y = values['y'].values.astype(float)
    held_out_r = _corr_cols(X[outer], y[outer])
    groups = {'identified': np.isin(feat, selected), 'other': ~np.isin(feat, selected)}

    rows = []
    for s_idx, seed in enumerate(fold['seeds']):
        for k in range(fold['align'].shape[1]):
            if not fold['rho'][s_idx, k] > min_rho:
                continue
            train = inner[inner_fold.loc[inner].values != k]
            alignment = fold['align'][s_idx, k, train, :].mean(axis=0)

            row = {'outer_fold': outer_fold, 'seed': seed, 'fold': k,
                   'rho': fold['rho'][s_idx, k], 'n_train': len(train),
                   'n_held_out': len(outer)}
            for name, mask in groups.items():
                if mask.sum() < 2:
                    row[f'{name}_signed_r'] = np.nan
                    row[f'{name}_pearson'] = np.nan
                    continue
                row[f'{name}_signed_r'] = float(
                    np.mean(np.sign(alignment[mask])*held_out_r[mask]))
                row[f'{name}_pearson'] = float(
                    np.corrcoef(alignment[mask], held_out_r[mask])[0, 1])
            row['n_identified'] = int(groups['identified'].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_wilcoxon(per_fold, a, b):
    '''One-sided Wilcoxon for identified > other, paired by outer fold.'''
    values = per_fold[[a, b]].dropna()
    if len(values) < 5 or np.allclose(values[a], values[b]):
        return np.nan
    return float(wilcoxon(values[a], values[b], alternative='greater').pvalue)


# Main -----------------------------------------------------------------------------------

def main(nested_dir, rule, top_n, n_null, min_rho, refresh, seed):
    nested_dir = add_project_root(nested_dir)
    analysis_dir = os.path.join(nested_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    values = pd.read_csv(add_project_root(BIOMARKER_VALUES)).sort_values('sub')
    outer_dirs = _outer_dirs(nested_dir)
    log = []

    def report(message=''):
        print(message)
        log.append(message)

    report(f'Nested-CV biomarker validation over {len(outer_dirs)} outer fold(s): '
           f'{sorted(outer_dirs)}')
    report(f'Selection rule: {rule}' + (f' (n = {top_n})' if rule == 'top_n' else '')
           + f'; inner fold models kept at rho > {min_rho}.')
    report()

    # Selection, one set per outer fold ----------------------------------------------
    folds, tables, sets_by_fold, splits = {}, [], {}, {}
    for outer_fold, outer_dir in outer_dirs.items():
        fold = load_outer_fold(outer_dir, refresh=refresh)
        folds[outer_fold] = fold

        table = select_biomarkers(fold, rule=rule, top_n=top_n, min_rho=min_rho)
        table.insert(0, 'outer_fold', outer_fold)
        tables.append(table)

        selected = table.loc[table['selected'], 'biomarker'].tolist()
        sets_by_fold[outer_fold] = selected
        split = fold['split']
        splits[outer_fold] = (split[split['role'] == 'inner']['position'].values,
                              split[split['role'] == 'outer']['position'].values)

        overlap = sorted(set(selected) & set(PUBLISHED_SET))
        report(f'Outer fold {outer_fold}: {len(selected)} biomarkers selected from '
               f"{len(fold['feat'])} candidates, {len(overlap)}/{len(PUBLISHED_SET)} of the "
               f'published set among them.')
        report(f'  {", ".join(selected) if selected else "(empty)"}')

    tables = pd.concat(tables, ignore_index=True)
    tables.to_csv(os.path.join(analysis_dir, 'nested_cv_selection.csv'), index=False)
    report()

    empty = [f for f, s in sets_by_fold.items() if len(s) < 2]
    if empty:
        report(f'Outer fold(s) {empty} selected fewer than 2 biomarkers, so neither '
               f'statistic can be computed. This is the selection rule losing sensitivity '
               f'at the reduced inner sample size: rerun with --rule top_n, which always '
               f'returns a set.')
        with open(os.path.join(analysis_dir, 'nested_cv_validation.txt'), 'w') as handle:
            handle.write('\n'.join(str(line) for line in log) + '\n')
        print(f'Wrote the selection table to {analysis_dir}')
        return

    feat = folds[min(folds)]['feat']

    # Statistic A ---------------------------------------------------------------------
    ridge_rows = []
    reference = {'selected (out of fold)': sets_by_fold,
                 'all candidates': {f: feat for f in sets_by_fold},
                 'published set (not out of fold)': {f: PUBLISHED_SET for f in sets_by_fold}}
    for name, sets in reference.items():
        stats, prediction, null = ridge_statistic(values, feat, sets, splits,
                                                  n_null=n_null, seed=seed)
        ridge_rows.append({'set': name, **stats})
        if name.startswith('selected'):
            np.save(os.path.join(analysis_dir, 'nested_cv_ridge_null.npy'), null)
            held_out = pd.DataFrame({'sub': values['sub'].values,
                                     'label': values['y'].values,
                                     'prediction': prediction})
            held_out.to_csv(os.path.join(analysis_dir, 'nested_cv_ridge_predictions.csv'),
                            index=False)

    ridge = pd.DataFrame(ridge_rows)
    ridge.to_csv(os.path.join(analysis_dir, 'nested_cv_ridge.csv'), index=False)
    report('Ridge on the held-out patients, pooled over outer folds')
    report(ridge.round(4).to_string(index=False))
    report(f'The null draws {n_null} random candidate sets of the same size per outer fold.')
    report()

    # Statistic B ---------------------------------------------------------------------
    per_model = pd.concat([directional_statistic(folds[f], values, feat,
                                                 sets_by_fold[f], f, min_rho=min_rho)
                           for f in sorted(folds)], ignore_index=True)
    per_model.to_csv(os.path.join(analysis_dir, 'nested_cv_directional_per_model.csv'),
                     index=False)

    # Seeds of one outer fold share its held-out patients, so collapse to one value per
    # outer fold before testing.
    stat_cols = [c for c in per_model.columns
                 if c.endswith('_signed_r') or c.endswith('_pearson')]
    per_seed = per_model.groupby(['outer_fold', 'seed'])[stat_cols].mean().reset_index()
    per_fold = per_seed.groupby('outer_fold')[stat_cols].mean().reset_index()
    per_fold.to_csv(os.path.join(analysis_dir, 'nested_cv_directional_per_fold.csv'),
                    index=False)

    summary = []
    for statistic in ('signed_r', 'pearson'):
        a, b = f'identified_{statistic}', f'other_{statistic}'
        summary.append({'statistic': statistic,
                        'n_outer_folds': len(per_fold),
                        'identified_mean': per_fold[a].mean(),
                        'other_mean': per_fold[b].mean(),
                        'identified_minus_other': (per_fold[a] - per_fold[b]).mean(),
                        'n_folds_identified_greater': int((per_fold[a] > per_fold[b]).sum()),
                        'p_identified_gt_other': _paired_wilcoxon(per_fold, a, b)})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(analysis_dir, 'nested_cv_directional_summary.csv'),
                   index=False)
    report('Held-out direction, one value per outer fold (seeds averaged first)')
    report(summary.round(4).to_string(index=False))
    report(f'Paired one-sided Wilcoxon over {len(per_fold)} outer folds needs 6 to reach '
           f'p < 0.05.')
    report()

    # Selection stability ---------------------------------------------------------------
    if len(sets_by_fold) > 1:
        counts = pd.Series([b for s in sets_by_fold.values() for b in s]).value_counts()
        stability = pd.DataFrame({'biomarker': counts.index,
                                  'n_outer_folds': counts.values,
                                  'frac_outer_folds': counts.values/len(sets_by_fold)})
        stability['in_published_set'] = stability['biomarker'].isin(PUBLISHED_SET)
        stability.to_csv(os.path.join(analysis_dir, 'nested_cv_stability.csv'), index=False)

        jaccard = []
        for i in sorted(sets_by_fold):
            for j in sorted(sets_by_fold):
                if i < j:
                    a, b = set(sets_by_fold[i]), set(sets_by_fold[j])
                    jaccard.append(len(a & b)/len(a | b) if a | b else np.nan)
        report(f'Selection stability: mean pairwise Jaccard {np.nanmean(jaccard):.3f} '
               f'over {len(jaccard)} outer fold pairs.')
        report(f'{int((stability["n_outer_folds"] == len(sets_by_fold)).sum())} biomarkers '
               f'are selected in every outer fold.')
        report(stability.head(15).round(3).to_string(index=False))
        report()

    with open(os.path.join(analysis_dir, 'nested_cv_validation.txt'), 'w') as handle:
        handle.write('\n'.join(str(line) for line in log) + '\n')
    print(f'Wrote tables to {analysis_dir}')


if __name__ == "__main__":
    """
    How to run:
    python -m scripts.nested_cv_validation
    python -m scripts.nested_cv_validation --rule top_n --top_n 10
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--nested_dir', type=str, default=NESTED_DIR,
                        help='Directory with the nested CV outputs')
    parser.add_argument('--rule', type=str, default='half', choices=['half', 'top_n'],
                        help='Selection rule applied to the inner patients')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of biomarkers to keep when --rule top_n')
    parser.add_argument('--n_null', type=int, default=1000,
                        help='Random candidate sets drawn for the ridge null')
    parser.add_argument('--min_rho', type=float, default=MIN_RHO,
                        help='Inner fold models below this test rho are dropped')
    parser.add_argument('--refresh', action='store_true',
                        help='Re-read the GRAIL CSVs instead of the cache')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed of the null draws')
    args = parser.parse_args()

    main(args.nested_dir, args.rule, args.top_n, args.n_null, args.min_rho,
         args.refresh, args.seed)
