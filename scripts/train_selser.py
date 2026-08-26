'''
SELSER-fMRI: SELSER-like baseline trained on fMRI BOLD covariance.

Adapts the SELSER algorithm (Wu et al., 2020, Nature Biotechnology) to fMRI.
SELSER learns a low-rank weight matrix W such that:

    ŷ_i = Tr(W^T C_i) + b

where C_i = X_i X_i^T / N is the (trace-normalised) spatial covariance of the
BOLD timeseries for subject i, and ||W||_* (nuclear norm) is penalised to enforce
a low-rank / sparse-filter solution.

This is equivalent to nuclear-norm-regularised linear regression on the FC matrix
entries, expressed in the SELSER factorised form. After training, W is eigen-
decomposed to recover L interpretable spatial filters w_k and regression weights β_k:

    W = sum_k β_k w_k w_k^T   (eigendecomposition)

Two separate models are fit — one per treatment arm — mirroring the original
SELSER paper. Cross-validation structure matches Medusa-graphTRIP (7-fold,
stratified by treatment arm).

Authors: Hanna M. Tolle
Date: 2025-04-02
'''
import sys
sys.path.append('../')

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from utils.helpers import permute_labels
from utils.plotting import true_vs_pred_scatter


# --- Covariance loading -------------------------------------------------------

def load_bold_covariance(subject_dir):
    '''
    Load BOLD timeseries and return the trace-normalised spatial covariance.

    Parameters:
    ----------
    subject_dir (str): path to directory containing bold.csv

    Returns:
    -------
    C_i (np.ndarray): shape (C, C), trace-normalised covariance matrix
    '''
    bold = np.loadtxt(os.path.join(subject_dir, 'bold.csv'), delimiter=',')
    # bold.csv: rows = timepoints, cols = ROIs -> transpose to (C, N)
    X = bold.T
    C_i = X @ X.T / X.shape[1]
    C_i /= np.trace(C_i)   # normalise as in Wu et al.
    return C_i


def load_dataset(data_dir, annotations_file, target, graph_attrs=None):
    '''
    Load covariance matrices and labels for all non-excluded subjects.

    Parameters:
    ----------
    graph_attrs : list of str or None — annotation columns to load as auxiliary
                  covariates (in addition to the BOLD covariance)

    Returns:
    -------
    covs        : (M, C, C) array
    labels      : (M,) array of target values
    subject_ids : (M,) int array — "Subject" column values (0-indexed)
    conditions  : (M,) int array — 1 for psilocybin (P), -1 for escitalopram (E)
    G           : (M, num_attrs) array, or None if graph_attrs is empty
    '''
    ann = pd.read_csv(annotations_file)
    ann = ann[ann['Exclusion'] == 0].reset_index(drop=True)

    covs, labels, subject_ids, conditions = [], [], [], []
    graph_attr_rows = [] if graph_attrs else None
    missing = []

    for _, row in ann.iterrows():
        patient_id = int(row['Patient'])   # 1-indexed; dirs are S{patient_id:02d}
        subject_id = int(row['Subject'])   # 0-indexed; used in prediction_results.csv
        subject_dir = os.path.join(data_dir, f'S{patient_id:02d}')

        if not os.path.exists(os.path.join(subject_dir, 'bold.csv')):
            missing.append(subject_dir)
            continue

        covs.append(load_bold_covariance(subject_dir))
        labels.append(float(row[target]))
        subject_ids.append(subject_id)
        conditions.append(1 if row['Condition'] == 'P' else -1)
        if graph_attrs:
            graph_attr_rows.append([float(row[a]) for a in graph_attrs])

    if missing:
        print(f'Warning: missing bold.csv for {len(missing)} subjects: {missing}')

    G = np.array(graph_attr_rows) if graph_attrs else None
    return (np.array(covs),
            np.array(labels),
            np.array(subject_ids, dtype=int),
            np.array(conditions, dtype=int),
            G)


# --- SELSER optimisation ------------------------------------------------------

def _nuclear_norm_prox(W, threshold):
    '''Proximal operator for ||W||_*: soft-threshold singular values.'''
    U, sigma, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ np.diag(np.maximum(sigma - threshold, 0.0)) @ Vt


def fit_selser(covs, y, lambda_reg, G=None, max_iter=3000, tol=1e-8):
    '''
    Fit SELSER via proximal gradient descent.

        min_{W, gamma, b}  sum_i (Tr(W^T C_i) + gamma^T g_i + b - y_i)^2
                           + lambda_reg * ||W||_*

    Nuclear norm regularisation applies only to W.  gamma and b are solved
    analytically at each iteration via block coordinate descent (exact
    minimisation over the unregularised linear parameters given W).

    The gradient of the smooth part w.r.t. W is Lipschitz with constant
    L = 2 * sigma_max(X_mat)^2,  where X_mat[i] = vec(C_i).
    Step size = 1/L guarantees convergence.

    Parameters:
    ----------
    covs       : (n, C, C) array of trace-normalised covariance matrices
    y          : (n,) target values
    lambda_reg : nuclear norm regularisation strength
    G          : (n, num_attrs) array of auxiliary covariates, or None
    max_iter   : maximum proximal gradient iterations
    tol        : convergence threshold on ||ΔW||_F

    Returns:
    -------
    W      : (C, C) learned (symmetric) weight matrix
    b      : scalar bias
    gamma  : (num_attrs,) covariate weights, or None if G is None
    n_iter : number of iterations until convergence
    '''
    n, C, _ = covs.shape
    X_mat = covs.reshape(n, C * C)   # (n, C*C) — vectorised covariances

    # Build the unregularised design matrix Z = [G | 1] for analytical updates
    ones = np.ones((n, 1))
    Z = np.hstack([G, ones]) if G is not None else ones   # (n, num_attrs+1) or (n, 1)
    ZtZ_inv_Zt = np.linalg.pinv(Z)                        # (num_attrs+1, n)

    # Lipschitz constant of squared-loss gradient w.r.t. W: 2 * lambda_max(X X^T)
    XXT = X_mat @ X_mat.T           # (n, n)
    lambda_max = np.linalg.eigvalsh(XXT).max()
    L = 2.0 * lambda_max
    step = 1.0 / L

    W = np.zeros((C, C))

    for n_iter in range(1, max_iter + 1):
        # Analytical block update: solve [gamma; b] = argmin ||r_W - Z z||^2
        r_W = y - X_mat @ W.ravel()
        z_params = ZtZ_inv_Zt @ r_W                       # (num_attrs+1,) or (1,)

        # residuals = ŷ - y = X_mat @ w + Z @ z - y = Z @ z - r_W
        residuals = Z @ z_params - r_W                    # (n,)
        grad_W = 2.0 * (X_mat.T @ residuals).reshape(C, C)

        W_new = _nuclear_norm_prox(W - step * grad_W, lambda_reg * step)
        W_new = (W_new + W_new.T) / 2.0                   # enforce symmetry

        delta = np.linalg.norm(W_new - W, 'fro')
        W = W_new

        if delta < tol and n_iter > 20:
            break

    # Final analytical update
    r_W = y - X_mat @ W.ravel()
    z_params = ZtZ_inv_Zt @ r_W
    if G is not None:
        gamma = z_params[:-1]
        b = float(z_params[-1])
    else:
        gamma = None
        b = float(z_params[0])

    return W, b, gamma, n_iter


def predict_selser(W, b, covs, gamma=None, G=None):
    '''Predict: ŷ_i = Tr(W^T C_i) + gamma^T g_i + b.'''
    n, C, _ = covs.shape
    pred = covs.reshape(n, C * C) @ W.ravel() + b
    if gamma is not None and G is not None:
        pred += G @ gamma
    return pred


def extract_filters(W, num_filters):
    '''
    Recover spatial filters and regression weights via eigendecomposition of W.
    Returns the top-L eigenvectors (filters) sorted by |eigenvalue|.

    Returns:
    -------
    filters : (L, C) array — spatial filters as rows
    weights : (L,) array — corresponding regression weights (eigenvalues)
    '''
    eigenvalues, eigenvectors = np.linalg.eigh(W)    # W is symmetric
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx][:num_filters]
    eigenvectors = eigenvectors[:, idx][:, :num_filters]
    return eigenvectors.T, eigenvalues                # (L, C), (L,)


# --- Evaluation and plotting --------------------------------------------------

def compute_metrics(labels, predictions):
    r, p = stats.pearsonr(labels, predictions)
    mae = np.abs(labels - predictions).mean()
    mae_std = np.abs(labels - predictions).std()
    rmse = np.sqrt(((labels - predictions) ** 2).mean())
    return {'r': float(r), 'p': float(p),
            'mae': float(mae), 'mae_std': float(mae_std),
            'rmse': float(rmse)}


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SELSER-fMRI baseline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--perm_seed', type=int, default=None,
                        help='Seed of the label permutation, for the permutation null. '
                             'Matches the perm_seed of scripts/permutation_null.py, so the '
                             'nulls of SELSER and graphTRIP are paired.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_filters', action='store_true', default=False,
                        help='Save W, filters, and filter_weights per fold')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    data_dir         = cfg.get('data_dir', 'data/raw/psilodep2/before/schaefer100')
    annotations_file = cfg.get('annotations_file', 'data/raw/psilodep2/annotations.csv')
    target           = cfg.get('target', 'QIDS_Final_Integration')
    num_folds        = cfg.get('num_folds', 7)
    lambda_reg       = cfg.get('lambda_reg', 1.0)
    num_filters      = cfg.get('num_filters', 5)
    graph_attrs      = cfg.get('graph_attrs', []) or None  # None if empty list

    if os.path.exists(os.path.join(args.output_dir, 'metrics.json')):
        print(f'SELSER run already exists in {args.output_dir}.')
        return

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load dataset
    covs, labels, subject_ids, conditions, G = load_dataset(
        data_dir, annotations_file, target, graph_attrs=graph_attrs)
    print(f'Loaded {len(labels)} subjects  '
          f'(P={( conditions==1).sum()}, E={(conditions==-1).sum()})')
    if graph_attrs:
        print(f'Graph attrs: {graph_attrs}')

    # Permute the target across the cohort, for the permutation null
    if args.perm_seed is not None:
        labels = permute_labels(subject_ids, labels, args.perm_seed)
        print(f'Permuted the target with perm_seed={args.perm_seed}.')

    # 7-fold cross-validation stratified by treatment
    strat = (conditions == 1).astype(int)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)

    all_rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(covs, strat)):
        # Fit one model per treatment arm (mirrors Wu et al.)
        for cond_val, cond_label in [(1, 'P'), (-1, 'E')]:
            train_mask = conditions[train_idx] == cond_val
            test_mask  = conditions[test_idx]  == cond_val
            arm_train  = train_idx[train_mask]
            arm_test   = test_idx[test_mask]

            G_train = G[arm_train] if G is not None else None
            G_test  = G[arm_test]  if G is not None else None

            W, b, gamma, n_iter = fit_selser(
                covs[arm_train], labels[arm_train], lambda_reg=lambda_reg,
                G=G_train)
            y_pred = predict_selser(W, b, covs[arm_test], gamma=gamma, G=G_test)

            if args.save_filters:
                np.save(os.path.join(
                    args.output_dir, f'W_fold{fold}_{cond_label}.npy'), W)
                filters, weights = extract_filters(W, num_filters)
                np.save(os.path.join(
                    args.output_dir, f'filters_fold{fold}_{cond_label}.npy'), filters)
                np.save(os.path.join(
                    args.output_dir, f'filter_weights_fold{fold}_{cond_label}.npy'), weights)

            for local_i, global_i in enumerate(arm_test):
                all_rows.append({
                    'subject_id': int(subject_ids[global_i]),
                    'fold':       fold,
                    'Condition':  cond_val,
                    'label':      float(labels[global_i]),
                    'prediction': float(y_pred[local_i]),
                })

            print(f'  Fold {fold} [{cond_label}]: train={len(arm_train)}, '
                  f'test={len(arm_test)}, converged in {n_iter} iters')

    # Save predictions
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(
        os.path.join(args.output_dir, 'prediction_results.csv'), index=False)

    # Compute and save metrics
    metrics = compute_metrics(results_df['label'].values,
                              results_df['prediction'].values)
    metrics.update({'seed': args.seed, 'perm_seed': args.perm_seed,
                    'lambda_reg': lambda_reg, 'target': target, 'num_folds': num_folds})
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nOverall: r={metrics["r"]:.4f}, p={metrics["p"]:.4e}, '
          f'MAE={metrics["mae"]:.4f} ± {metrics["mae_std"]:.4f}, '
          f'RMSE={metrics["rmse"]:.4f}')

    # Plot
    true_vs_pred_scatter(
        results_df,
        save_path=os.path.join(args.output_dir, 'true_vs_predicted.png'))

    # Save full config used
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump({**cfg, 'seed': args.seed, 'output_dir': args.output_dir}, f, indent=2)


if __name__ == '__main__':
    main()
