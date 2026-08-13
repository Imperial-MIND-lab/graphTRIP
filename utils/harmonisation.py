"""
Test-time harmonisation of clinical (graph-level) model inputs for zero-shot transfer.

A model trained on a source cohort expects its clinical inputs on the scale it was trained
on, so deploying it on a target cohort requires an explicit choice of input mapping. This
module implements the two that matter:

    no harmonisation   The model's own training transform, applied to the new cohort. This
                       is the correct non-transductive way to deploy it: the identity for a
                       model trained on raw scores, and z-scoring by the source cohort's
                       TRAINING-fold statistics for a model trained on standardised ones.

    harmonised         The target cohort's own statistics replace the source statistics, so
                       each score is mapped onto the distribution the model was fit on. This
                       is unsupervised domain adaptation -- the 1-D case of site
                       harmonisation. 

Both are the same affine map and differ only in the source statistics:

    x' = (x - mu_source) / sd_source * sd_target + mu_target

(mu_target, sd_target) is the scale the model was trained to receive: (0, 1) for an
attribute that was standardised during training, and the source training-fold mean and SD
otherwise -- which makes the unharmonised map exactly the identity.

Target-cohort statistics are estimated LEAVE-ONE-OUT: subject i is rescaled using the mean
and SD of the other n-1 subjects, so no subject contributes to its own harmonisation. No
fold structure is involved. Nothing is fitted on the target cohort, so every subject is
equally unseen and there is no split to respect.

All standard deviations use ddof=0, matching get_graph_attrs_stats_dict() in
experiments/ingredients/data_ingredient.py.

Author: Hanna M. Tolle
Date: 2026-08-13
License: BSD 3-Clause
"""

import os
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Saved alongside the model weights
FOLD_INDICES_FILE = 'test_fold_indices.csv'

# (mean, sd) of one attribute, one entry per fold model
FoldStats = List[Tuple[float, float]]


# Accessors --------------------------------------------------------------------
def graph_attr_matrix(dataset) -> np.ndarray:
    '''Returns the [num_subjects, num_graph_attrs] matrix of raw graph attributes.'''
    return np.stack([data.graph_attr[0].numpy() for data in dataset])


def read_fold_indices(weights_dir: str) -> np.ndarray:
    '''
    Reads the source cohort's test-fold assignment saved next to the weights. This is read
    from disk rather than replicated because the source split may be a balanced k-fold,
    which cannot be reproduced from the seed alone.
    '''
    path = os.path.join(weights_dir, FOLD_INDICES_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{FOLD_INDICES_FILE} not found in {weights_dir}. It is needed to identify the "
            "training subjects of each fold model.")
    return pd.read_csv(path, header=None).values.ravel().astype(int)


def resolve_standardised_attrs(weights_dir: str,
                               graph_attrs: Sequence[str],
                               override: Optional[Sequence[str]] = None) -> List[str]:
    '''
    Which clinical attributes the trained model received in standardised form.

    `override` is used as-is when given, and is the recommended way to supply this. Sacred
    serialises config.json with jsonpickle, so when a training script assigned the same
    list object to both `graph_attrs` and `graph_attrs_to_standardise`, the latter was
    written as a back-reference ({'py/id': N}) whose target cannot be recovered from the
    saved file alone. The fallback below therefore accepts the saved value only when it is
    a plain list of known attribute names.
    '''
    if override is not None:
        unknown = [a for a in override if a not in graph_attrs]
        if unknown:
            raise ValueError(f"Standardised attributes {unknown} are not in graph_attrs "
                             f"{list(graph_attrs)}.")
        return list(override)

    path = os.path.join(weights_dir, 'config.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.json not found in {weights_dir}.")
    with open(path, 'r') as f:
        saved = json.load(f)
    value = saved.get('dataset', {}).get('graph_attrs_to_standardise', None)
    if isinstance(value, list) and all(isinstance(a, str) and a in graph_attrs for a in value):
        return list(value)
    raise ValueError(
        f"Cannot determine graph_attrs_to_standardise from {path} (got {value!r}). "
        "This happens when sacred wrote it as a jsonpickle back-reference. Pass the value "
        "explicitly via source_standardised_attrs.")


# Statistics -------------------------------------------------------------------
def loo_stats(values) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Leave-one-out mean and SD. Entry i is computed on every subject except i, so no
    subject contributes to its own rescaling.

    Returns two arrays of length n, so the resulting map is subject-specific (and hence
    piecewise affine across the cohort).
    '''
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    if n < 3:
        raise ValueError(f"Leave-one-out statistics need at least 3 subjects, got {n}.")
    others = np.broadcast_to(values, (n, n))[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    return others.mean(axis=1), others.std(axis=1)


def cohort_stats(values) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Whole-cohort mean and SD, broadcast to one entry per subject so it is a drop-in
    replacement for loo_stats(). Kept as a robustness comparison: if the two disagree
    materially, the harmonisation is being driven by individual subjects.
    '''
    values = np.asarray(values, dtype=float).ravel()
    return (np.full(len(values), values.mean()), np.full(len(values), values.std()))


def fold_train_stats(values, fold_of, num_folds: int) -> FoldStats:
    '''
    Training-fold (mean, SD) of one attribute. Entry k is computed on every subject NOT in
    test fold k, i.e. exactly the subjects that fold model k was trained on.
    '''
    values = np.asarray(values, dtype=float).ravel()
    fold_of = np.asarray(fold_of).ravel()
    stats = []
    for k in range(num_folds):
        train = values[fold_of != k]
        if len(train) == 0:
            raise ValueError(f"Fold {k} has no training subjects.")
        stats.append((float(train.mean()), float(train.std())))
    return stats


def compute_train_stats(values, attr_names: Sequence[str], fold_of, num_folds: int,
                        attrs: Optional[Sequence[str]] = None) -> Dict[str, FoldStats]:
    '''Per-fold training statistics of the requested attributes of the source cohort.'''
    attr_names = list(attr_names)
    attrs = attr_names if attrs is None else list(attrs)
    values = np.asarray(values, dtype=float)
    return {a: fold_train_stats(values[:, attr_names.index(a)], fold_of, num_folds)
            for a in attrs}


def target_scale(train_stats: Dict[str, FoldStats],
                 standardised_attrs: Sequence[str]) -> Dict[str, FoldStats]:
    '''
    The scale each fold model was trained to receive: (0, 1) for attributes standardised
    during training, and the source training-fold statistics otherwise -- which is what
    makes the unharmonised map the identity for a model trained on raw scores.
    '''
    standardised = set(standardised_attrs or ())
    return {a: ([(0.0, 1.0)] * len(fs) if a in standardised else list(fs))
            for a, fs in train_stats.items()}


# The map ----------------------------------------------------------------------
def map_clinical_inputs(x_raw,
                        graph_attrs: Sequence[str],
                        model_k: int,
                        train_stats: Dict[str, FoldStats],
                        target_stats: Dict[str, FoldStats],
                        harmonise: Sequence[str] = (),
                        new_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
                        ) -> np.ndarray:
    '''
    Maps raw clinical scores into the input space of fold model `model_k`.

    Parameters:
    ----------
    x_raw (np.ndarray): [n_subjects, n_graph_attrs] raw values of the cohort to map.
    graph_attrs (list): names of the columns of x_raw.
    model_k (int): index of the fold model whose input space is being targeted.
    train_stats (dict): {attr: FoldStats} source training statistics.
    target_stats (dict): {attr: FoldStats} scale the model expects, from target_scale().
    harmonise (list): attributes to rescale from the new cohort's own statistics. Anything
                      not listed gets the model's own training transform.
    new_stats (dict): {attr: (mu[n], sd[n])} new-cohort statistics, e.g. from loo_stats().
                      Required for every attribute in `harmonise`.
    '''
    x_raw = np.asarray(x_raw, dtype=float)
    graph_attrs = list(graph_attrs)
    harmonise = set(harmonise)

    missing = sorted(a for a in harmonise if a not in train_stats)
    if missing:
        raise ValueError(f"No source training statistics for harmonised attribute(s) "
                         f"{missing}; the target scale cannot be determined.")
    if harmonise and new_stats is None:
        raise ValueError("new_stats is required when harmonising.")
    absent = sorted(a for a in harmonise if new_stats is not None and a not in new_stats)
    if absent:
        raise ValueError(f"No new-cohort statistics supplied for {absent}.")

    x = x_raw.copy()
    for j, attr in enumerate(graph_attrs):
        if attr not in train_stats:
            continue
        mu_t, sd_t = target_stats[attr][model_k]
        if attr in harmonise:
            mu_s, sd_s = (np.asarray(s, dtype=float) for s in new_stats[attr])
        else:
            mu_s, sd_s = train_stats[attr][model_k]
        if sd_t == 0 or np.any(np.asarray(sd_s) == 0):
            continue
        x[:, j] = (x_raw[:, j] - mu_s) / sd_s * sd_t + mu_t
    return x


# Persistence ------------------------------------------------------------------
def write_train_stats(stats: Dict[str, FoldStats], path: str) -> str:
    '''
    Records which source statistics a transfer run used, one row per (attribute, fold
    model). Provenance only -- the statistics are always recomputed from the source
    dataset.
    '''
    rows = [{'feature': attr, 'fold': k, 'mean': mean, 'std': std}
            for attr, folds in stats.items() for k, (mean, std) in enumerate(folds)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def clinical_inputs_record(subject_ids, graph_attrs: Sequence[str], attrs: Sequence[str],
                           x_raw, mapped_by_condition: Dict[str, np.ndarray], model_k: int,
                           train_stats: Dict[str, FoldStats],
                           target_stats: Dict[str, FoldStats],
                           new_stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    '''
    Long-format record of what the mapping did to each clinical input: one row per
    (subject, attribute, fold model), with the raw value, the value under each condition,
    and every statistic that went into the map. 
    '''
    graph_attrs = list(graph_attrs)
    rows = []
    for attr in attrs:
        j = graph_attrs.index(attr)
        mu_target, sd_target = target_stats[attr][model_k]
        mu_train, sd_train = train_stats[attr][model_k]
        mu_new, sd_new = new_stats[attr]
        for i, subject in enumerate(subject_ids):
            row = {'subject_id': int(subject), 'attr': attr, 'pretrained_model': model_k,
                   'raw': float(x_raw[i, j]),
                   'mu_source_train': mu_train, 'sd_source_train': sd_train,
                   'mu_new_cohort': float(mu_new[i]), 'sd_new_cohort': float(sd_new[i]),
                   'mu_target': mu_target, 'sd_target': sd_target}
            for condition, mapped in mapped_by_condition.items():
                row[condition] = float(mapped[i, j])
            rows.append(row)
    return pd.DataFrame(rows)
