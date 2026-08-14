"""
Stage 6 -- put the ds005917 REACT betas on the psilodep intensity scale.

Usage (from project root):
    python -m preprocessing.ketamine.rescale_node_features
    python -m preprocessing.ketamine.rescale_node_features --dry-run

Idempotency: the original file is copied to node.csv.orig before the first rewrite, and the
presence of that backup means "already rescaled" -- the file is then skipped. Running this
script twice can therefore never divide twice.

License: BSD-3-Clause
Author: Hanna M. Tolle
Date: 2026-08-14
"""

import os
import glob
import shutil
import argparse

import numpy as np
import pandas as pd

from utils.files import project_root

STUDY = 'ds005917'
SESSION = 'before'
ATLASES = ['schaefer100', 'schaefer200', 'aal']

# Ratio of the ds005917 to the psilodep2 pooled node-feature sd; see docstring.
REACT_SCALE = 4.68

# Only the REACT receptor-enriched maps are on the scanner-intensity scale.
REACT_MARKER = '_Believeau-'

# psilodep2 pooled sd of the Believeau-3 maps used as node_attrs -- the target of the fix.
# Measured over all 42 psilodep2 subjects; reported per atlas so the log is comparable.
REFERENCE_ATTRS = ['5-HT1A_Believeau-3', '5-HT2A_Believeau-3', '5-HTT_Believeau-3']
REFERENCE_SD = {'schaefer100': 0.747, 'schaefer200': 0.756, 'aal': 0.558}


def study_dir(study=STUDY, atlas='schaefer100'):
    return os.path.join(project_root(), 'data', 'raw', study, SESSION, atlas)


def node_files(atlas):
    '''All node.csv paths for one atlas, sorted by subject.'''
    return sorted(glob.glob(os.path.join(study_dir(atlas=atlas), 'S*', 'node.csv')))


def pooled_sd(files, columns=REFERENCE_ATTRS):
    '''Pooled sd of the given columns across every subject, or None if unavailable.'''
    values = []
    for f in files:
        df = pd.read_csv(f)
        if not all(c in df.columns for c in columns):
            return None
        values.append(df[columns].to_numpy())
    return np.vstack(values).std() if values else None


def rescale_atlas(atlas, dry_run=False):
    '''
    Divides the REACT columns of every subject's node.csv by REACT_SCALE.

    Returns (n_rescaled, n_skipped). A file is skipped when node.csv.orig already
    exists, which is the marker that this script has already run on it.
    '''
    files = node_files(atlas)
    if not files:
        print(f'  {atlas}: no node.csv files found, skipping.')
        return 0, 0

    before = pooled_sd(files)
    n_rescaled, n_skipped = 0, 0

    for f in files:
        backup = f + '.orig'
        if os.path.exists(backup):
            n_skipped += 1
            continue

        df = pd.read_csv(f)
        react_cols = [c for c in df.columns if REACT_MARKER in c]
        if not react_cols:
            print(f'  WARNING: no REACT columns in {f}, leaving it alone.')
            n_skipped += 1
            continue

        if not dry_run:
            shutil.copy2(f, backup)
            df[react_cols] = df[react_cols] / REACT_SCALE
            df.to_csv(f, index=False)
        n_rescaled += 1

    after = pooled_sd(files) if not dry_run else (before / REACT_SCALE if before else None)
    verb = 'would rescale' if dry_run else 'rescaled'
    print(f'  {atlas}: {verb} {n_rescaled}, skipped {n_skipped} (already done)')
    if before is not None and after is not None:
        ref = REFERENCE_SD.get(atlas)
        ref_str = f'{ref:.3f}' if ref is not None else 'unknown'
        print(f'    pooled sd {before:.3f} -> {after:.3f}   '
              f'(psilodep2 {atlas} reference {ref_str})')
    return n_rescaled, n_skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing anything.')
    parser.add_argument('--atlases', nargs='+', default=ATLASES,
                        help=f'Atlases to rescale (default: {" ".join(ATLASES)}).')
    args = parser.parse_args()

    print(f'Rescaling {STUDY} REACT node features by 1/{REACT_SCALE}'
          + (' [DRY RUN]' if args.dry_run else ''))
    totals = [rescale_atlas(atlas, dry_run=args.dry_run) for atlas in args.atlases]
    n_rescaled = sum(t[0] for t in totals)
    n_skipped = sum(t[1] for t in totals)
    print(f'\nDone: {n_rescaled} files rescaled, {n_skipped} skipped.')
    if n_rescaled and not args.dry_run:
        print('Originals are kept alongside as node.csv.orig.\n'
              'Remember to delete data/processed/data_ds005917_*.pt -- the dataset cache is '
              'keyed on study/session/atlas only and will not notice this change.')


if __name__ == '__main__':
    main()
