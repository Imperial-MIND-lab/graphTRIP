"""
Generate subject_map.csv mapping BIDS IDs (sub-MOA*) to sequential study IDs (S01, S02, ...).
Includes MDD and BP subjects only (excludes healthy controls).

Run once from the project root before any other preprocessing steps:
    python -m preprocessing.ketamine.create_subject_map
"""

import os
import pandas as pd
from utils.files import project_root


def main():
    participants_file = os.path.join(project_root(), 'data', 'raw', 'ds005917', 'participants.tsv')
    df = pd.read_csv(participants_file, sep='\t')

    df = df[df['group'].isin(['MDD', 'BP'])].reset_index(drop=True)

    df['study_id'] = range(1, len(df) + 1)
    df['s_id'] = df['study_id'].apply(lambda i: f'S{i:02d}')

    subject_map = df[['participant_id', 'study_id', 's_id', 'group']].rename(
        columns={'participant_id': 'bids_id'}
    )

    output_file = os.path.join(project_root(), 'preprocessing', 'ketamine', 'subject_map.csv')
    subject_map.to_csv(output_file, index=False)
    print(f'Saved {len(subject_map)} subjects to {output_file}')
    print(subject_map.to_string(index=False))


if __name__ == '__main__':
    main()
