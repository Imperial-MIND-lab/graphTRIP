"""
Generate annotations.csv for ds005917 (NIMH Ketamine dataset).

Reads data/raw/ds005917/participants.tsv and phenotype/phenotype.tsv,
merges with subject_map.csv, and writes data/raw/ds005917/annotations.csv
in the format expected by load_annotations() and preprocess.py.

Run after create_subject_map.py:
    python -m preprocessing.ketamine.create_annotations
"""

import os
import pandas as pd
from utils.files import project_root


def main():
    data_dir = os.path.join(project_root(), 'data', 'raw', 'ds005917')
    submap_file = os.path.join(project_root(), 'preprocessing', 'ketamine', 'subject_map.csv')

    participants = pd.read_csv(os.path.join(data_dir, 'participants.tsv'), sep='\t')
    phenotype = pd.read_csv(
        os.path.join(data_dir, 'phenotype', 'phenotype.tsv'),
        sep='\t', na_values='n/a'
    )
    subject_map = pd.read_csv(submap_file)

    # Pivot phenotype long to wide: one row per subject, one column per (scale, session)
    pheno_wide = phenotype.pivot(
        index='participant_id', columns='session_id',
        values=['MADRS_Total', 'HAMD_Bech', 'HAM17_Total']
    )
    pheno_wide.columns = [f'{var}_{ses}' for var, ses in pheno_wide.columns]
    pheno_wide = pheno_wide.reset_index().rename(columns={'participant_id': 'bids_id'})

    pheno_wide = pheno_wide.rename(columns={
        'MADRS_Total_ses-b0': 'MADRS_b0', 'MADRS_Total_ses-d2': 'MADRS_d2',
        'MADRS_Total_ses-d10': 'MADRS_d10', 'MADRS_Total_ses-p2': 'MADRS_p2',
        'MADRS_Total_ses-p10': 'MADRS_p10',
        'HAM17_Total_ses-b0': 'HAM17_b0', 'HAM17_Total_ses-d2': 'HAM17_d2',
        'HAM17_Total_ses-d10': 'HAM17_d10', 'HAM17_Total_ses-p2': 'HAM17_p2',
        'HAM17_Total_ses-p10': 'HAM17_p10',
        'HAMD_Bech_ses-b0': 'HAMD6_b0', 'HAMD_Bech_ses-d2': 'HAMD6_d2',
        'HAMD_Bech_ses-d10': 'HAMD6_d10', 'HAMD_Bech_ses-p2': 'HAMD6_p2',
        'HAMD_Bech_ses-p10': 'HAMD6_p10',
    })

    participants = participants.rename(columns={'participant_id': 'bids_id'})
    merged = subject_map.merge(
        participants[['bids_id', 'age', 'sex', 'BMI', 'infusion_1', 'infusion_2']],
        on='bids_id'
    )
    merged = merged.merge(pheno_wide, on='bids_id', how='left')

    # Treatment response: MADRS decrease from baseline (positive = improvement).
    # ses-d2 is always 2 days post-ketamine; ses-p2 is always 2 days post-placebo,
    # regardless of which infusion the subject received first.
    merged['MADRS_response_ket'] = merged['MADRS_b0'] - merged['MADRS_d2']
    merged['MADRS_response_pbo'] = merged['MADRS_b0'] - merged['MADRS_p2']

    # All 36 patients (33 MDD + 3 BP) are preprocessed and loadable
    merged['Exclusion'] = 0
    merged['missing_raw_before'] = 0

    # sub-MOA117 has an empty participants.tsv row (no age, sex, BMI, infusion order,
    # or any clinical scale) but still has a usable baseline scan.
    baseline_covariates = ['age', 'sex', 'MADRS_b0', 'HAM17_b0', 'HAMD6_b0']
    merged['missing_clinical'] = merged[baseline_covariates].isna().any(axis=1).astype(int)

    merged = merged.rename(columns={
        'study_id': 'Patient',
        'sex': 'Sex',
        'age': 'Age',
        'group': 'Group',
    })

    cols = [
        'Patient', 'bids_id', 's_id', 'Group', 'Exclusion',
        'missing_raw_before', 'missing_clinical',
        'Age', 'Sex', 'BMI', 'infusion_1', 'infusion_2',
        'MADRS_b0', 'MADRS_d2', 'MADRS_d10', 'MADRS_p2', 'MADRS_p10',
        'MADRS_response_ket', 'MADRS_response_pbo',
        'HAM17_b0', 'HAM17_d2', 'HAM17_d10', 'HAM17_p2', 'HAM17_p10',
        'HAMD6_b0', 'HAMD6_d2', 'HAMD6_d10', 'HAMD6_p2', 'HAMD6_p10',
    ]
    merged = merged[cols]

    output_file = os.path.join(data_dir, 'annotations.csv')
    merged.to_csv(output_file, index=False)
    print(f'Saved {len(merged)} subjects to {output_file}')
    print(merged[['Patient', 'bids_id', 's_id', 'Group', 'Exclusion',
                   'MADRS_b0', 'MADRS_d2', 'MADRS_response_ket']].to_string(index=False))
    print(f'\nGroup counts: {merged["Group"].value_counts().to_dict()}')
    print(f'Subjects with a usable MADRS_d2 target: '
          f'{int(merged["MADRS_d2"].notna().sum())}/{len(merged)}')
    incomplete = merged.loc[merged['missing_clinical'] == 1, 's_id'].tolist()
    print(f'Subjects with incomplete baseline clinical data: '
          f'{incomplete if incomplete else "none"}')
    usable = merged[(merged['missing_clinical'] == 0) & merged['MADRS_d2'].notna()]
    print(f'Fully usable for supervised training: {len(usable)} '
          f'({len(usable[usable.Group == "MDD"])} MDD, {len(usable[usable.Group == "BP"])} BP)')


if __name__ == '__main__':
    main()
