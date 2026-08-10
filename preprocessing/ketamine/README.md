# Ketamine dataset preprocessing

This module preprocesses the **NIMH Ketamine Mechanism of Action Study** (OpenNeuro
[ds005917](https://openneuro.org/datasets/ds005917/versions/1.0.1)) from raw BIDS
fMRI into the node/edge feature files consumed by graphTRIP. It also produces the
cleaned voxelwise NIfTIs required for REACT analysis.

Unlike the psilodep datasets (which arrived pre-processed), ds005917 is raw BIDS
data and requires full preprocessing from scratch: MNI registration, motion
correction, nuisance regression, and band-pass filtering before parcellation and
REACT can run.

---

## Dataset summary

- **n = 36** non-HC subjects (33 MDD + 3 BP)
- **Treatment:** single IV ketamine infusion (0.5 mg/kg) vs. saline placebo,
  double-blind crossover
- **Depression scales:** MADRS, HAMD-17, HAMD-6 (at every session)
- **Sessions:** `ses-b0` (baseline), `ses-d2`/`ses-d10` (2/10 days post-ketamine),
  `ses-p2`/`ses-p10` (2/10 days post-placebo)
- **Imaging:** 3T GE Signa HDx, TR = 2.5 s, resting-state fMRI (8 min, eyes
  closed), T1w, DWI (baseline only)
- **License:** CC0 (fully open, no DUA)

---

## Pipeline overview

```
Raw BIDS (data/raw/ds005917/)
    │
    ▼  Stage 1 · fmriprep.sh
fMRIPrep: brain extraction, motion correction, MNI registration
    │         Output: data/preprocessed/ds005917/
    │
    ▼  Stage 2 · postprocess.sh / postprocess.py
nilearn: nuisance regression, bandpass 0.01–0.08 Hz, 6 mm smoothing
    │         Output: ~/data/ds005917/before/S{N}/before_rest_preproc.nii.gz
    │
    ▼  Stage 3 · react_masks_ketamine.sh
REACT: build shared stage-1 and stage-2 masks, once per receptor set
    │         Output: data/raw/ds005917/before/MNI_2mm/REACT_Believeau-{5,3}/out_masks/
    │
    ▼  Stage 4 · react_ketamine.sh
REACT: per-subject receptor-enriched connectivity maps, once per receptor set
    │         Output: data/raw/ds005917/before/MNI_2mm/REACT_Believeau-{5,3}/S{N}/
    │
    ▼  Stage 5 · preprocess_features.sh
preprocess.py: parcellate BOLD, compute node/edge features, save node.csv / edge.csv
              Output: data/raw/ds005917/before/{atlas}/S{N}/node.csv, edge.csv, bold.csv
```

All five stages are chained automatically by `submit_pipeline.sh` using PBS
`afterokarray` / `afterok` job dependencies.

---

## One-time local setup

Run these two scripts **once locally** (before copying to HPC) to generate the
mapping and annotation files from the downloaded participants/phenotype data:

```bash
# From project root (graphTRIP/)
python -m preprocessing.ketamine.create_subject_map   # → preprocessing/ketamine/subject_map.csv
python -m preprocessing.ketamine.create_annotations   # → data/raw/ds005917/annotations.csv
```

**`create_subject_map.py`** reads `data/raw/ds005917/participants.tsv` and assigns
sequential IDs (`S01`–`S36`) to the 33 MDD and 3 BP subjects (HCs are excluded).
This mapping is the bridge between BIDS IDs (`sub-MOA101`) and the `S0N` convention
used throughout graphTRIP and the REACT pipeline.

**`create_annotations.py`** merges participants and phenotype data into
`data/raw/ds005917/annotations.csv` with the `Patient` / `Exclusion` columns
required by `load_annotations()` and `preprocessing/preprocess.py`. Key columns:

| Column | Description |
|---|---|
| `Patient` | Integer ID (1–36), matches `S0N` via `get_subject_id()` |
| `Group` | `MDD` (33) or `BP` (3) — **use this to filter diagnosis at training time** |
| `Exclusion` | 0 for all 36 patients; reserved for subjects that cannot be used at all |
| `missing_raw_before` | 1 if the cleaned NIfTI is absent (set by `preprocess.py`) |
| `missing_clinical` | 1 if any baseline covariate is missing — currently only `S16` |
| `MADRS_b0` | Baseline MADRS score |
| `MADRS_d2` | Day-2 post-ketamine MADRS — the **default target** |
| `MADRS_response_ket` | MADRS decrease b0 → d2 (positive = improvement) |
| `MADRS_response_pbo` | MADRS decrease b0 → p2 (placebo arm) |

All 36 patients are preprocessed and loadable so their scans are available for
VGAE/reconstruction training regardless of clinical completeness. Restrict the cohort
at training time via the prefilter, e.g. `{'Exclusion': 0, 'missing_raw_before': 0,
'missing_clinical': 0, 'Group': 'MDD'}`. `load_annotations()` applies the filter
*before* `convert_to_numerical()`, so `Group` is still the raw `'MDD'`/`'BP'` string
when the filter runs.

Counts after filtering: **33 MDD**, of which **27** have a `MADRS_d2` target; **29**
subjects total (27 MDD + 2 BP) are fully usable for supervised training.

`sub-MOA117` (S16) has an entirely empty `participants.tsv` row — no age, sex, BMI,
infusion order, or clinical scale at any timepoint — but a usable baseline scan. It is
flagged by `missing_clinical`, not excluded: every default `graph_attr` would be NaN,
which breaks the MLP forward pass rather than merely dropping the subject from the loss.

Both files are already generated and committed. Re-run the scripts only if
`participants.tsv` changes.

### Root-level BIDS files

The `aws s3 sync` command in `ketamine_dataset.md` does **not** fetch root-level files —
each `--include` pattern requires a `sub-*/`, `phenotype/`, or `participants*` prefix.
fMRIPrep will not start without them:

```bash
for f in dataset_description.json task-rest_bold.json task-rest_physio.json .bidsignore; do
    aws s3 cp --no-sign-request "s3://openneuro.org/ds005917/${f}" "./data/raw/ds005917/${f}"
done

# sub-MOA136 has no ses-b0 T1w; borrow the one from ses-d2
aws s3 cp --no-sign-request --recursive \
    s3://openneuro.org/ds005917/sub-MOA136/ses-d2/anat/ \
    ./data/raw/ds005917/sub-MOA136/ses-d2/anat/
```

---

## One-time HPC setup

The HPC runs **Apptainer 1.5.1** (available system-wide at `/usr/bin/apptainer`; no `module load` needed). `singularity` is an alias for the same binary.

```bash
# Run these in a PBS interactive session (not on the login node — the image is ~14 GB):
#   qsub -I -l select=1:ncpus=2:mem=8gb -l walltime=1:00:00

# 1. Redirect Apptainer's build cache off your home quota
export APPTAINER_CACHEDIR=/rds/general/user/hmt23/home/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR

# 2. Pull the fMRIPrep image (Docker → SIF conversion, takes ~15 min)
mkdir -p /rds/general/user/hmt23/home/projects/graphTRP/containers
apptainer pull \
    /rds/general/user/hmt23/home/projects/graphTRP/containers/fmriprep.sif \
    docker://nipreps/fmriprep:24.1.1

# 3. Verify
apptainer run \
    /rds/general/user/hmt23/home/projects/graphTRP/containers/fmriprep.sif \
    --version   # expected: fMRIPrep 24.1.1

# 4. FreeSurfer licence (free; register at https://surfer.nmr.mgh.harvard.edu/registration.html)
#    Copy the emailed license.txt from your local machine:
#      scp ~/Downloads/license.txt hmt23@login.cx3.hpc.ic.ac.uk:~/.freesurfer/license.txt
mkdir -p ~/.freesurfer
# cp /path/to/license.txt ~/.freesurfer/license.txt
```

---

## Running the pipeline

### Pilot first

Never submit the full array without piloting. `pilot.sh` runs all five stages
sequentially for **S01** and **S32** in one job, so every failure lands in a single log
in the order it happened. Those two subjects are chosen because they exercise the only
two non-standard input paths: S01 has two complete baseline rest runs (tests
`select_run()`), and S32 has no `ses-b0` T1w (tests the cross-session anatomical lookup).

```bash
# From the project root
qsub preprocessing/ketamine/pilot.sh
qstat -u hmt23                       # ~4-7 h
```

Then inspect `outputs/qc_ds005917/qc_S01_schaefer100.png` and `qc_S32_schaefer100.png`.

**After a successful pilot, before the full run**, reset the raw-data flags:

```bash
python -m preprocessing.ketamine.create_annotations
```

Stage 5 marks the 34 unprocessed subjects as `missing_raw_before=1`, and the full run
does not clear that by itself (`preprocess.py` only rewrites the column when some
subject is missing). Stage 3 must also re-run for the full cohort — the pilot's masks
are built from 2 subjects and are not valid for the group.

### Full run

```bash
# From the HPC login node, project root
bash preprocessing/ketamine/submit_pipeline.sh
```

`qsub` only queues work — it performs no computation — so it is safe on a login node.
To submit only a single stage (e.g. for re-runs after partial failure), use `qsub`
directly and add a `depend=` flag manually if needed.

---

## Stage details

### Stage 1 — fMRIPrep (`fmriprep.sh`)

**Resources:** 8 CPUs, 32 GB RAM, 4 h walltime · PBS array 1–36

fMRIPrep handles all steps that require anatomical–functional co-registration and
MNI normalisation. The key choices:

- **`--output-spaces MNI152NLin6Asym:res-2`** — the FSL MNI152 2 mm grid (91×109×91).
  This is a hard requirement, not a preference: the REACT PET atlas, the REACT GM mask,
  and nilearn's Schaefer parcellation are all on this grid.
  `MNI152NLin2009cAsym:res-2` is a *different template on a different grid*
  (97×115×97) — REACT fails on it outright, and `NiftiLabelsMasker` silently resamples
  the parcellation instead of erroring, yielding misregistered features that look
  entirely normal. Check with `python -m preprocessing.ketamine.qc grid --s-id S01`.
- **`--bids-filter-file bids_filter.json`** — restricts BOLD to `ses-b0`/`task-rest`.
  **fMRIPrep has no `--session-id` argument**; passing one aborts at argument parsing.
  The filter deliberately leaves `t1w` unconstrained by session so `sub-MOA136` picks
  up its `ses-d2` T1w. Anatomical processing is subject-level, so no file renaming is
  needed — and renaming would falsify the BIDS record.
- `--fs-no-reconall` — skips FreeSurfer surface reconstruction (saves ~6 h/subject);
  not needed for voxelwise or parcellated analyses.
- `--fd-spike-threshold 0.5` — emits `motion_outlierXX` spike regressors in the
  confounds TSV, which stage 2 consumes.
- `--task-id rest` — skips dot-probe, emotion-evaluation and n-back runs.

Output lands in `data/preprocessed/ds005917/` (project root on HPC).

**No slice-timing correction is performed — this is a divergence from psilodep, not a
free choice.** fMRIPrep 24.1.1 performs STC by default, but only when `SliceTiming` is
present in the metadata. It is absent from the root `task-rest_bold.json` and from every
per-run sidecar (verified across `ses-b0` and `ses-d2`), despite the dataset `CHANGES`
file claiming v1.0.1 added it — the v1.1.0 "Add resting-state fMRI data" entry appears to
have overwritten them. fMRIPrep therefore skips STC silently. Guessing a slice order for
a GE 2D EPI is a coin flip and a wrong guess is worse than none. At TR = 2.5 s with a
0.01–0.08 Hz band-pass the cost is modest, but it must be stated in the methods.

### Stage 2 — Post-processing (`postprocess.sh` / `postprocess.py`)

**Resources:** 4 CPUs, 16 GB RAM, 30 min · PBS array 1–36

Applies the cleaning steps that fMRIPrep intentionally omits, in this order:

1. **Non-steady-state trimming** — leading T1-saturation volumes, flagged by fMRIPrep as
   `non_steady_state_outlierXX`, are dropped outright.
2. **Nuisance regression** — 6 rigid-body motion parameters + whole-mask mean WM and CSF
   signals + `motion_outlierXX` spike regressors.
3. **Band-pass filtering** — 0.01–0.08 Hz (via nilearn `NiftiMasker`).
4. **Spatial smoothing** — 6 mm FWHM Gaussian kernel.

No global signal regression: psilodep did not use it, and Kraus et al. 2020
(PMC7162890) showed GSR materially changes conclusions on this exact dataset.

#### Why non-steady-state volumes are trimmed rather than regressed

nilearn's `clean()` applies operations in a fixed order (verified against nilearn 0.10.3,
`nilearn/signal.py`):

```
interpolate/censor → detrend → band-pass (signals AND confounds) → confound regression → standardize
```

Confound regression comes **last**. A `non_steady_state_outlier` regressor therefore
cannot stop the saturated volumes from contaminating the linear detrend and the
Butterworth filter, and those volumes sit at the very start of the series where filter
edge effects are worst. Trimming them up front is the only ordering that works, and it
matches what the psilodep pipeline did (the leading `r` in its filename).

nilearn's `sample_mask` is deliberately *not* used for this. `clean()`'s `extrapolate`
argument defaults to `True`, so leading out-of-bounds volumes would be spline-*extrapolated*
rather than dropped — the opposite of what saturated data needs — and reaching
`extrapolate=False` through `NiftiMasker` requires an untyped `**kwargs` passthrough that
is fragile across versions. Non-steady-state volumes are always leading, so there is
nothing to interpolate across anyway.

Motion spikes stay as regressors: with the same Butterworth applied to signal and
regressor alike, a scaled delta is removed exactly. Any spike regressor that flagged only
trimmed volumes becomes all-zero after trimming and is dropped — otherwise the design
matrix is rank-deficient at nilearn's QR step.

#### Degrees of freedom

The band-pass discards most of the spectrum, so the raw volume count badly overstates the
DoF available for nuisance regression. At TR = 2.5 s the Nyquist frequency is 0.2 Hz, and
the 0.07 Hz passband retains ~35 % of it — roughly **67 effective DoF from 192 volumes**.
`load_confounds()` warns when regressors exceed 25 % of *effective* DoF (~17 regressors),
not 25 % of the volume count.

#### Confound set vs psilodep

The psilodep input file is `..._rdsmffms6FWHM_bd_M_V_DV_WMlocal2_modecorr.nii.gz`. Its
`WMlocal2` term is a *local* white-matter estimate (a spherical neighbourhood per voxel,
AFNI `3dLocalstat`), and `V`/`DV` are separate ventricle and vessel terms.

This pipeline uses fMRIPrep's `white_matter` and `csf`, which are **whole-mask mean**
signals. That is not equivalent. A whole-mask WM mean carries more of the global signal
than a local WM estimate does, so this is a partial step toward GSR — the exact axis the
no-GSR choice is meant to protect — and it can shift the FC edge distribution toward more
negative edges relative to psilodep. That shift is precisely the covariate shift that
hurts transfer.

This is left as-is pending measurement rather than changed pre-emptively, because the
`qc cohort` report tests it directly: the pooled edge-weight distribution and the
per-subject mean/SD/fraction-negative panels compare ds005917 against psilodep1. Decide
from those numbers. If they diverge materially, the options are fMRIPrep's `a_comp_cor_*`
components or replicating the local-WM regression in AFNI — both are themselves
deviations from psilodep, in a different direction, so neither is automatically better.

*Note: the expansion of the `DV` term above is inferred from the filename; it is not
documented anywhere in this repository. Confirm against the Methods write-up before it
goes into the paper.*

#### Run selection

`sub-MOA101` (S01) and `sub-MOA201` (S34) each have two complete 192-volume baseline
runs with no metadata indicating a preference. Both are preprocessed by fMRIPrep;
`select_run()` picks the one with lower mean framewise displacement and logs the choice.
The criterion is outcome-independent and affects only these two subjects — record it in
the methods.

Cleaned NIfTIs are saved to `~/data/ds005917/before/S{N}/before_rest_preproc.nii.gz`,
the path pattern expected by both the REACT scripts and `preprocessing/preprocess.py`
(via `utils.files.get_raw_filename('ds005917', 'before')`).

### Stage 3 — REACT masks (`react_masks_ketamine.sh`)

**Resources:** 4 CPUs, 16 GB RAM, 1 h · Single job

Runs `react_masks` (from the `react-fmri` package) across all subjects to build two
shared spatial masks used in the two-stage REACT regression:

- `mask_stage1.nii.gz` — voxels used to estimate subject-level receptor time series
- `mask_stage2.nii.gz` — voxels for which receptor-associated connectivity is computed

This runs **once per receptor set** (`Believeau-5` and `Believeau-3`), because each set
has its own PET atlas and therefore its own masks. Both are required:
`preprocessing/preprocess.py` loops over both, and the default `node_attrs` in
`experiments/ingredients/data_ingredient.py` are the **Believeau-3** maps — so producing
only `Believeau-5` makes stage 5 fail.

This reuses the existing **Phase A PET atlas** (`data/raw/react_data/5-HT_atlas_2mm/`)
that was built for the psilodep datasets. No changes to Phase A are needed. Note that
the code spells the receptor set `Believeau`.

### Stage 4 — REACT per subject (`react_ketamine.sh`)

**Resources:** 4 CPUs, 16 GB RAM, 30 min · PBS array 1–36

Runs the two-stage REACT regression for each subject, producing per-receptor
connectivity maps (`*_react_stage2.nii.gz`). These are later parcellated and stored
as columns of `node.csv`.

### Stage 5 — Feature extraction (`preprocess_features.sh`)

**Resources:** 4 CPUs, 16 GB RAM, 2 h · Single job

Runs `preprocessing/preprocess.py` for `study=ds005917`, `session=before`, for all
three atlases (`schaefer100`, `schaefer200`, `aal`). This script:

1. Parcellates the cleaned BOLD time series → `bold.csv`
2. Computes node features (including REACT maps) → `node.csv`
3. Computes edge features (functional connectivity) → `edge.csv`

Output structure:
```
data/raw/ds005917/before/{atlas}/S{N}/
├── bold.csv    # parcellated BOLD [T × n_rois]
├── node.csv    # node features (REACT maps, etc.) [n_rois × n_features]
└── edge.csv    # edge features (FC) [n_edges × n_features]
```

---

## Relationship to the psilodep pipeline

The goal for this dataset is transfer *from* psilodep, which requires comparably
preprocessed features. 

### Matches psilodep

| Choice | Note |
|---|---|
| No GSR | Consistent with psilodep and with the paper's framing |
| Band-pass 0.01–0.08 Hz | Same as Daws et al. 2022 |
| 6 mm FWHM smoothing | Same as Daws |
| Confounds band-pass filtered with the same Butterworth as the data | nilearn applies the filter to the confound matrix too (Lindquist et al. 2018 orthogonalisation), avoiding the Hallquist 2013 noise-reintroduction problem. **State this explicitly — reviewers ask.** |
| FD > 0.5 mm spike criterion | `--fd-spike-threshold 0.5` |
| Non-steady-state volumes dropped before filtering | Equivalent to psilodep's leading `r` |

### Diverges from psilodep

| Divergence | Impact | Status |
|---|---|---|
| **No slice-timing correction** | `SliceTiming` absent from all sidecars, so fMRIPrep skips STC. psilodep did correct. Modest at TR = 2.5 s with this passband | Unfixable — dataset metadata gap |
| **Whole-mask WM/CSF instead of local WM + ventricle/vessel terms** | May shift the FC edge distribution; see Stage 2 | Measure via `qc cohort`, then decide |
| **No despiking** | psilodep ran AFNI `3dDespike`; fMRIPrep has no equivalent | Accepted |
| **Linear detrend only** | nilearn `detrend=True` is linear; psilodep detrended linear + quadratic. Low impact — the 0.01 Hz high-pass removes most low-order drift | Accepted |
| **No intensity mode normalisation** | psilodep files end in `modecorr` (mode-1000 scaling); fMRIPrep `desc-preproc_bold` is in arbitrary scanner units | Accepted, with the caveat below |

**The mode-normalisation caveat matters for transfer.** It mostly washes out because node
attributes are re-standardised per training fold
([data_ingredient.py](../../experiments/ingredients/data_ingredient.py)) and FC is computed
from z-scored parcel series. But it does mean REACT betas are **not on the psilodep
scale**, so do not apply a psilodep-pretrained model zero-shot without re-standardising.

---

## Utility extensions

The following functions were extended to support `ds005917`:

| Function | File | Addition |
|---|---|---|
| `get_raw_filename()` | `utils/files.py` | Returns `'before_rest_preproc.nii.gz'` |
| `get_tr()` | `utils/annotations.py` | Returns `2.5` (GE Signa HDx) |
| `get_categorical_annotations()` | `utils/annotations.py` | Returns `['Group', 'Sex']` |
| `get_cat2num_dict()` | `utils/annotations.py` | Maps `Sex` and `Group` |
| `patient_ids_to_sample_ids()` | `utils/annotations.py` | Prefilter `{'Exclusion': 0, 'missing_raw_before': 0}` |
| `get_default_prefilter()` | `datasets.py` | Prefilter `{'Exclusion': 0, 'missing_raw_before': 0}` |
| `get_default_target()` | `datasets.py` | Returns `'MADRS_d2'` |
| `Attrs.add_clinical_graph_attrs()` | `datasets.py` | `['Sex', 'Age', 'MADRS_b0', 'HAM17_b0', 'HAMD6_b0']` |

`Group` is deliberately **not** a `graph_attr`: it is a filtering column, and it becomes
constant — hence zero-variance, dividing by zero under standardisation — as soon as the
cohort is filtered to a single diagnosis.

---

## Quality control

`qc.py` covers the failure modes of all five stages. Run the grid check *before*
submitting anything; it costs two seconds and catches the template-space error that
would otherwise produce silently misregistered features.

```bash
python -m preprocessing.ketamine.qc bids                  # BIDS tree complete?
python -m preprocessing.ketamine.qc grid --s-id S01       # template-space agreement
python -m preprocessing.ketamine.qc subject --s-id S01    # per-subject report
python -m preprocessing.ketamine.qc cohort                # motion, QC-FC, psilodep1 comparison
```

`bids` and `grid` need no derivatives and exit non-zero on failure, so they are safe to
chain ahead of a submission. `bids` confirms the four root files are present, that
`TaskName` is reachable by inheritance, and that all 36 subjects have both a rest run and
a T1w in some session. Its expected output flags exactly three subjects: `S01` and `S34`
(two rest runs) and `S32` (T1w in `ses-d2`).

Figures land in `outputs/qc_ds005917/`. The per-subject report covers motion, carpet
plots before and after cleaning, the power spectrum (proving the band-pass fired),
registration against the REACT GM mask, REACT stage-2 beta distributions, per-parcel
signal, the FC matrix, and node features overlaid on psilodep1's.

The cohort report has six panels: ranked mean FD, volume-level motion burden, QC-FC,
group-mean FC agreement with psilodep1, the pooled edge-weight distribution of both
cohorts, and per-subject FC summaries (mean edge weight vs fraction of negative edges).
It also prints a subject-level statistics table:

```
FC edge-weight comparison (subject-level means):
   cohort  n   mean     sd  frac_neg
psilodep1 16 0.1981 0.2335    0.2173
ds005917  29 ...    ...       ...
```

The last three panels plus that table are the go/no-go for transfer, and the evidence for
the confound-set decision described under Stage 2. A group-mean FC correlation below ~0.7,
or a clear separation in mean edge weight / fraction-negative, indicates a covariate shift
that will hurt transfer.

**Pilot before the full run.** Process `S01` (which exercises the two-run selection path)
and `S32` (the cross-session T1w path) end-to-end first.

No subject-level motion exclusion is applied automatically. Set the threshold from the
cohort plot — conventional cut-offs are mean FD > 0.3 mm or > 20 % of volumes over the
0.5 mm spike threshold — then apply it uniformly.

---

## Loading the dataset in graphTRIP

Once preprocessing is complete, the dataset loads identically to psilodep:

```python
from datasets import BrainGraphDataset

# All 36 patients (33 MDD + 3 BP), for VGAE/reconstruction training
dataset = BrainGraphDataset(
    study='ds005917',
    session='before',
    atlas='schaefer100'
)

# MDD only, with complete clinical covariates — for supervised training
dataset = BrainGraphDataset(
    study='ds005917',
    session='before',
    atlas='schaefer100',
    prefilter={'Exclusion': 0, 'missing_raw_before': 0,
               'missing_clinical': 0, 'Group': 'MDD'}
)
```

*Authors: Hanna M. Tolle, June 2026.*
