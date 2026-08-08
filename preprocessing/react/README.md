# REACT node-feature preprocessing

This directory contains the pipeline that produces **REACT** (Receptor-Enriched
Analysis of functional Connectivity by Targets) maps, which are used as **node
features** in the brain graphs. REACT estimates, for every voxel/region, how
strongly its BOLD time series is associated with the spatial distribution of a
set of neurotransmitter receptors (here: serotonin / 5-HT receptor PET maps).

The pipeline has **two phases**:

- **Phase A — Build the PET receptor atlas** (run once, locally; originally on a
  Mac by Tim Lawn). Turns the raw 1 mm NRU serotonin PET maps into a single
  normalised 4D `pet_atlas.nii.gz` in 2 mm MNI152 space.
- **Phase B — Run REACT per subject** (run on the Imperial RDS/PBS HPC). Uses the
  atlas from Phase A together with each subject's preprocessed resting-state fMRI
  to produce subject-level receptor-associated connectivity maps.

> The scripts contain **hard-coded absolute paths** from the original machines
> (e.g. `/Users/timlawn/...` for Phase A and `/rds/general/user/hmt23/...` for
> Phase B). Update these to your environment before running.

## Dependencies

| Tool | Used by | Notes |
|------|---------|-------|
| [react-fmri](https://github.com/ottaviadipasquale/react-fmri) | `react_masks.sh`, `react.sh` | `pip install react-fmri`; provides the `react_masks` and `react` CLI commands |
| [ANTs](https://github.com/ANTsX/ANTs) (`ResampleImage`) | `resample_and_mask_5HT_atlas.sh` | resampling 1 mm → 2 mm |
| [FSL](https://fsl.fmrib.ox.ac.uk/) (`fslmaths`) | `resample_and_mask_5HT_atlas.sh` | binarising / applying masks; also supplies the Harvard-Oxford atlas |
| Python: `nibabel`, `numpy`, `pandas` | `normalise_5HT_atlas.py`, `concat_5HT_atlases.py` | available in the `graphtrp` conda env |

Input PET maps come from the **NRU Serotonin Atlas**:
<https://nru.dk/index.php/allcategories/category/90-nru-serotonin-atlas-and-clustering>

---

## Phase A — Build the PET receptor atlas (local)

### Step A1 — `resample_and_mask_5HT_atlas.sh`

**What it does**

1. Resamples each 1 mm-isotropic MNI152 PET map to **2 mm** isotropic using ANTs
   `ResampleImage 3 <in> <out> 2x2x2 0 0` (linear interpolation).
2. Builds a binary mask from the **Harvard-Oxford subcortical** atlas
   (`HarvardOxford-sub-maxprob-thr25-2mm`, copied from the FSL install and
   binarised with `fslmaths -bin`).
3. Applies that mask with `fslmaths -mas` to remove cerebellar reference regions.

**Inputs**
- `INPUT_DIR=.../5-HT_atlas` — one subdirectory per receptor (`5-HT*/`), each
  containing `*MNI152*.nii.gz` PET maps at 1 mm.
- FSL's `HarvardOxford-sub-maxprob-thr25-2mm.nii.gz`.

**Outputs** (into `OUTPUT_DIR=.../5-HT_atlas_2mm`, mirroring the receptor subdir layout)
- `*_2mm.nii.gz` — resampled maps
- `*_2mm_masked.nii.gz` — resampled + masked maps
- `MASK_DIR/HarvardOxford-sub-maxprob-thr25-2mm_binary.nii.gz` — the binary mask

### Step A2 — `normalise_5HT_atlas.py`

**What it does**
- For every `*_2mm_masked.nii.gz` (found recursively under
  `5-HT*/`), normalises non-zero voxels to **[0, 1]** (min-max over non-zero
  voxels, matching `react`'s internal normalisation). Handles 3D and 4D volumes.
- Writes per-map summary statistics (min/max/mean/median/std/quartiles).

**Inputs**: `*_2mm_masked.nii.gz` from Step A1
(`base_dir` is hard-coded to `.../5-HT_atlas_2mm`).

**Outputs**
- `*_2mm_masked_normalized.nii.gz` — normalised maps (input to Step A3)
- `statistics/normalization_statistics_<timestamp>.csv`

### Step A3 — `concat_5HT_atlases.py`

**What it does**
- Stacks the chosen normalised receptor maps into a single **4D** volume, one
  receptor per 4th-dimension index. This is the `pet_atlas.nii.gz` that REACT
  consumes.

**Run**
```bash
python concat_5HT_atlases.py \
    --atlas_dir /path/to/5-HT_atlas_2mm \
    --output_dir Believeau-5 \
    --receptors 5HT1A 5HT1B 5HT2A 5HT4 5HTT
```
- `--atlas_dir` defaults to `../data/raw/react_data/5-HT_atlas_2mm` relative to
  the script.
- `--receptors` are the receptor subdirectory names to include; the script
  globs `*_2mm_masked_normalized.nii.gz` inside each.
- `--output_dir` is the name of the **receptor set** (e.g. `Believeau-5`), used
  for all downstream paths in Phase B.

**Outputs** (into `<atlas_dir>/concatenated/<output_dir>/`)
- `pet_atlas.nii.gz` — the 4D atlas (the REACT target maps)
- `input_maps.txt` — `receptor,filename` lines recording the stacking order

> Note: the `--output_dir` (receptor-set name) must match `receptor_set` in the
> Phase B scripts. The Phase B scripts use `receptor_set="Believeau-5"`; the
> docstring examples in `concat_5HT_atlases.py` use `Beliveau-3` only as an
> illustration.

---

## Phase B — Run REACT per subject (HPC / PBS)

The Phase B scripts assume this layout on the HPC:
```
$project_dir = /rds/general/user/hmt23/home/projects/graphTRP
  data/raw/react_data/<dataset>/concatenated/<receptor_set>/pet_atlas.nii.gz   # from Phase A
  data/raw/react_data/<dataset>/concatenated/<receptor_set>/input_maps.txt
  data/raw/react_data/masks/gm_mask.nii.gz                                     # grey-matter mask (2mm MNI152)
  data/raw/<study>/<session>/MNI_2mm/subject_list.txt
  data/raw/<study>/<session>/MNI_2mm/REACT_<receptor_set>/                     # outputs land here

$home_data = /rds/general/user/hmt23/home/data/<study>/<session>/S*/          # subject fMRI
```
with `study=psilodep1`, `session=before`, `dataset=5-HT_atlas_2mm`,
`receptor_set=Believeau-5`.

You also need a **grey-matter mask** `gm_mask.nii.gz` in 2 mm MNI152 space at
`data/raw/react_data/masks/` — REACT restricts the analysis to these voxels.
(This file is a prerequisite; it is not produced by any script here.)

### Step B1 — `create_subject_list.sh`

**What it does**: finds every `*.nii.gz` under
`/rds/general/user/hmt23/home/data/<study>/<session>/` and writes the absolute
paths to `subject_list.txt` (one per line). This is the list of subject fMRI
volumes REACT will operate on.

**Output**: `subject_list.txt` (in the current working directory). It must then
be placed at `data/raw/<study>/<session>/MNI_2mm/subject_list.txt` so that
`react_masks.sh` can copy it (see below).

### Step B2 — `react_masks.sh`  (PBS job: 1 node, 4 cpus, 16 gb, ~1 h)

**What it does**: creates the two REACT masks used in the two-stage regression.

1. `cd`s into the output dir
   `data/raw/<study>/<session>/MNI_2mm/REACT_<receptor_set>/` (created if absent).
2. Copies in the prerequisites: `subject_list.txt`, `pet_atlas.nii.gz`,
   `input_maps.txt`, and `gm_mask.nii.gz`.
3. Runs:
   ```bash
   react_masks subject_list.txt pet_atlas.nii.gz gm_mask.nii.gz out_masks
   ```

**Outputs** (in `REACT_<receptor_set>/out_masks/`)
- `mask_stage1.nii.gz` — voxels used to estimate subject-level receptor
  time series (stage-1 spatial regression).
- `mask_stage2.nii.gz` — voxels for which receptor-associated connectivity is
  estimated (stage-2 temporal regression).

**Submit**
```bash
qsub react_masks.sh
```

### Step B3 — `react.sh`  (PBS **array** job: `-J 1-16`, 4 cpus, 16 gb, ~30 min each)

**What it does**: one array task per subject. Maps `PBS_ARRAY_INDEX` (1-based)
onto the sorted list of subject directories (`S*`) and runs the two-stage REACT
regression for that subject.

```bash
react <subject_fMRI> out_masks/mask_stage1.nii.gz out_masks/mask_stage2.nii.gz \
      pet_atlas.nii.gz <output_dir>/<subject_id>
```

- Subject fMRI input file pattern:
  `<study>/<session>/<subject_id>/<session>_rest_rdsmffms6FWHM_bd_M_V_DV_WMlocal2_modecorr.nii.gz`
- Depends on `out_masks/` (Step B2) and `pet_atlas.nii.gz` (Phase A).

**Outputs** (in `REACT_<receptor_set>/<subject_id>/`)
- `<subject_id>_react_stage1.txt` — subject-specific receptor time series
- `<subject_id>_react_stage2.nii.gz` — per-receptor connectivity maps (these are
  the REACT maps that become node features)

**Submit** (array size must match the number of subjects; adjust `-J 1-N`)
```bash
qsub react.sh
```

---

## Quick reference — run order

```bash
# Phase A (local: ANTs + FSL + python)
./resample_and_mask_5HT_atlas.sh          # A1: 1mm -> 2mm, mask cerebellum
python normalise_5HT_atlas.py             # A2: normalise to [0,1] + stats
python concat_5HT_atlases.py \            # A3: stack receptors -> pet_atlas.nii.gz
    --output_dir Believeau-5 --receptors 5HT1A 5HT1B 5HT2A 5HT4 5HTT

# Phase B (HPC: react-fmri via PBS)
./create_subject_list.sh                  # B1: build subject_list.txt
qsub react_masks.sh                       # B2: -> out_masks/{mask_stage1,mask_stage2}.nii.gz
qsub react.sh                             # B3: array job, REACT per subject
```

## Downstream use

The per-subject stage-2 maps are parcellated (e.g. schaefer100/200) and stored as
`node.csv` columns in the `BrainGraphDataset` raw layout
(`data/raw/{study}/{session}/{atlas}/S{subject}/node.csv`), where each receptor
map contributes one node feature.

---

*Authors: Tim Lawn (Phase A spatial processing / normalisation, 16.12.2024),
Hanna Tolle (atlas concatenation + REACT HPC pipeline, 21.12.2024).*
*Supersedes the earlier `react_readme.md` in this directory.*
