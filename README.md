# graphTRIP: Graph-based Treatment Response Interpretability and Prediction

This repository contains the full codebase for reproducing results from our study *Accurate and Interpretable Prediction of Antidepressant Treatment Response from Receptor-informed Neuroimaging*. It also includes some hopefully useful guidance for how to apply the graphTRIP pipeline to your own data.

## Installation

**System requirements**

- **Operating systems:** Developed and tested on Linux (Ubuntu 22.04 and CentOS/RHEL, including HPC clusters). The code will likely run on other operating systems as well.
- **Dependencies:** Python 3.12.4, PyTorch 2.2.2, PyTorch Geometric 2.6.1, and the packages listed in [`environment.yml`](environment.yml).
- **Hardware:** A standard machine with at least 8 GB RAM. A GPU is not required; the code runs on CPU as well. If you have a CUDA-capable GPU, training can use it automatically—see the note in [`environment.yml`](environment.yml) about choosing the PyTorch/CUDA version that matches your system.

Clone the repository and install all dependencies via Conda:

```bash
conda env create -f environment.yml
conda activate graphtrip
```

Typical installation time is about 15–20 minutes on a standard desktop, depending on internet speed and disk I/O.

Experiments are optionally integrated with [Neptune.ai](https://neptune.ai/) for logging configurations and outputs. If you wish to use Neptune, set up an account and export your API token. Alternatively, you can bypass Neptune entirely using the `FileStorageObserver`.

## Apply graphTRIP to Your Data

The full pipeline consists of four main stages. Each step depends on outputs from the previous.

#### Step 1: Construct a Dataset

The main interface is the `BrainGraphDataset` class (`datasets.py`), uniquely defined by a combination of `study`, `session`, and `atlas`.

```python
from dataset import BrainGraphDataset

dataset = BrainGraphDataset(
    study='psilodep2',
    session='before',
    atlas='schaefer100'
)
```

This initializes a dataset object using graph files stored in:

```
data/raw/{study}/{session}/{atlas}/S{subject_id}/
├── node.csv
├── edge.csv
├── bold.csv (optional)
```

Each graph can also include graph-level attributes (e.g., clinical scores, treatment condition) via:

```
data/raw/{study}/annotations.csv
```

Ensure this file contains a `Patient` column matching subject IDs, and that any additional graph attributes appear as columns. To define which attributes to load by default, modify the `Attrs.add_clinical_graph_attrs()` method inside `datasets.py`.

To force reload a modified dataset, run:

```python
dataset = BrainGraphDataset(
    study='psilodep2',
    session='before',
    atlas='schaefer100',
    force_reload=True
)
```

I recommend including all candidate node, edge and graph features upon first initialization. You can later train a model on a subset of features by specifying this in the configuration files.

#### Step 2: Train a Model

To train a graphTRIP model with specific configurations, specified in `experiments/configs/my_config.json`:

```bash
python -m experiments.run_experiment train_jointly FileStorageObserver --config_json=my_config.json
```

Results will be saved to:

```
outputs/runs/{config_json}/
```

unless overridden via the `--output_dir` argument or by setting the `output_dir` field in your JSON config.

Note: training on fully connected graphs increases training time and often leads to oversmoothing. You want to use a sparsely connected yet sufficiently informative graph structure.

## Demo with synthetic data

The notebook [`notebooks/demo.ipynb`](notebooks/demo.ipynb) demonstrates how to set up the data and train a model on a small synthetic dataset (*n* = 10). It walks through creating mock node/edge files and annotations, loading a `BrainGraphDataset`, and running the `train_jointly` experiment with the demo config [`experiments/configs/demo_config.json`](experiments/configs/demo_config.json) (no real neuroimaging data required). On a typical desktop, the full demo takes about 10–30 seconds.

## Approximate training time

Training a graphTRIP model with the standard pipeline (`train_jointly.py`) and configurations ([`experiments/configs/graphtrip.json`](experiments/configs/graphtrip.json); i.e., 7-fold cross-validation, 42 samples, 300 epochs per fold, 1 random seed) takes about **27 minutes**.

This benchmark was run on a single node with an **AMD EPYC 7742 64-Core Processor** (Linux, Python 3.12). The same run can be launched via [`scripts/graphtrip.py`](scripts/graphtrip.py), which calls the `train_jointly` experiment and writes weights to `outputs/graphtrip/weights/seed_<seed>/`. Actual runtimes will vary with hardware, dataset size, and config (e.g. `num_folds`, `num_epochs`).

## Reproducing Our Results

To fully replicate the results reported in our paper, you will need:

* Access to the `psilodep2` (primary) and `psilodep1` (validation) datasets.
* All open-access normative molecular target maps described in the paper.

Example directory structure:

```
data/
├── processed/
├── raw/
│   ├── psilodep1/
│   ├── psilodep2/
│   │   ├── annotations.csv
│   │   └── before/
│   │       └── schaefer100/
│   │           ├── S01/
│   │           │   ├── node.csv
│   │           │   ├── edge.csv
│   │           │   └── bold.csv
│   │           └── ...
│   ├── receptor_maps/
│   │   └── schaefer100/
│   │       └── schaefer100_receptor_maps.csv
│   ├── spatial_coordinates/
│   │   └── Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
│   └── transmodal_axis/
│       └── Schaefer100_SA_Axis.csv
```

Note: The "processed" directory contains files created by `torch_geometric` upon dataset creation. Do not place any files there. The "raw" directory should contain your pre-processed fMRI data. See `preprocessing/preprocess.py` for how the files in `data/raw/` are created (instructions below).

#### Replication Steps

1. Download the 5-HT PET data and compute REACT maps as described in
   `preprocessing/react/react_readme.md`.

   This creates:

   ```
   data/raw/{study}/{session}/MNI_2mm/
   ```

2. Run preprocessing for each dataset, session, and atlas. For example:

   ```bash
   python -m preprocessing.preprocess psilodep2 before schaefer100
   ```

3. Run primary scripts (no dependencies on other outputs):

   ```bash
   qsub job_scripts/primary_jobs.sh
   ```

4. Run secondary scripts (depend on outputs from primary scripts):

   ```bash
   qsub job_scripts/secondary_jobs.sh
   qsub job_scripts/short_secondary_jobs.sh
   ```

5. Post-hoc analysis:

   ```bash
   qsub job_scripts/posthoc_analysis.sh
   ```

6. Generate all figures with the jupyter notebook `notebooks/figures.ipynb`.

## Citation

If you use this codebase or model in your work, we'd be grateful if you could cite:

**Tolle et al. (2025).**
*Accurate and Interpretable Prediction of Antidepressant Treatment Response from Receptor-informed Neuroimaging.*

Thank you and have fun with graphTRIP!
