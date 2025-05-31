# graph-based Treatment Response Interpretable Predictions (graphTRIP)
This repository contains code for predicting clinical outcomes of anitdepressant treatment from neuroimaging data using graphTRIP.

## Installation
Dependencies can be installed with conda using the `environment.yml` file.
```conda env create -f environment.yml```

This code connects to the Neptune.ai database to store outputs and configurations. Running the code as is requires setting up a Neptune account and API token.
Alternatively, Neptune can be avoided by running all experiments with the FileStorageObserver instead.

## Usage
First, the neuroimaging data must be preprocessed. This can be done for the desired parcellations by running the preprocessing submodule.
Preprocessing requires fMRI data and clinical data from each patient (see dependencies).
```
atlas_configs=('schaefer100')
atlas=${atlas_configs[$PBS_ARRAY_INDEX-1]}
```

python -m preprocessing.preprocess psilodep2 before $atlas

After preprocessing, the graphTRP pipeline can be run with the command:
```
python main.py ${PBS_JOBNAME} ${PBS_ARRAY_INDEX}
```

## Usage
Replicating the entire project involves three steps, each producing the dependencides of the next step. 

#### Step 1: Preprocessing
Th preprocessing submodule computes edge and node features (like FC and REACT maps) from the voxelwise fMRI data and saves a torch_geometric dataset object into project_root/data/processed. This dataset interfaces with our models. Note your fMRI data is assumed to be preprocessed (as in denoised, registered, etc) beforehand.

Dependencies:
- The voxelwise fMRI data for each subject should be stored in `data_dir/f{study}/f{session}/f"S{subject_id}"/raw_filename`. Make sure to specify data_dir and raw_filename inside utils/files.py > raw_data_dir() and get_raw_filename(), respectively.
- There should also be a file with the clinical outcome measures: data_dir/{study}/clinical_outcomes.csv
- The clinical_outcomes.csv file is expected to have a column called "Patients" with the patient ids.

Steps:
1. Download 5-HT PET data and compute REACT maps as described in preprocessing/react/react_readme.md. 
   This will create a folder `data/raw/{study}/{session}/MNI_2mm`, which is required for the preprocessing submodule.
2. Run the preprocessing submodule for each dataset:
    ```
    python -m preprocessing.preprocess study session atlas
    ```
   For example, we processed the pre-treatment (session=before) data for our primary dataset (study=psilodep2) and the validation dataset (study=psilodep1) for the three atlases: schaefer100, schaefer200, and AAL.
   All of this was run with `job_scripts/preprocessing.sh`.

#### Step 2: Model training
This submodule performs the model training and saves weights.

Dependencies:
- experiments/configs/graphTRIP.json     # Model configuration of the graphTRIP model
- experiments/configs/atlas_bound.json   # Model configuration of the atlas-bound model
- experiments/configs/hybrid.json        # Model configuration of the hybrid model
- experiments/configs/control_mlp.json   # Model configuration of the control MLP model

Steps:
1. We trained the main models as follows.
    ```
    python -m training.train_models -c experiments/configs/ -o outputs/ -s 291
    ```
   This trains the graphTRIP, atlas-bound and hybrid models and saves the weights inside `outputs/f{model_name}/weights`.
2. We trained models on psilodep1 (the validation dataset) without pretraining. This step is only necessary for replicating the results in Fig.4d of the paper. This was done with the job scripts `job_scripts/train_graphTRIP_on_psilodep1.sh` and `job_scripts/train_mlp_on_psilodep1.sh`.

#### Step 3: Post-hoc analysis


#### Step 4 (optional): Figure making
All figures were produced with `figures.ipynb`. 
The notebook depends on the results from the previous steps.


## Citation
When using this code, we'd be grateful if you cite us: Tolle et al. (2025). Cheers!
