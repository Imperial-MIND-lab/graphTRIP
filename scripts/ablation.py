"""
This script trains the ablation models.

Model ablations replace one component of graphTRIP and keep every input domain:
- Ablate VGAE: train an MLP on PCA-reduced neuroimaging features
- Ablate VGAE: train an MLP on t-SNE-reduced neuroimaging features
- Ablate MLP: train a VGAE with a linear regression head

Feature ablations keep the model and remove an input domain. Together with the full
models they span clinical {in, out} x REACT node features {in, out}, plus the two
clinical-only models, which have no neuroimaging input at all.

Dependencies:
- experiments/configs/graphtrip.json
- experiments/configs/medusa_graphtrip.json

Outputs:
- outputs/ablation/pca_benchmark/
- outputs/ablation/tsne_benchmark/
- outputs/ablation/vgae_linreg_head/
- outputs/ablation/feature_ablation/control_mlp_raw/         clinical only
- outputs/ablation/feature_ablation/linreg_on_clinical_data/ clinical only
- outputs/ablation/feature_ablation/no_node_features/        FC + clinical
- outputs/ablation/feature_ablation/no_clinical_features/    FC + REACT + arm
- outputs/ablation/feature_ablation/no_react_no_clinical/    FC + arm
- outputs/medusa_ablation/no_node_features/                  FC + clinical
- outputs/medusa_ablation/no_clinical_features/              FC + REACT
- outputs/medusa_ablation/no_react_no_clinical/              FC only

Every model saves its weights, so that any of them can be transferred zero-shot onto
the validation cohort.

Author: Hanna M. Tolle
Date: 2025-12-04
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import copy
import argparse
from utils.files import add_project_root
from utils.annotations import load_annotations
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


MEDUSA_CONFIG_FILE = 'experiments/configs/medusa_graphtrip.json'
DEFAULT_OUTPUT_BASE = 'outputs/ablation'
MEDUSA_OUTPUT_BASE = 'outputs/medusa_ablation'

# Feature ablations are kept apart from the model ablations, because they are read
# together as one factorial design. Every Medusa ablation is a feature ablation, so the
# Medusa outputs stay flat.
FEATURE_ABLATION_SUBDIR = 'feature_ablation'


def control_mlp_config(config):
    '''
    Config for an MLP trained on clinical data only, without any neuroimaging input.
    '''
    config_updates = copy.deepcopy(config)

    # Remove vgae_model and irrelevant training configs
    del config_updates['vgae_model']
    del config_updates['num_z_samples']
    del config_updates['alpha']

    # Make sure we don't use any transforms on data that doesn't exist
    config_updates['dataset']['edge_tfm_type'] = None
    config_updates['dataset']['edge_tfm_params'] = {}
    config_updates['dataset']['add_3Dcoords'] = False
    config_updates['dataset']['standardise_x'] = False

    # Set neuroimaging and context attributes to empty lists
    config_updates['dataset']['node_attrs'] = []
    config_updates['dataset']['edge_attrs'] = []
    config_updates['dataset']['context_attrs'] = []
    config_updates['dataset']['graph_attrs_to_standardise'] = []

    return config_updates


def main(config_file, output_dir, verbose, debug, seed, jobid=-1, config_id=0,
         medusa_output_dir=None):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")

    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)

    # Load medusa config (used for jobs 7 and 8)
    medusa_config_file = add_project_root(MEDUSA_CONFIG_FILE)
    if not os.path.exists(medusa_config_file):
        raise FileNotFoundError(f"{medusa_config_file} not found")
    medusa_config = load_configs_from_json(medusa_config_file)
    medusa_config = fetch_job_config(medusa_config, 0)
    medusa_output_base = add_project_root(medusa_output_dir or MEDUSA_OUTPUT_BASE)

    # Feature ablation output directory
    feature_dir = os.path.join(output_dir, FEATURE_ABLATION_SUBDIR)

    # Experiment settings
    observer = 'FileStorageObserver'
    if debug:
        config['num_epochs'] = 2
        medusa_config['num_epochs'] = 2

    # Check valid job ID input
    valid_jobids = list(range(11)) + [-1]
    if jobid not in valid_jobids:
        raise ValueError(f"Invalid job ID: {jobid}. Must be one of {valid_jobids}.")

    # 1. Control MLP benchmark -----------------------------------------------
    # Train MLP on clinical data only, without neuroimaging features
    if jobid == 0 or jobid == -1:
        exname = 'train_mlp'
        ex_dir = os.path.join(feature_dir, 'control_mlp_raw', f'seed_{seed}')
        if os.path.exists(add_project_root(ex_dir)):
            print(f"Experiment {exname} already exists in {ex_dir}")
        else:
            # Use the same ingredient configs as the main model
            config_updates = control_mlp_config(config)

            # Add more config to the config_updates
            config_updates['save_weights'] = True
            config_updates['output_dir'] = ex_dir    
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose

            # Run experiment
            run(exname, observer, config_updates)
            
    # 2. PCA benchmark -------------------------------------------------------
    # Train MLP on PCA-reduced neuroimaging features, without VGAE
    if jobid == 1 or jobid == -1:
        exname = 'pca_benchmark'
        ex_dir = os.path.join(output_dir, 'pca_benchmark', f'seed_{seed}')

        # Check if the experiment has already been run
        if os.path.exists(add_project_root(ex_dir)):
            print(f"Experiment {exname} already exists in {ex_dir}")
        else:
            # Use the same training, MLP and dataset configs as the main model
            config_updates = {}
            config_updates['balance_attrs'] = config['balance_attrs']
            config_updates['lr'] = config['lr']
            config_updates['num_epochs'] = config['num_epochs']
            config_updates['mlp_model'] = copy.deepcopy(config['mlp_model'])
            config_updates['dataset'] = copy.deepcopy(config['dataset'])

            # Make sure the dataset has no edge transform
            config_updates['dataset']['edge_tfm_type'] = None
            config_updates['dataset']['edge_tfm_params'] = {}

            # Add PCA and training configs
            config_updates['n_components'] = 32
            config_updates['lr'] = config['lr']
            config_updates['num_epochs'] = config['num_epochs']
            
            # Add more config to the config_updates
            config_updates['save_weights'] = True
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            
            # Run experiment
            run(exname, observer, config_updates)

    # 3. t-SNE benchmark -----------------------------------------------------
    # Train MLP on t-SNE-reduced neuroimaging features, without VGAE
    if jobid == 2 or jobid == -1:
        exname = 'tsne_benchmark'
        ex_dir = os.path.join(output_dir, 'tsne_benchmark', f'seed_{seed}')

        # Check if the experiment has already been run
        if os.path.exists(add_project_root(ex_dir)):
            print(f"Experiment {exname} already exists in {ex_dir}")
        else:
            # Use the same training, MLP and dataset configs as the main model
            config_updates = {}
            config_updates['balance_attrs'] = config['balance_attrs']
            config_updates['lr'] = config['lr']
            config_updates['num_epochs'] = config['num_epochs']
            config_updates['mlp_model'] = copy.deepcopy(config['mlp_model'])
            config_updates['dataset'] = copy.deepcopy(config['dataset'])

            # Make sure the dataset has no edge transform
            config_updates['dataset']['edge_tfm_type'] = None
            config_updates['dataset']['edge_tfm_params'] = {}

            # Add PCA and training configs
            config_updates['n_components'] = 3 # that's the max for sklearn t-SNE
            config_updates['perplexity'] = 30
            config_updates['lr'] = config['lr']
            config_updates['num_epochs'] = config['num_epochs']
            
            # Add more config to the config_updates
            config_updates['save_weights'] = True
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            
            # Run experiment
            run(exname, observer, config_updates)

    # 4. Linear regression head benchmark -------------------------------------
    # Train VGAE with a linear regression head end-to-end
    if jobid == 3 or jobid == -1:
        exname = 'train_jointly'
        ex_dir = os.path.join(output_dir, 'vgae_linreg_head', f'seed_{seed}')

        # Check if the experiment has already been run
        if not os.path.exists(ex_dir):
            config_updates = copy.deepcopy(config)
            config_updates['mlp_model']['params']['num_layers'] = 1 # no hidden layers
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"Train VGAE with regression head experiment already exists in {ex_dir}.")

    # 5. Linear regression on clinical data benchmark ---------------------------
    # Train a linear regression model on clinical data only
    if jobid == 4 or jobid == -1:
        exname = 'train_linreg_on_clinical'
        ex_dir = os.path.join(feature_dir, 'linreg_on_clinical_data', f'seed_{seed}')
        if not os.path.exists(add_project_root(ex_dir)):
            config_updates = {}

            # Dataset configs
            config_updates['dataset'] = copy.deepcopy(config['dataset'])
            config_updates['dataset']['batch_size'] = -1 # linear regression uses full batch
            config_updates['dataset']['graph_attrs_to_standardise'] = []

            # Other configs
            config_updates['regression_model'] = 'LinearRegression'
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"Train linear regression on clinical data experiment already exists in {ex_dir}.")

    # 6. graphTRIP without node features ----------------------------------------
    # Train graphTRIP without node features (only conditional node feats)
    if jobid == 5 or jobid == -1:
        exname = 'train_jointly'
        ex_dir = os.path.join(feature_dir, 'no_node_features', f'seed_{seed}')
        if not os.path.exists(add_project_root(ex_dir)):
            config_updates = copy.deepcopy(config)
            config_updates['dataset']['node_attrs'] = []
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"graphTRIP without node features experiment already exists in {ex_dir}.")

    # 7. graphTRIP without clinical features ------------------------------------
    if jobid == 6 or jobid == -1:
        exname = 'train_jointly'
        ex_dir = os.path.join(feature_dir, 'no_clinical_features', f'seed_{seed}')
        if not os.path.exists(add_project_root(ex_dir)):
            config_updates = copy.deepcopy(config)
            config_updates['dataset']['graph_attrs'] = ['Condition']
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"graphTRIP without clinical features experiment already exists in {ex_dir}.")

    # 8. medusa_graphTRIP without node features ---------------------------------
    if jobid == 7 or jobid == -1:
        exname = 'train_cfrnet'
        ex_dir = os.path.join(medusa_output_base, 'no_node_features', f'seed_{seed}')
        if not os.path.exists(ex_dir):
            config_updates = copy.deepcopy(medusa_config)
            config_updates['dataset']['node_attrs'] = []
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"medusa_graphTRIP without node features experiment already exists in {ex_dir}.")

    # 9. medusa_graphTRIP without clinical features -----------------------------
    if jobid == 8 or jobid == -1:
        exname = 'train_cfrnet'
        ex_dir = os.path.join(medusa_output_base, 'no_clinical_features', f'seed_{seed}')
        if not os.path.exists(ex_dir):
            config_updates = copy.deepcopy(medusa_config)
            config_updates['dataset']['graph_attrs'] = [] # Medusa never takes condition; separate heads per condtion
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"medusa_graphTRIP without clinical features experiment already exists in {ex_dir}.")

    # 10. graphTRIP with neither node nor clinical features -----------------------
    # The FC-only cell of the feature ablation design.
    if jobid == 9 or jobid == -1:
        exname = 'train_jointly'
        ex_dir = os.path.join(feature_dir, 'no_react_no_clinical', f'seed_{seed}')
        if not os.path.exists(add_project_root(ex_dir)):
            config_updates = copy.deepcopy(config)
            config_updates['dataset']['node_attrs'] = []
            config_updates['dataset']['graph_attrs'] = ['Condition']
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"graphTRIP without node or clinical features experiment already exists in {ex_dir}.")

    # 11. medusa_graphTRIP with neither node nor clinical features ----------------
    if jobid == 10 or jobid == -1:
        exname = 'train_cfrnet'
        ex_dir = os.path.join(medusa_output_base, 'no_react_no_clinical', f'seed_{seed}')
        if not os.path.exists(add_project_root(ex_dir)):
            config_updates = copy.deepcopy(medusa_config)
            config_updates['dataset']['node_attrs'] = []
            config_updates['dataset']['graph_attrs'] = [] # Medusa never takes condition
            config_updates['output_dir'] = ex_dir
            config_updates['seed'] = seed
            config_updates['verbose'] = verbose
            config_updates['save_weights'] = True
            run(exname, observer, config_updates)
        else:
            print(f"medusa_graphTRIP without node or clinical features experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python ablation.py -c experiments/configs/graphtrip.json -o outputs/ablation/ -s 0 -v -dbg -j 0 -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Path to the output directory. Defaults to '
                             f'{DEFAULT_OUTPUT_BASE}/, in which case the Medusa jobs write '
                             f'to {MEDUSA_OUTPUT_BASE}/. If given explicitly, the Medusa '
                             'jobs write to <output_dir>/medusa/ instead, so that -o is '
                             'never silently ignored.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-j', '--jobid', type=int, default=-1,
                        help='Job ID (0-10). If -1, runs all jobs sequentially.')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # The Medusa jobs
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_BASE
        medusa_output_dir = MEDUSA_OUTPUT_BASE
    else:
        medusa_output_dir = os.path.join(args.output_dir, 'medusa')

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
        medusa_output_dir = os.path.join(medusa_output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.jobid,
         args.config_id, medusa_output_dir)
