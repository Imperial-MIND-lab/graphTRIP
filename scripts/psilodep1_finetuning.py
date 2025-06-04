"""
This script trains models on the Psilodep1 dataset (the validation dataset).
Performs pre-training and finetuning.

Dependencies:
- experiments/configs/graphtrip.json
- experiments/configs/psilodep1_finetuning.json

Outputs:
- outputs/psilodep1/pretrained_weights/
- outputs/psilodep1/finetuned_weights/

Author: Hanna M. Tolle
Date: 2025-05-31
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
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


def main(config_file1, config_file2, output_dir, 
         verbose, debug, seed, config_id1=0, config_id2=0):
    # Add project root to paths
    config_file1 = add_project_root(config_file1)
    config_file2 = add_project_root(config_file2)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file1):
        raise FileNotFoundError(f"{config_file1} not found")
    if not os.path.exists(config_file2):
        raise FileNotFoundError(f"{config_file2} not found")
    
    # Load the config
    psilodep2_config = load_configs_from_json(config_file1)
    psilodep2_config = fetch_job_config(psilodep2_config, config_id1)
    psilodep1_config = load_configs_from_json(config_file2)
    psilodep1_config = fetch_job_config(psilodep1_config, config_id2)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    psilodep1_config['seed'] = seed
    psilodep1_config['verbose'] = verbose
    psilodep2_config['seed'] = seed
    psilodep2_config['verbose'] = verbose
    if debug:
        psilodep2_config['num_epochs'] = 2
        psilodep1_config['num_epochs'] = 2

    # 1. Pretraining ----------------------------------------------------------
    exname = 'train_jointly'
    pretraining_dir = os.path.join(output_dir, 'pretrained_weights')

    # Run the experiment if it doesn't exist
    if not os.path.exists(pretraining_dir):
        # Pretraining config
        config_updates = copy.deepcopy(psilodep2_config)
        config_updates['output_dir'] = pretraining_dir
        config_updates['save_weights'] = True

        # Make sure the model is compatible with psilodep1
        incompatible = ['Condition', 'Stop_SSRI']
        new_attrs = [attr for attr in psilodep2_config['dataset']['graph_attrs'] if attr not in incompatible]
        config_updates['dataset']['graph_attrs'] = new_attrs
        config_updates['dataset']['context_attrs'] = incompatible # add as readout-context only

        run(exname, observer, config_updates)
    else:
        print(f"Pretraining weights already exists in {pretraining_dir}.")

    # 2. Finetuning ------------------------------------------------------------
    exname = 'transfer_and_finetune'
    finetuning_dir = os.path.join(output_dir, 'finetuned_weights')
    if not os.path.exists(finetuning_dir):
        config_updates = copy.deepcopy(psilodep1_config)
        config_updates['output_dir'] = finetuning_dir
        config_updates['weights_dir'] = pretraining_dir
        run(exname, observer, config_updates)

if __name__ == "__main__":
    """
    How to run:
    python psilodep1_finetuning.py -c1 experiments/configs/graphtrip.json -c2 experiments/configs/psilodep1_finetuning.json -o outputs/ -s 291 -v -dbg -ci1 0 -ci2 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config_file1', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-c2', '--config_file2', type=str, 
                        default='experiments/configs/psilodep1_finetuning.json', 
                        help='Path to the config file with Psilodep1 finetuning config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/psilodep1/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci1', '--config_id1', type=int, default=None, help='Config ID')
    parser.add_argument('-ci2', '--config_id2', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id1 is not None:
        if args.config_id2 is not None:
            args.output_dir = os.path.join(args.output_dir, f'config1_{args.config_id1}', f'config2_{args.config_id2}')
        else:
            args.config_id2 = 0
            args.output_dir = os.path.join(args.output_dir, f'config1_{args.config_id1}')
    elif args.config_id2 is not None:
        args.config_id1 = 0
        args.output_dir = os.path.join(args.output_dir, f'config2_{args.config_id2}')
    else:
        args.config_id1 = 0
        args.config_id2 = 0

    # Run the main function
    main(args.config_file1, args.config_file2, args.output_dir, 
         args.verbose, args.debug, args.seed, args.config_id1, args.config_id2)
