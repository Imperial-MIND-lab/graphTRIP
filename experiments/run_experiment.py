'''
Main entry point for running sacred experiments.

Author: Hanna M. Tolle
License: BSD 3-Clause
'''

import sys
sys.path.append('../')
import os
from typing import Dict
import argparse

import neptune
from neptune.integrations.sacred import NeptuneObserver
from sacred.observers import FileStorageObserver

from utils.configs import *
from utils.files import *
from utils.helpers import *


def run(exname: str, observer: str, config_updates: Dict = {}):
    '''
    Allows running a sacred experiment from within python using the given configs.
    Parameters:
    ----------
    exname (str):   The name of the sacred experiment to run.
    observer (str): The observer to use for logging the experiment outputs.
    config_updates (Dict): Configuration updates to pass to the experiment.
    '''
    # Check if config inputs are valid
    match_config = load_match_config(exname)
    if match_config is not None:
        config_updates = match_config(config_updates)

    # Define default output directory and update configs
    run_name = config_updates.get('run_name', 'default')
    output_dir = config_updates.get('output_dir', None)
    config_updates['run_name'] = run_name
    config_updates['output_dir'] = output_dir or os.path.join('outputs', 'runs', exname, run_name)
    output_dir = add_project_root(config_updates['output_dir'])

    # Load the sacred experiment
    ex = load_experiment(exname)
    ex.observers = [] # Clear previous observers

    # Set up observers and run experiment
    if observer == 'NeptuneObserver':

        # Set up Neptune observer to monitor experiment
        api_token = os.getenv('NEPTUNE_API_TOKEN')
        npt_run = neptune.init_run(
                project="graphTRP/graphTRP", 
                api_token=api_token, 
                tags=sys.argv[1])
        ex.observers.append(NeptuneObserver(run=npt_run))

        # Run the experiment
        r = ex.run(config_updates=config_updates)
        npt_run.stop()

        # Save the config as a json file, if it doesn't exist
        if not os.path.exists(os.path.join(output_dir, 'config.json')):
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(config_updates, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))

    elif observer == 'FileStorageObserver':
        # Set up FileStorage observer to save experiment outputs to disk
        ex.observers.append(FileStorageObserver(output_dir, 
                                                copy_artifacts=True, 
                                                copy_sources=False))
        # Run the experiment
        r = ex.run(config_updates=config_updates)

        # FileStorageObserver will save all outputs within a new folder output_dir/r._id
        # To avoid having duplicate files, copy everything in output_dir and delete r._id/
        run_dir = os.path.join(output_dir, str(r._id))
        move_files_into_parentdir(parent_dir=output_dir, sub_dir=run_dir)
        if not files_missing(output_dir, ['config.json', 'run.json']):
            remove_dir(run_dir)
        else:
            print("Some files are missing. Cannot remove run directory.")
        
    else:
        raise ValueError("Invalid observer. Choose 'NeptuneObserver' or 'FileStorageObserver'.")    
    print(f"Experiment completed. Outputs saved in {output_dir}.")
    
def main():
    '''
    Main function to run a sacred experiment from the command line.

    Example commands:
    ----------------
    python -m experiments.run_experiment joint_vgae_mlp FileStorageObserver
    python -m experiments.run_experiment joint_vgae_mlp FileStorageObserver --jobid=0 --config_json='joint_vgae_mlp_config_ranges.json' 
    python -m experiments.run_experiment joint_vgae_mlp NeptuneObserver --jobid=0 --config_json='joint_vgae_mlp_config_ranges.json'
    python -m experiments.run_experiment joint_vgae_mlp FileStorageObserver --output_dir='outputs/runs/exname/my_run'

    Output directories are created as follows:
    ------------------------------------------
    - Run without any optional arguments: outputs/runs/exname/default/
    - Run with --output_dir specified: uses the provided path
    - Run with --jobid specified: appends job_id/ to the output path
    - Run with --config_json specified: outputs/runs/exname/config_name/
    '''

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a Sacred experiment.')
    parser.add_argument('exname', type=str, help='The name of the sacred experiment to run.')
    parser.add_argument('observer', type=str, help='The observer to use for logging the experiment outputs.')
    parser.add_argument('--output_dir', type=str, default=None, help='The name of the output directory (for FileStorageObserver).')
    parser.add_argument('--jobid', type=int, default=None, help='Job index (for array jobs).')
    parser.add_argument('--config_json', type=str, default=None, help='JSON file with config updates or ranges, located in experiments/configs/.')
    args = parser.parse_args()
    
    if args.config_json is None:
        # Run with default configurations
        config_updates = {}
        config_name = 'default'

    else:
        # Load the configurations from the json file
        json_file = os.path.join(project_root(), 'experiments', 'configs', args.config_json)
        config_ranges = load_configs_from_json(json_file)
        config_name = args.config_json.split('.')[0]

        # Fetch configs for the specific job
        if args.jobid is not None:
            config_updates = fetch_job_config(config_ranges, args.jobid)
            config_updates['jobid'] = args.jobid
        else:
            config_updates = config_ranges

    # Add run_name and output_dir
    config_updates['run_name'] = config_updates.get('run_name', config_name)
    if args.output_dir is not None:
        config_updates['output_dir'] = args.output_dir
    else:
        default_output_dir = os.path.join('outputs', 'runs', args.exname, config_updates['run_name'])
        config_updates['output_dir'] = config_updates.get('output_dir', default_output_dir)

    if args.jobid is not None:
        config_updates['output_dir'] = os.path.join(config_updates['output_dir'], f'job_{args.jobid}')
    
    # Set up experiment and observer
    run(args.exname, args.observer, config_updates)


if __name__ == '__main__':
    main()
