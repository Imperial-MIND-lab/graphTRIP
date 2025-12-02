"""
This scripts trains t-learners (separate models for each condition) 
by loading a pre-trained VGAE and training MLP regression heads on the 
latent representations + clinical data for each condition separately.

Dependencies:
- experiments/configs/tlearners.json

Outputs:
- outputs/x_graphtrip/tlearners_torch/seed_{seed}/

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
import time
import pandas as pd
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run
from utils.helpers import aggregate_prediction_results
from utils.plotting import true_vs_pred_scatter_multi, true_vs_pred_scatter, regression_scatter

# Helper functions ----------------------------------------------------------------
def aggregate_tlearner_final_metrics(tlearner_base_dir: str) -> pd.DataFrame:
    """
    Iterates over all seed directories in tlearner_base_dir and loads the four final_metrics CSVs:
    - final_metrics_E_from_P_avg.csv
    - final_metrics_E_same_cond.csv
    - final_metrics_P_from_E_avg.csv
    - final_metrics_P_same_cond.csv

    Adds a column "Evaluation" to indicate the type, and concatenates them together.

    Returns:
        pd.DataFrame: All metrics, each row corresponding to one Evaluation/seed/file.
    """
    metrics_suffixes = [
        ("E_from_P_avg", "final_metrics_E_from_P_avg.csv"),
        ("E_same_cond", "final_metrics_E_same_cond.csv"),
        ("P_from_E_avg", "final_metrics_P_from_E_avg.csv"),
        ("P_same_cond", "final_metrics_P_same_cond.csv"),
    ]
    data_rows = []

    # Iterate over all f'seed_{seed}' dirs inside tlearner_base_dir
    for entry in os.listdir(tlearner_base_dir):
        seed_dir = os.path.join(tlearner_base_dir, entry)
        if not os.path.isdir(seed_dir) or not entry.startswith("seed_"):
            continue
        for suffix, filename in metrics_suffixes:
            file_path = os.path.join(seed_dir, filename)
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            if df.shape[0] != 1:
                continue  # Expect single-row
            df = df.copy()
            df["Evaluation"] = suffix
            df["SeedDir"] = entry  # optionally annotate which seed for traceability
            data_rows.append(df)

    if data_rows:
        result = pd.concat(data_rows, ignore_index=True)
    else:
        result = pd.DataFrame()  # Return empty DF if nothing found
    return result

# Main function ------------------------------------------------------------------
def main(config_file, output_dir, verbose, num_seeds, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)
    vgae_base_dir = os.path.join(output_dir, 'vgae_weights')

    # Add config subdirectory into output directory, if config_id is provided
    output_dir = os.path.join(output_dir, f'config_{config_id}')

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    
    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    init_seed = 0

    # Experiment directories
    tlearner_base_dir = os.path.join(output_dir, 'tlearners')
    single_cate_base_dir = os.path.join(output_dir, 'single_cate_models')
    two_cate_base_dir = os.path.join(output_dir, 'two_cate_models')

    # Run experiments with num_seeds seeds --------------------------------------------
    start_time = time.time()
    for seed in range(init_seed, init_seed + num_seeds):
        # Update seed in config
        config['seed'] = seed

        # Directory with pre-trained VGAE weights
        vgae_weights_dir = os.path.join(vgae_base_dir, f'seed_{seed}')
        assert os.path.exists(vgae_weights_dir), f"VGAE weights directory {vgae_weights_dir} does not exist"

        # Train T-learners ------------------------------------------------------------
        exname = 'train_tlearners'
        ex_dir = os.path.join(tlearner_base_dir, f'seed_{seed}')
        if not os.path.exists(ex_dir):     
            config_updates = copy.deepcopy(config)     
            config_updates['output_dir'] = ex_dir
            config_updates['save_weights'] = False
            config_updates['weights_dir'] = vgae_weights_dir
            run(exname, observer, config_updates)
        else:
            print(f"T-learner experiment already exists in {ex_dir}.")
        tlearner_dir = ex_dir

        # Train S-CATE model ----------------------------------------------------------
        exname = 'train_scate'
        ex_dir = os.path.join(single_cate_base_dir, f'seed_{seed}')
        if not os.path.exists(ex_dir):
            config_updates = {}
            config_updates['dataset'] = copy.deepcopy(config['dataset'])

            # S-CATE model settings
            config_updates['prediction_head_type'] = 'Ridge'
            config_updates['n_pca_components'] = 0
            config_updates['standardize_data'] = True
            config_updates['n_permutations'] = 1000        

            # ITE label settings
            config_updates['dataset']['target'] = None
            config_updates['t0_pred_file'] = os.path.join(tlearner_dir, 'prediction_results_P_from_E_avg.csv')
            config_updates['t1_pred_file'] = os.path.join(tlearner_dir, 'prediction_results_E_from_P_avg.csv')
            config_updates['dataset']['graph_attrs_to_standardise'] = []

            # VGAE pooling config
            config_updates['vgae_model'] = {}
            config_updates['vgae_model']['pooling_cfg'] = {
                'model_type': 'MeanStdPooling'
            }

            # Output and VGAE weights directories
            config_updates['output_dir'] = ex_dir
            config_updates['weights_dir'] = vgae_weights_dir
            config_updates['save_weights'] = False
            config_updates['verbose'] = verbose
            config_updates['seed'] = seed
            run(exname, observer, config_updates)
        else:
            print(f"CATE model already exists in {ex_dir}.")

        # Train T-CATE models ------------------------------------------------------------
        exname = 'train_cate_models'
        ex_dir = os.path.join(two_cate_base_dir, f'seed_{seed}')
        if not os.path.exists(ex_dir):
            # Experiment-specific configurations
            config_updates = {}
            config_updates['prediction_head_type'] = 'Ridge'
            config_updates['n_pca_components'] = 0
            config_updates['standardize_data'] = True
            config_updates['n_permutations'] = 1000
            config_updates['condition_specs'] = {'cond0': 'E', 'cond1': 'P'}

            # S-CATE model settings
            config_updates['prediction_head_type'] = 'Ridge'
            config_updates['n_pca_components'] = 0
            config_updates['standardize_data'] = True
            config_updates['n_permutations'] = 1000        

            # ITE label settings
            config_updates['dataset'] = copy.deepcopy(config['dataset'])
            config_updates['dataset']['target'] = None
            config_updates['t0_pred_file'] = os.path.join(tlearner_dir, 'prediction_results_P_from_E_avg.csv')
            config_updates['t1_pred_file'] = os.path.join(tlearner_dir, 'prediction_results_E_from_P_avg.csv')
            config_updates['dataset']['graph_attrs_to_standardise'] = []

            # VGAE pooling config
            config_updates['vgae_model'] = {}
            config_updates['vgae_model']['pooling_cfg'] = {
                'model_type': 'MeanStdPooling'
            }

            # Output and VGAE weights directories
            config_updates['output_dir'] = ex_dir
            config_updates['weights_dir'] = vgae_weights_dir
            config_updates['save_weights'] = False
            config_updates['verbose'] = verbose
            config_updates['seed'] = seed
            run(exname, observer, config_updates)
        else:
            print(f"CATE model experiment already exists in {ex_dir}.")

    # ------------------------------------------------------------------------------
    # Evaluation across seeds
    # ------------------------------------------------------------------------------

    # 1) T-learner evaluation ------------------------------------------------------
    same_cond_E_filename = 'prediction_results_E_same_cond.csv'
    same_cond_P_filename = 'prediction_results_P_same_cond.csv'
    other_cond_E_filename = 'prediction_results_E_from_P_avg.csv'
    other_cond_P_filename = 'prediction_results_P_from_E_avg.csv'
    ypreds_list_both = []
    titles_both = []
    target = config['dataset']['target']
    
    # E condition, same-condition
    same_cond_e_file = os.path.join(tlearner_base_dir, same_cond_E_filename)
    ypreds_e_same = aggregate_prediction_results(results_file=same_cond_e_file)
    ypreds_list_both.append(ypreds_e_same)
    titles_both.append(f"{target} same (Train: E, Test: E)")

    # E condition, other-condition (predict P from E)
    other_cond_e_file = os.path.join(tlearner_base_dir, other_cond_P_filename)
    ypreds_e_other = aggregate_prediction_results(results_file=other_cond_e_file)
    ypreds_list_both.append(ypreds_e_other)
    titles_both.append(f"{target} other (Train: E, Test: P)")

    # P condition, same-condition
    same_cond_p_file = os.path.join(tlearner_base_dir, same_cond_P_filename)
    ypreds_p_same = aggregate_prediction_results(results_file=same_cond_p_file)
    ypreds_list_both.append(ypreds_p_same)
    titles_both.append(f"{target} same (Train: P, Test: P)")

    # P condition, other-condition (predict E from P)
    other_cond_p_file = os.path.join(tlearner_base_dir, other_cond_E_filename)
    ypreds_p_other = aggregate_prediction_results(results_file=other_cond_p_file)
    ypreds_list_both.append(ypreds_p_other)
    titles_both.append(f"{target} other (Train: P, Test: E)")

    # Plot and save combined
    save_path = os.path.join(tlearner_base_dir, f'{target}_same_and_other.png')
    true_vs_pred_scatter_multi(ypreds_list_both, titles=titles_both, save_path=save_path, max_cols=2)

    # Aggregate final metrics
    final_metrics = aggregate_tlearner_final_metrics(tlearner_base_dir)
    save_path = os.path.join(tlearner_base_dir, 'final_metrics.csv')
    final_metrics.to_csv(save_path, index=False)

    # 2) S-CATE model evaluation ---------------------------------------------------
    results_file = os.path.join(single_cate_base_dir, 'prediction_results.csv')
    results = aggregate_prediction_results(results_file=results_file)
    save_path = os.path.join(single_cate_base_dir, 'imputed_vs_pred_ITEs.png')
    true_vs_pred_scatter(results, save_path=save_path, title=f"{target} (S-CATE)")

    # 3) T-CATE model evaluation ---------------------------------------------------
    tau0_filename = 'prediction_results_tau0.csv'
    tau1_filename = 'prediction_results_tau1.csv'
    titles = ['tau0', 'tau1']
    ypreds_list = []

    # Load tau0 results
    tau0_results_file = os.path.join(two_cate_base_dir, tau0_filename)
    tau0_results = aggregate_prediction_results(results_file=tau0_results_file)
    ypreds_list.append(tau0_results)

    # Load tau1 results
    tau1_results_file = os.path.join(two_cate_base_dir, tau1_filename)
    tau1_results = aggregate_prediction_results(results_file=tau1_results_file)
    ypreds_list.append(tau1_results)

    # Plot and save
    save_path = os.path.join(two_cate_base_dir, 'tau0_and_tau1_ITE_preds.png')
    true_vs_pred_scatter_multi(ypreds_list, titles=titles, save_path=save_path)  

    # 4) Sanity check: compare t-learner and two-CATE models ------------------------
    mu0_same = pd.read_csv(os.path.join(tlearner_base_dir, 'prediction_results_E_same_cond.csv'))   # trained on E, evaluated on E
    mu1_same = pd.read_csv(os.path.join(tlearner_base_dir, 'prediction_results_P_same_cond.csv'))   # trained on P, evaluated on P
    mu1_other = pd.read_csv(os.path.join(tlearner_base_dir, 'prediction_results_E_from_P_avg.csv')) # trained on P, evaluated on E
    mu0_other = pd.read_csv(os.path.join(tlearner_base_dir, 'prediction_results_P_from_E_avg.csv')) # trained on E, evaluated on P

    # Concatenate mu0_same and mu0_other, keeping only 'subject_id' and 'prediction' columns
    mu0_all = pd.concat([
        mu0_same[['subject_id', 'prediction']],
        mu0_other[['subject_id', 'prediction']]
    ], ignore_index=True)

    # Concatenate mu1_same and mu1_other, keeping only 'subject_id' and 'prediction' columns
    mu1_all = pd.concat([
        mu1_same[['subject_id', 'prediction']],
        mu1_other[['subject_id', 'prediction']]
    ], ignore_index=True)

    # Load CATE model predictions
    tau0_results = pd.read_csv(os.path.join(two_cate_base_dir, 'prediction_results_tau0.csv'))
    tau1_results = pd.read_csv(os.path.join(two_cate_base_dir, 'prediction_results_tau1.csv'))
    combined_cate_filename = 'prediction_results_cate_weighted_avg.csv'
    combined_cate_file = os.path.join(two_cate_base_dir, combined_cate_filename)
    combined_cate = aggregate_prediction_results(results_file=combined_cate_file)

    # Sort by subject_id
    mu0_all = mu0_all.sort_values('subject_id')
    mu1_all = mu1_all.sort_values('subject_id')
    combined_cate = combined_cate.sort_values('subject_id')

    # If Y1(x) for psilocybin patients is roughly constant (we see that it's clustered for many patients), we should expect the following correlations:
    # 1) tau1 vs mu0 (should be negative, significant)
    df = pd.DataFrame({'tau1': combined_cate['tau1_prediction'].values, 
                       'mu0': mu0_all['prediction'].values})
    save_path = os.path.join(two_cate_base_dir, 'tau1_vs_mu0.png')
    regression_scatter(df, show_ci=False, equal_aspect=True, ycol='mu0', xcol='tau1', save_path=save_path)

    # 2) tau0 vs mu0 (should be negative, significant)
    df = pd.DataFrame({'tau0': combined_cate['tau0_prediction'].values, 
                       'mu0': mu0_all['prediction'].values})
    save_path = os.path.join(two_cate_base_dir, 'tau0_vs_mu0.png')
    regression_scatter(df, show_ci=False, equal_aspect=True, ycol='mu0', xcol='tau0', save_path=save_path)

    # 3) tau0 vs tau1 (should be positive, significant)
    df = pd.DataFrame({'tau0': combined_cate['tau0_prediction'].values, 
                       'tau1': combined_cate['tau1_prediction'].values})
    save_path = os.path.join(two_cate_base_dir, 'tau0_vs_tau1.png')
    regression_scatter(df, show_ci=False, equal_aspect=True, ycol='tau1', xcol='tau0', save_path=save_path)

    # Print time
    end_time = time.time()
    print(f"Done! Elapsed time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    """
    How to run:
    python tlearners.py -c experiments/configs/tlearners.json -o outputs/x_graphtrip/ -s 0 -v -dbg -j 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/sklearn_head_screen.json', 
                        help='Path to the config file with t-learner config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/x_graphtrip/', help='Path to the output directory')
    parser.add_argument('-ns', '--num_seeds', type=int, default=1, help='Number of seeds')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-ci', '--config_id', type=int, default=0, help='Config ID')
    args = parser.parse_args()

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.num_seeds, args.config_id)
