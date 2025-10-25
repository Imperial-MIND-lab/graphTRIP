"""
Functions for computing metrics from neuroimaging data.

License: BSD-3-Clause
Author: Hanna M. Tolle
Date: 2024-10-31
"""

import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
from typing import List
import glob 

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
import networkx as nx
import torch
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.ndimage import center_of_mass

from utils.files import *
from utils.annotations import *


# ======================================================================================= #
# HELPER FUNCTIONS

def get_atlas(atlas):
    '''
    Fetches brain parcellation (a.k.a. atlas) from nilearn.

    Parameters:
    ----------
    atlas (str): Name of atlas to load.

    '''

    available = ['schaefer100', 'schaefer200', 'aal']
    if atlas not in available:
        raise ValueError("Invalid name. Valid options are: " + ", ".join(available))

    if atlas == 'schaefer100':
        return datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2, yeo_networks=7)

    elif atlas == 'schaefer200':
        return datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2, yeo_networks=7)
    
    elif atlas == 'aal':
        return datasets.fetch_atlas_aal()
    
def load_3d_coords(atlas, drop_roi_name=True):
    '''Loads the 3D coordinates of the nodes in the brain graph.'''
    coord_file_dir = os.path.join(project_root(), 'data', 'raw', 'spatial_coordinates')
    if atlas == 'schaefer100':
        file_name = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
        coords = pd.read_csv(os.path.join(coord_file_dir, file_name))
    elif atlas == 'schaefer200':
        file_name = 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
        coords = pd.read_csv(os.path.join(coord_file_dir, file_name))
    elif atlas == 'aal':
        coords = compute_roi_centers(atlas)
    else:
        raise ValueError(f'Atlas {atlas} currently not supported.')
    # Always drop ROI Label
    coords = coords.drop(columns=['ROI Label'])
    # Drop ROI Name if requested
    if drop_roi_name:
        coords = coords.drop(columns=['ROI Name'])
    return coords

def compute_roi_centers(atlas):
    """
    Computes the center of mass (in MNI space) for each region in the atlas.
    
    Returns:
    --------
    centers_df: pandas DataFrame with columns 'ROI', 'R', 'A', 'S' containing the 
               region labels and their MNI coordinates.
    """
    # Fetch the atlas using helper function
    atlas = get_atlas(atlas)
    
    # Load the atlas image from the file path stored in the 'maps' attribute
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    
    # Get unique region labels (typically, label 0 is the background)
    region_labels = np.unique(atlas_data)
    region_labels = region_labels[region_labels != 0]  # Exclude background
    region_names = [label.decode('utf-8') if isinstance(label, bytes) else label for label in atlas.labels]
    
    # Lists to store coordinates
    roi_labels = []
    r_coords = []
    a_coords = []
    s_coords = []
    
    for label in region_labels:
        # Compute the center of mass in voxel coordinates for the current label
        voxel_com = center_of_mass(atlas_data == label)
        # Convert voxel coordinates to MNI space using the affine matrix
        mni_com = nib.affines.apply_affine(affine, voxel_com)
        
        # Store coordinates and label
        roi_labels.append(int(label))
        r_coords.append(mni_com[0])
        a_coords.append(mni_com[1]) 
        s_coords.append(mni_com[2])
    
    # Create dataframe
    centers_df = pd.DataFrame({
        'ROI Label': roi_labels,
        'ROI Name': region_names,
        'R': r_coords,
        'A': a_coords,
        'S': s_coords})
    
    return centers_df
    
def get_rsn_mapping(atlas: str):
    '''Returns a mapping from ROI labels to RSNs for a specific atlas in Nilearn.'''
    if atlas in ['schaefer100', 'schaefer200']:
        # Load the atlas and fetch the ROI labels
        atlas_data = get_atlas(atlas)
        roi_labels = atlas_data.labels

        # RSN names in the nilearn atlas
        rsn_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

        # Convert nilearn roi_labels to strings (from bytes)
        roi_labels = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in roi_labels])

        # Create the mapping array
        mapping = []
        for roi_label in roi_labels:
            for idx, rsn_name in enumerate(rsn_names):
                if rsn_name in roi_label:
                    mapping.append(idx)
                    break

        # Translate names to what I'm used to
        rsn_names = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
        return np.array(mapping), rsn_names
    
    elif atlas == 'aal':
        return None, None
    
    else:
        raise ValueError(f'Atlas {atlas} currently not supported.')


def compute_metrics(bold, atlas, feature_file, feature_funs):
    '''
    Helper function that applies multiple functions to the data and
    writes them to the disk, making sure that already computed metrics
    aren't re-computed and overwritten.

    Parameters:
    ----------
    bold (nd array): BOLD data numpy array.
    atlas (str): Name of the brain atlas used to parcellate the brain graph.
    feature_file (str): Filepath/filename of the feature file.
    feature_funs (list): List of functions to apply to data.

    '''
    # Check if the feature file already exists
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        features = df.to_dict('list')
    else:
        features = {}

    # Compute new features, unless they already exist
    features_updated = False
    for i, fun in enumerate(feature_funs):
        feature_names = fun(bold=None, atlas=atlas)  # Get the names of the features
        # If any of the features are missing, compute them
        if not all(feature_name in features for feature_name in feature_names):
            features_updated = True
            new_features = fun(bold, atlas) # Output is a dictionary
            for feature_name in feature_names:
                features[feature_name] = new_features[feature_name].flatten()

    # Write features to disk as pandas dataframe to include headers
    if features_updated:
        new_features = pd.DataFrame(features)
        new_features.to_csv(feature_file, sep=',', index=False)
        print(f'Updated features {feature_file} saved.')

def compute_single_metric(feature_file, feature_fun, data, **kwargs):
    '''
    Helper function that applies a single function to a custom input.

    Parameters:
    ----------
    bold (nd array): BOLD data numpy array.
    atlas (str): Name of the brain atlas used to parcellate the brain graph.
    feature_file (str): Filepath/filename of the feature file.
    feature_funs (list): List of functions to apply to data.

    '''
    # Check if the feature file already exists
    if os.path.exists(feature_file):
        df = pd.read_csv(feature_file)
        features = df.to_dict('list')
    else:
        features = {}

    # Compute new features
    features_updated = False
    try:
        # Try to get feature names without data
        feature_names = feature_fun(None, **kwargs)
        # If any features are missing, compute them
        if not all(feature_name in features for feature_name in feature_names):
            features_updated = True
            new_features = feature_fun(data, **kwargs)
            for feature_name in feature_names:
                features[feature_name] = new_features[feature_name].flatten()
    except:
        # If getting feature names fails, compute all features and override
        print(f"Warning: Computing all features for {feature_file}, possibly overwriting old features.")
        features_updated = True
        new_features = feature_fun(data, **kwargs)
        # Get feature names from computed features
        for feature_name, feature_values in new_features.items():
            features[feature_name] = feature_values.flatten()

    # Write features to disk as pandas dataframe to include headers
    if features_updated:
        new_features = pd.DataFrame(features)
        new_features.to_csv(feature_file, sep=',', index=False)
        print(f'Updated features {feature_file} saved.')

def get_node_feature_funs():
    '''Returns all currently defined node feature functions.'''
    funs = [lempel_ziv]
            #receptor_enriched_roimap]
    return funs

def get_node_feature_names(atlas) -> List[str]:
    '''Returns the names of all currently defined node feature functions.'''
    funs = get_node_feature_funs()
    names = []
    for fun in funs:
        names.extend(fun(bold=None, atlas=atlas))
    # Tentativ fix:
    receptor_sets = ['Believeau-5', 'Believeau-3']
    for receptor_set in receptor_sets:
        receptor_names = get_react_receptor_names(receptor_set)
        names += [f'{receptor}_{receptor_set}' for receptor in receptor_names]
    return names

def get_react_receptor_names(receptor_set):
    '''Returns the names of the receptors in a given receptor set.'''
    react_dir = os.path.join(project_root(), 'data', 'raw', 'psilodep2', 'before', 'MNI_2mm')
    receptor_set_dir = os.path.join(react_dir, f'REACT_{receptor_set}')
    with open(os.path.join(receptor_set_dir, 'input_maps.txt'), 'r') as file:
        receptor_names = [line.split(',')[0] for line in file]
    return receptor_names

def get_edge_feature_funs():
    '''Returns all currently defined edge feature functions.'''
    funs = [functional_connectivity,
            granger_causality]
    return funs

def get_edge_feature_names() -> List[str]:
    '''Returns the names of all currently defined edge feature functions.'''
    funs = get_edge_feature_funs()
    names = [fun.__name__ for fun in funs]
    return names
    

# ======================================================================================= #
# DATA DERIVED FROM THE RAW NIFTI FILES

def parcellate(raw_file, atlas):
    '''
    Aggregates voxelwise data (e.g. 4D fMRI BOLD or 3D PET images) 
    into parcels according to a brain atlas.

    Parameters:
    ----------
    raw_file (str): filepath/filename to the voxelwise data.
    atlas (str): Name of the brain atlas to be used.

    Returns:
    -------
    parcellated_data (array): Parcellated data. 
        shape: (numROIs, numTimeSteps) if 4D input, or (numROIs, ) if 3D input.
    '''

    # Make sure the raw file exists
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f'Raw file {raw_file} not found.')

    # Fetch brain atlas (parcellation)
    atlas_map = get_atlas(atlas)
    atlas_filename = atlas_map['maps']

    # Create masker for aggregating voxel signals into brain regions
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, verbose=5)

    # Parcellate the raw data
    parcellated_data = masker.fit_transform(raw_file)

    return parcellated_data

def parcellate_5HTmaps(pet_atlas_dir, gm_mask_nifti, atlas):
    '''
    Parcellates the 5HT maps (nii.gz files) into brain regions according to a brain atlas.

    Parameters:
    ----------
    pet_atlas_dir (str): filepath to the directory containing subdirectories with the 5HT maps.
    gm_mask_nifti (str): filepath/filename to the GM mask.
    atlas (str): Name of the brain atlas to be used.

    Returns:
    -------
    parcellated_maps (dict): Dictionary mapping receptor names to parcellated PET data arrays.
    '''
    # Make sure the input files exist
    if not os.path.exists(pet_atlas_dir):
        raise FileNotFoundError(f'PET atlas directory {pet_atlas_dir} not found.')
    if not os.path.exists(gm_mask_nifti):
        raise FileNotFoundError(f'GM mask file {gm_mask_nifti} not found.')

    # Find all 5HT subdirectories
    receptor_dirs = glob.glob(os.path.join(pet_atlas_dir, "5-HT*")) + glob.glob(os.path.join(pet_atlas_dir, "5HT*"))
    if not receptor_dirs:
        raise FileNotFoundError(f'No 5HT directories found in {pet_atlas_dir}')

    # Load GM mask
    gm_mask = nib.load(gm_mask_nifti)
    gm_mask_data = gm_mask.get_fdata() > 0  # Convert to boolean mask

    # First load all PET maps to create a common mask
    pet_maps = {}
    common_mask = gm_mask_data.copy()  # Start with GM mask
    
    for receptor_dir in receptor_dirs:
        receptor_name = os.path.basename(receptor_dir)
        
        # Find the normalized PET map file
        pet_files = glob.glob(os.path.join(receptor_dir, "*_masked_normalized.nii.gz"))
        if not pet_files:
            print(f"Warning: No normalized PET map found in {receptor_dir}")
            continue
        pet_file = pet_files[0]
        
        # Load PET map
        pet_img = nib.load(pet_file)
        pet_data = pet_img.get_fdata()
        
        # Update common mask to include only voxels that are:
        # 1. In gray matter
        # 2. Have non-zero, non-NaN values in this PET map
        valid_voxels = (pet_data != 0) & ~np.isnan(pet_data)
        common_mask = common_mask & valid_voxels
        
        # Store the PET data and image
        pet_maps[receptor_name] = {'data': pet_data, 'img': pet_img}

    if not pet_maps:
        raise RuntimeError("No PET maps were successfully loaded")

    # Create a new NIfTI image for the common mask
    common_mask_img = nib.Nifti1Image(
        common_mask.astype(np.int16),
        affine=gm_mask.affine)

    # Get atlas for parcellation
    atlas_map = get_atlas(atlas)
    atlas_filename = atlas_map['maps']
    
    # Create masker with the common mask
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        mask_img=common_mask_img,
        standardize=False,  # PET data is already normalized
        verbose=5)

    # Now parcellate each masked PET map
    parcellated_maps = {}
    for receptor_name, pet_map in pet_maps.items():
        try:
            # Apply common mask to PET data
            masked_data = pet_map['data'].copy()
            masked_data[~common_mask] = 0
            
            # Create new NIfTI image with masked data
            masked_img = nib.Nifti1Image(
                masked_data,
                affine=pet_map['img'].affine)
            
            # Parcellate the masked data
            parcellated_data = masker.fit_transform(masked_img)
            parcellated_maps[receptor_name] = parcellated_data.flatten()
            
        except Exception as e:
            print(f"Error processing {receptor_name}: {str(e)}")
            continue

    if not parcellated_maps:
        raise RuntimeError("No PET maps were successfully parcellated")

    return parcellated_maps

# ======================================================================================= #
# NODE FEATURE FUNCTIONS
# Functions that return Nx1 metrics, derived from BOLD data.

def LZ76(ss):
    """
    Calculate Lempel-Ziv's algorithmic complexity using the LZ76 algorithm
    and the sliding-window implementation.

    Reference:

    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the
    complexity of spatiotemporal patterns", Physical Review A, Volume 36,
    Number 2 (1987).

    Input:
      ss -- array of integers

    Output:
      c  -- integer
    """

    ss = ss.flatten().tolist()
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(ss)
    while True:
        if ss[i + k - 1] == ss[l + k - 1]:
            k = k + 1
            if l + k > n:
                c = c + 1
                break
        else:
            if k > k_max:
               k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c


def lempel_ziv(bold = None, atlas = None):
    '''Calls LZ76 to compute lempel ziv for each regional time series in bold-shaped data.

    Parameters: 
    ----------
    bold (nd array): BOLD time series data with shape (numTimeSteps, numROIs).
    '''
    # If function is called without arguments, return the name of the feature
    if bold is None:
        return ['lempel_ziv']

    # Get number of times steps and regions
    numTRs, numROIs = bold.shape

    # Compute lempel ziv complexity for each brain region
    output = np.zeros(numROIs)
    for roi, ts in enumerate(bold.T):
        binarised = ts>np.median(ts)
        output[roi] = LZ76(binarised)*np.log2(len(binarised))/len(binarised)

    return {'lempel_ziv': output}

def molecular_enriched_roimap(bold, molecular_regressors):
    '''
    Computes molecular enriched maps as explained in Lawn et al., 2023, but for
    brain regions rather than voxels.

    Parameters:
    ----------
    bold (nd array): BOLD time series data with shape (numTimeSteps, numROIs).
    molecular_regressors (pd DataFrame): DataFrame with molecular regressors.

    Returns:
    -------
    roi_map (dict): Dictionary with molecular names as keys and molecular enriched maps as values.
    '''
    # Make sure bold dimensions match molecular regressors dimensions
    if bold.shape[1] != molecular_regressors.shape[0]:
        raise ValueError("Number of ROIs in bold and molecular regressors must match.")

    numTRs, numROIs = bold.shape
    numRegressors = molecular_regressors.shape[1]
    regressor_names = molecular_regressors.columns.tolist()

    # GLM 1: regress each BOLD time point against the molecular regressors
    regressor_timeseries = np.zeros((numTRs, numRegressors))
    X = molecular_regressors.to_numpy()
    for t in range(numTRs):
        # Fit the linear model
        y = bold[t, :]
        model = LinearRegression()
        model.fit(X, y)
        # Save the coefficients of each molecular regressor
        regressor_timeseries[t, :] = model.coef_ # shape: (numRegressors, )

    # GLM 2: regress each molecular timeseries against the BOLD timeseries of each ROI
    roimaps = np.zeros((numROIs, numRegressors))
    X = regressor_timeseries
    for roi in range(numROIs):
        y = bold[:, roi]
        model = LinearRegression()
        model.fit(X, y)
        roimaps[roi, :] = model.coef_ # shape: (numRegressors, )
        
    # Add regressor names to the roimaps
    roimaps = {regressor: roimaps[:, i] for i, regressor in enumerate(regressor_names)}

    return roimaps

def receptor_enriched_maps(react_subject_dir, atlas, receptor_set='Believeau-3'):
    '''
    Computes receptor enriched maps as explained in Lawn et al., 2023. Requires
    prior computation of voxelwise receptor-enriched maps (e.g. using the react-fmri
    toolbox).

    Parameters:
    ----------
    react_subject_dir (str): Directory containing the receptor-enriched maps of 
        a single subject (e.g. S01_react_stage2_map1.nii.gz).
    atlas (str): Name of the brain atlas used to parcellate the brain graph.

    Returns:
    -------
    roi_maps (dict): Dictionary with receptor names as keys and receptor enriched maps as values.
    '''
    # Get the receptor names from input_maps.txt
    with open(os.path.join(os.path.dirname(react_subject_dir), 'input_maps.txt'), 'r') as file:
        receptor_names = [line.split(',')[0] for line in file]

    # React's names for the receptors are map1, map2, etc.
    map_names = {f'map{i+1}': receptor for i, receptor in enumerate(receptor_names)}

    # Load each map (nii.gz file) in the directory and parcellate it
    receptor_maps = {}
    for map_name, receptor in map_names.items():
        # Get the map file
        pattern = os.path.join(react_subject_dir, f'*_react_stage2_{map_name}.nii.gz')
        matching_files = glob.glob(pattern)
        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        # Parcellate the map
        map_file = matching_files[0]
        parcellated_map = parcellate(map_file, atlas)
        receptor_maps[f'{receptor}_{receptor_set}'] = parcellated_map
        
    return receptor_maps

# ======================================================================================= #
# EDGE FEATURE FUNCTIONS 
# Functions that return NxN metrics, derived from BOLD data.

def functional_connectivity(bold = None, atlas = None, connectivity_kind='correlation'):
    '''Computes functional connectivity from (BOLD) time series data.'''
    # If function is called without arguments, return the name of the feature
    if bold is None:
        return ['functional_connectivity']
    
    # Define connectivity measure
    connectivity_measure = ConnectivityMeasure(kind=connectivity_kind)

    # Compute FC and add self-loops (default)
    output = connectivity_measure.fit_transform([bold])[0]
    np.fill_diagonal(output, 1)

    return {'functional_connectivity': output}


def granger_causality(bold = None, atlas = None, maxlag=1, test_name='ssr_chi2test'):
    '''
    Computes granger causality test between two time series and returns p-value 
    according to the specfied test. 
    (see statsmodels.tsa.stattools.grangercausalitytest)

    Parameters:
    ----------
    bold (nd array Tx2): array with two time series on the columns
    maxlag (int): time lag of expected causal effect between the two time series
    test_name (str): name of granger causality test

    Returns:
    -------
    p_value (float): p-value for the null hypothesis that there is no causal effect

    '''
    # If function is called without arguments, return the name of the feature
    if bold is None:
        return ['granger_causality']
    
    # Get number of times steps and regions and define output variable
    numTRs, numROIs = bold.shape
    output = np.ones((numROIs, numROIs))

    # Compute granger causality test for each pair of brain regions
    for i in range(numROIs):
        for j in range(numROIs):
            if i!=j:
                test_results = grangercausalitytests(bold[:, [i,j]], maxlag=[maxlag], verbose=False)
                p_value = test_results[1][0][test_name][1]
                output[i,j] = p_value

    return {'granger_causality': output}

def compute_spd(edge_index: torch.Tensor, num_nodes: int, max_dist: int = 10) -> torch.Tensor:
    '''
    Computes the shortest path distance (SPD) matrix for a given edge index.
    
    Parameters:
    ----------
    edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges)
    num_nodes (int): Number of nodes in the graph
    max_dist (int): Maximum distance to consider

    Returns:
    --------
    spd (torch.Tensor): Shortest path distance matrix of shape (num_nodes, num_nodes)
    '''
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    spd = torch.full((num_nodes, num_nodes), fill_value=max_dist + 1)
    for i in range(num_nodes):
        for j, d in lengths[i].items():
            spd[i, j] = min(d, max_dist) 
    return spd

# ======================================================================================= #
# GRAPH FEATURE FUNCTIONS 
# Functions that return scalars, derived from adjacency matrices.

def compute_modularity(adj, seed=291):
    '''
    Compute the modularity index Q for the weighted FC graph.
    Parameters:
    ----------
    adj (2D numpy array): The weighted FC graph/ adjacency.
    '''
    G = nx.from_numpy_array(adj)
    communities = nx.community.louvain_communities(G, seed=seed)
    return nx.community.modularity(G, communities)

def compute_modularity_torch(adj: torch.Tensor, communities, resolution: float = 1.0) -> torch.Tensor:
    """
    Computes the modularity Q for a given weighted adjacency matrix and community assignments,
    matching the NetworkX implementation but differentiable w.r.t adj.
    Automatically handles self-loops by masking the diagonal in a differentiable way.
    This function matches the NetworkX implementation of community.modularity(), but only
    if self-connections are zero. They are automatically masked out.
    
    Parameters:
    -----------
    adj (torch.Tensor): Weighted adjacency matrix of shape (N, N)
    communities (list of sets or numpy.ndarray): community assignments, accepts nx community format or numpy.
    resolution (float, optional): Resolution parameter gamma (default is 1.0)
        
    Returns:
    --------
    Q (torch.Tensor): The modularity score, differentiable w.r.t adj
    """
    # Convert communities to list of node indices if it's not already
    if isinstance(communities, torch.Tensor):
        unique_comms = torch.unique(communities)
        communities = [torch.where(communities == c)[0].tolist() for c in unique_comms]
    elif isinstance(communities, np.ndarray):
        unique_comms = np.unique(communities)
        communities = [np.where(communities == c)[0].tolist() for c in unique_comms]
    elif not isinstance(communities, list):
        raise ValueError("communities must be a torch.Tensor, numpy.ndarray, or list of sets/lists")

    # Create mask for non-diagonal elements
    N = adj.shape[0]
    mask = torch.ones((N, N), device=adj.device, dtype=adj.dtype)
    mask.fill_diagonal_(0)
    
    # Mask out diagonal elements in a differentiable way
    adj_no_loops = adj * mask
    
    # Calculate degrees using adjacency without self-loops
    degrees = adj_no_loops.sum(dim=1)
    m = degrees.sum() / 2
    
    Q = torch.tensor(0.0, device=adj.device, dtype=adj.dtype)
    
    for community in communities:
        comm = set(community)
        comm_idx = torch.tensor(list(comm), device=adj.device)
        
        # Get the submatrix for this community
        comm_submatrix = adj_no_loops[comm_idx][:, comm_idx]
        
        # Calculate L_c
        L_c = comm_submatrix.sum() / 2
        
        # Calculate degree sum for the community
        degree_sum = degrees[comm_idx].sum()
        
        # Add community contribution to modularity
        Q = Q + L_c/m - resolution * (degree_sum/(2*m))**2
    
    return Q

def compute_efficiency(adj, threshold=0.5):
    '''
    Compute the global efficiency Eglob for the weighted graph.
    Eglob = (1/N(N-1)) * sum_ij(1/d(i,j)),
    where d(i,j) is the shortest path length between nodes i and j,
    and N is the number of nodes in the graph.

    Parameters:
    ----------
    adj (2D numpy array): The weighted FC graph/ adjacency.
    '''
    # Binarise adjacency matrix
    adj = np.where(adj > threshold, 1, 0)
    
    # Convert the adjacency matrix to a NetworkX graph (weighted)
    G = nx.from_numpy_array(adj, create_using=nx.Graph)

    # Compute shortest path lengths between all pairs of nodes
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    N = len(G.nodes)
    efficiency_sum = 0.0

    # Sum the inverse of the path lengths for all pairs of nodes
    for i in range(N):
        for j in range(i+1, N):  # Only consider pairs i < j to avoid double counting
            if i != j:
                d_ij = path_lengths[i].get(j, np.inf)  # Get the shortest path distance
                if d_ij != np.inf:  # Ensure nodes i and j are connected
                    efficiency_sum += 1 / d_ij

    # Compute the global efficiency
    Eglob = (2 / (N * (N - 1))) * efficiency_sum

    return Eglob

def compute_withinRSN_conn(adj, atlas):
    '''
    Compute the mean within-RSN connectivity strength for the adjacency matrix.
    Parameters:
    ----------
    adj (2D numpy array): The weighted FC graph/ adjacency.
    atlas (str): Name of the atlas used to parcellate the brain graph.

    Returns:
    -------
    withinRSN_fc (dict): Maps the name of each RSN to the average
        FC strength of within-RSN connections.
    '''

    # Get the RSN mapping (RSN assignments for each node)
    rsn_mapping, rsn_names = get_rsn_mapping(atlas)
    rsn_mapping = np.array(rsn_mapping)

    # Compute the mean within-RSN connectivity strength for each RSN
    np.fill_diagonal(adj, np.nan) # Exclude self-connections
    mean_withinRSN_fc = {rsn: np.nanmean(adj[rsn_mapping == rsn, :][:, rsn_mapping == rsn]) for rsn in rsn_names}

    return mean_withinRSN_fc

def compute_rsn_conn(adj, rsn_mapping):
    '''
    Compute the mean RSN connectivity (between and within).
    Parameters:
    ----------
    adj (2D numpy array): The weighted FC graph/ adjacency.
    atlas (str): Name of the atlas used to parcellate the brain graph.

    Returns:
    -------
    rsn_conn (2D array): Matrix with average FC strength of 
                         between RSNs i and j.
    '''
    # Compute the mean between-RSN connectivity strength for each RSN pair
    n_rsn = max(rsn_mapping) + 1
    rsn_conn = np.zeros((n_rsn, n_rsn))
    
    for rsn_i in range(n_rsn):
        row_idx = np.where(rsn_mapping == rsn_i)[0]
        for rsn_j in range(rsn_i, n_rsn):
            col_idx = np.where(rsn_mapping == rsn_j)[0]
            if rsn_i == rsn_j:
                # For within-RSN connectivity, exclude diagonal elements (because they're all 1)
                mask = ~np.eye(len(row_idx), dtype=bool)
                rsn_conn[rsn_i, rsn_j] = np.mean(adj[row_idx][:, col_idx][mask])
            else:
                rsn_conn[rsn_i, rsn_j] = np.mean(adj[row_idx][:, col_idx])
            rsn_conn[rsn_j, rsn_i] = rsn_conn[rsn_i, rsn_j]
    
    return rsn_conn

def compute_dist_adjacency(atlas):
    """
    Compute Euclidean adjacency matrix between brain regions based on their 3D coordinates.
    
    Parameters:
    -----------
    atlas (str): Name of the atlas used to parcellate the brain graph.
        
    Returns:
    --------
    distances (np.ndarray): Square matrix of shape (num_regions, num_regions) containing pairwise Euclidean distances
    """
    # Load the coordinates
    coord_array = load_3d_coords(atlas).values # (num_regions, 3)
    num_regions = coord_array.shape[0]

    # Initialize distance matrix
    distances = np.zeros((num_regions, num_regions))
    
    # Compute pairwise Euclidean distances
    for i in range(num_regions):
        for j in range(num_regions):
            distances[i,j] = np.sqrt(np.sum((coord_array[i] - coord_array[j])**2))
    
    return distances
