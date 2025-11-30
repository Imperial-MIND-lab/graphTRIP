'''
Ingredient for data loading and creating data loaders.

License: BSD 3-Clause
Author: Hanna M. Tolle
'''

import sys
sys.path.append('../../')

from sacred import Ingredient
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from torch_geometric.seed import seed_everything
import pandas as pd
import numpy as np
import torch

from datasets import *
from preprocessing.metrics import load_3d_coords, compute_spd


# Create the ingredient --------------------------------------------------------
data_ingredient = Ingredient('dataset')

# Define configurations --------------------------------------------------------
@data_ingredient.config
def data_cfg():
    # Dataset configurations
    study = "psilodep2"
    session = "before"
    atlas = "schaefer100"
    target = "QIDS_Final_Integration" 
    prefilter = None # Filter samples based on annotations.csv; e.g., {'Condition': 'P'}

    # Get the number of nodes
    num_nodes = None
    if isinstance(atlas, str) or len(atlas) == 1:
        num_nodes = get_num_nodes(atlas)

    # Node, edge, graph attributes
    node_attrs = ["5-HT1A_Believeau-3", "5-HT2A_Believeau-3", "5-HTT_Believeau-3"]
    edge_attrs = ["functional_connectivity"]
    graph_attrs = ["QIDS_Before", "BDI_Before", "Condition", "Stop_SSRI"]
    context_attrs = ["Condition"] 

    # Edge index transform
    edge_tfm_type = None
    edge_tfm_params = {} # Needs to be specified in update_configs

    # Edge transforms
    self_loop_fill_value = 1
    transforms = []
    if self_loop_fill_value != 1:
        transforms += [RemoveSelfLoops(), 
                       AddSelfLoops(fill_value=self_loop_fill_value)]
    max_spd_dist = None # Max shortest path dist. If None, no SPD embeddings are added.
        
    # Node transforms
    add_3Dcoords = False         # Whether to add 3D coordinates as node features.
    cond_attrs = []
    if add_3Dcoords:
        cond_attrs += ["R", "A", "S"]

    # Graph transforms
    drug_condition = None # e.g. 1.0 for setting all drug conditions to psilocybin.
        
    # Dataloader configurations
    standardise_x = False # Whether to standardise the node features.
    val_split = 0.        # Fraction of training data to use for validation; 0. for no early stopping.
    num_folds = 6         # Number of folds for k-fold cross-validation. 1 for no cross-validation.
    fold_shift = 0        # Shift the fold indices by this amount.
    batch_size = 7        # Batch size.
    graph_attrs_to_standardise = [] # List of graph attributes to standardise.

# Captured functions -----------------------------------------------------------
@data_ingredient.capture
def load_data(study, session, atlas, target, prefilter,
              node_attrs, edge_attrs, graph_attrs, context_attrs,
              edge_tfm_type, edge_tfm_params, transforms, 
              add_3Dcoords, drug_condition, max_spd_dist):
    '''Loads the dataset for the given configurations.'''   
    # Create the attributes object
    attrs = Attrs(node=node_attrs, edge=edge_attrs, graph=graph_attrs)

    # Create the edge transformation object
    edge_tfm = get_edge_tfm(edge_tfm_type, edge_tfm_params)

    # Get the prefilter that is always applied
    default_prefilter = get_default_prefilter(study)

    # Get the node feature transformations
    all_tfms = [] + transforms
    all_tfms += get_x_tfms(add_3Dcoords, atlas)

    # Get graph attribute transformations
    if drug_condition is not None:
        if 'Condition' in graph_attrs:
            all_tfms += [SetGraphAttr('Condition', drug_condition)]
        else:
            print(f"Condition not in graph_attrs, so not adding drug condition to graph attributes.")

    # Load the full dataset
    dataset = BrainGraphDataset(study=study,
                                session=session,
                                atlas=atlas,
                                prefilter=default_prefilter,
                                attrs=attrs,
                                target=target,
                                edge_tfm=edge_tfm,
                                transforms=all_tfms)

    # Add SPD embeddings if max_spd_dist is not None
    if max_spd_dist is not None:
        spd_cache = {}
        for data in dataset:
            spd = compute_spd(data.edge_index, data.num_nodes, max_spd_dist)
            spd_cache[data.subject.item()+1] = spd
        dataset.transform.transforms.append(AddSPD(spd_cache))
    
    # Update context attributes in the filter transform if needed
    if context_attrs:
        for transform in dataset.transform.transforms:
            if isinstance(transform, FilterAttributesWithContext):
                transform.context_attrs = context_attrs
                break
    
    # Filter the dataset if prefilter is not None and if it differs from the default prefilter
    different_prefilter = False
    if prefilter is not None:
        for k, v in prefilter.items():
            if k not in default_prefilter:
                different_prefilter = True
                break
            if v != default_prefilter[k]:
                different_prefilter = True
                break
    if different_prefilter:
        annotations = load_annotations(study, default_prefilter)
        bool_idx = np.ones(len(annotations), dtype=bool)
        for k, v in prefilter.items():
            bool_idx = bool_idx & (annotations[k] == v)
        filter_indices = np.where(bool_idx)[0]
        dataset = dataset[filter_indices]

    # Remove all subjects with missing target values
    if target is not None:
        missing_targets = np.array([torch.isnan(graph.y).item() for graph in dataset])
        dataset = dataset[~missing_targets]

    return dataset

def load_dataset_from_configs(config):
    '''
    Helper function to load dataset from outside a 
    sacred experiment using configs.
    '''
    # Get all valid input arguments
    valid_args = ['study', 'session', 'atlas', 'target', 'prefilter',
                  'node_attrs', 'edge_attrs', 'graph_attrs', 'context_attrs',
                  'edge_tfm_type', 'edge_tfm_params', 'transforms',
                  'add_3Dcoords', 'drug_condition', 'max_spd_dist']
    input_args = {k: config[k] for k in valid_args if k in config}
    return load_data(**input_args)

@data_ingredient.capture
def load_multiple_datasets(atlases: list[str]):
    '''Loads the dataset for the given configurations.'''
    datasets = []
    for atlas in atlases:
        datasets.append(load_data(atlas=atlas))
    return datasets

@data_ingredient.capture
def get_edge_tfm(edge_tfm_type, edge_tfm_params):
    '''Returns the edge transformation object.'''
    if edge_tfm_type is None:
        return None
    edge_tfm_class = globals()[edge_tfm_type]
    edge_tfm = edge_tfm_class(**edge_tfm_params)
    return edge_tfm

@data_ingredient.capture
def get_x_tfms(add_3Dcoords, atlas):
    '''
    Returns data transofmrations for node attribute transformations that
    are shared/ normative, i.e. the same for all data samples.
    Features are added as conditional data, i.e. they are passed to the encoder
    but are not reconstructed by the decoder.
    '''
    tfms = []
    if add_3Dcoords:
        coords_tfm = AddNormNodeAttr_asConditional(features=get_3Dcoords(atlas))
        tfms.append(coords_tfm)
    return tfms

@data_ingredient.capture
def get_3Dcoords(atlas):
    return load_3d_coords(atlas)

@data_ingredient.capture
def get_context(batch):
    '''Access pre-stored context attributes.'''
    # If there are no context attributes, return an empty tensor
    if not hasattr(batch, 'context_attr') or batch.context_attr.shape[1] == 0:
        return torch.empty((batch.num_nodes, 0), dtype=torch.float32)
    # Expand to node dimension
    context = batch.context_attr[batch.batch]
    return context
                
@data_ingredient.capture
def get_context_idx(context_attrs, graph_attrs):    
    return np.array([graph_attrs.index(attr) for attr in context_attrs])

@data_ingredient.capture
def get_triu_edges(batch):
    '''Returns the upper triangular edges of the batch.'''
    edge_index = batch.edge_index               # [2, num_edges_in_batch]
    triu_mask = edge_index[0] < edge_index[1]   # [num_edges_in_batch]
    triu_edges = batch.edge_attr[triu_mask]     # [num_triu_edges, num_edge_attr]
    return triu_edges, edge_index[:, triu_mask]
    
# Accessor functions ----------------------------------------------------------
@data_ingredient.capture
def get_num_attr(attr_type, node_attrs, edge_attrs, graph_attrs, cond_attrs):
    '''Returns the number of attributes of the given type.'''
    if attr_type == 'node':
        return len(node_attrs)
    elif attr_type == 'edge':
        return len(edge_attrs)
    elif attr_type == 'graph':
        return len(graph_attrs)
    elif attr_type == 'cond':
        return len(cond_attrs)
    else:
        raise ValueError(f"Unknown attribute type: {attr_type}")
    
@data_ingredient.capture
def get_attr_names(dataset):
    """Returns the currently employed attr names."""
    return dataset[0].attr_names

@data_ingredient.capture
def get_num_triu_edges(num_nodes):
    '''Returns the number of upper triangular edges.'''
    return num_nodes * (num_nodes - 1) // 2

@data_ingredient.capture
def init_outputs_dict(dataset):
    '''Initializes the outputs dictionary.'''
    clinical_attrs = get_attr_names(dataset).graph
    return {name: [] for name in clinical_attrs+['prediction', 'label', 'subject_id']}

@data_ingredient.capture
def get_conditions(dataset, graph_attrs):
    '''Returns the conditions for the dataset.'''
    if 'Condition' not in graph_attrs:
        return None
    condition_idx = graph_attrs.index('Condition')
    conditions = [dataset[sub].graph_attr[0, condition_idx].item() for sub in range(len(dataset))]
    return conditions

@data_ingredient.capture
def update_best_outputs(best_outputs, outputs, graph_attrs):
    '''Updates the best outputs dictionary.'''
    best_outputs['prediction'].extend(outputs['prediction'])
    best_outputs['label'].extend(outputs['label'])
    best_outputs['subject_id'].extend(outputs['subject_id'])
    for clinical in outputs['clinical_data']:
        for i, name in enumerate(graph_attrs):
            best_outputs[name].append(clinical[i])

@data_ingredient.capture
def add_drug_condition_to_outputs(outputs, study):
    '''Adds the drug condition to the outputs dataframe, if it doesn't already exist.'''
    if 'Condition' not in outputs.columns:
        # Load annotations
        default_prefilter = get_default_prefilter(study)
        annotations = load_annotations(study=study, filter=default_prefilter)
        df = convert_to_numerical(annotations[['Condition', 'Patient']], ['Condition'])
        
        # Create mapping from subject_id to condition, accounting for Patient index starting at 1
        drug_cond_dict = pd.Series(df['Condition'].values, index=df['Patient']-1).to_dict()
        
        # Add conditions to outputs dataframe by matching subject_ids
        outputs['Condition'] = outputs['subject_id'].map(drug_cond_dict)
    return outputs

# Dataloader functions ---------------------------------------------------------
def get_train_mean_std(train_dataset):
    '''Returns the mean and standard deviation of the node attributes of the training set.'''
    all_features = torch.cat([data.x for data in train_dataset], dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    return mean, std

def get_graph_attrs_stats_dict(train_dataset, graph_attrs_to_standardise):
    '''Returns the mean and standard deviation of the graph attributes of the training set.'''
    stats = {}
    all_graph_attrs = train_dataset[0].attr_names.graph
    for attr in graph_attrs_to_standardise:
        attr_idx = get_list_idx(all_graph_attrs, attr)
        attr_values = [data.graph_attr[0, attr_idx].item() for data in train_dataset]
        stats[attr] = (np.mean(attr_values), np.std(attr_values))
    return stats

def reverse_standardisation(x_reconstructed, mean, std):
    '''Reverses the standardisation of the node attributes.'''
    return x_reconstructed * std + mean

@data_ingredient.capture
def get_kfold_dataloaders(dataset, num_folds, batch_size, val_split, 
                          fold_shift, standardise_x, 
                          graph_attrs_to_standardise, seed=None):
    '''
    Returns the K-fold train, validation, and test dataloaders.
    If val_split > 0, also returns the validation dataloader. Validation
    set is split off from the training set.

    Parameters:
    ----------
    dataset (BrainGraphDataset): Dataset to split into K-folds.
    seed (int): Random seed for reproducibility.
    num_folds (int): Number of folds to split the dataset into.
    batch_size (int): Batch size for the dataloaders.
    val_split (float): Fraction of the training set to use as validation set.
    fold_shift (int): Shift the fold indices by this amount.
    standardise_x (bool): Whether to standardise the node features.
    '''
    # Set the seed
    if seed is not None:
        seed_everything(seed)
    
    # Get all indices and apply the fold_shift
    all_indices = np.arange(len(dataset))
    if fold_shift > 0:
        all_indices = np.roll(all_indices, fold_shift)
    
    # Create k-fold split on the shifted indices
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    train_loaders = []
    val_loaders = []
    test_loaders = []
    test_indices = []
    mean_std = {'mean': [], 'std': []} if standardise_x else None

    # If batch_size is -1, use full batch training (no mini-batching)
    use_full_batch = (batch_size == -1)
    
    for train_index, test_index in kf.split(all_indices):
        # Convert back to original indices
        train_index = all_indices[train_index]
        test_index = all_indices[test_index]

        if val_split > 0:
            train_index, val_index = train_test_split(
                train_index, 
                test_size=val_split, 
                random_state=seed)
            val_dataset = dataset[val_index]

        train_dataset = dataset[train_index]
        test_dataset = dataset[test_index]

        # Node attribute standardisation ---------------------------------------
        if standardise_x:
            # Get the mean and standard deviation of the training set
            mean, std = get_train_mean_std(train_dataset)
            mean_std['mean'].append(mean)
            mean_std['std'].append(std)

            # Build the standardisation transform
            standardise_tfm = StandardiseNodeAttributes(mean=mean, std=std)
            train_dataset.transform = T.Compose([*train_dataset.transform.transforms, standardise_tfm])
            test_dataset.transform = T.Compose([*test_dataset.transform.transforms, standardise_tfm])

            if val_split > 0:
                val_dataset.transform = T.Compose([*val_dataset.transform.transforms, standardise_tfm])

        # Graph attribute standardisation ---------------------------------------
        if len(graph_attrs_to_standardise) > 0:
            graph_attrs_stats = get_graph_attrs_stats_dict(train_dataset, graph_attrs_to_standardise)
            standardise_graph_tfm = StandardiseGraphAttributes(stats=graph_attrs_stats)
            train_dataset.transform = T.Compose([*train_dataset.transform.transforms, standardise_graph_tfm])
            test_dataset.transform = T.Compose([*test_dataset.transform.transforms, standardise_graph_tfm])
            if val_split > 0:
                val_dataset.transform = T.Compose([*val_dataset.transform.transforms, standardise_graph_tfm])

        # Make the train, validation, and test dataloaders
        train_loaders.append(DataLoader(
            train_dataset, 
            batch_size=len(train_dataset) if use_full_batch else batch_size, 
            shuffle=True))
        test_loaders.append(DataLoader(
            test_dataset, 
            batch_size=len(test_dataset) if use_full_batch else batch_size, 
            shuffle=False))    

        # Save test indices
        test_indices.append(test_index)

        if val_split > 0:
            val_loaders.append(DataLoader(val_dataset, 
                                          batch_size=len(val_dataset) if use_full_batch else batch_size, 
                                          shuffle=False))

    return train_loaders, val_loaders, test_loaders, test_indices, mean_std

@data_ingredient.capture
def get_dataloaders_from_test_indices(dataset, test_indices, batch_size, val_split, standardise_x, seed=None):
    '''
    Returns train, validation, and test dataloaders based on test fold assignments.
    If val_split > 0, splits training data into train and validation sets.

    Parameters:
    ----------
    dataset (BrainGraphDataset): Dataset to split.
    test_indices (numpy.array): Array where test_indices[i] indicates which test fold sample i belongs to.
    batch_size (int): Batch size for the dataloaders.
    val_split (float): Fraction of the training set to use as validation set.
    standardise_x (bool): Whether to standardise the node features.
    seed (int): Random seed for reproducibility.
    '''
    if seed is not None:
        seed_everything(seed)
    
    num_folds = len(np.unique(test_indices))
    train_loaders = []
    val_loaders = []
    test_loaders = []
    mean_std = {'mean': [], 'std': []} if standardise_x else None
    batch_size = int(abs(batch_size))

    # Get all indices
    all_indices = np.arange(len(dataset))
    
    for fold in range(num_folds):
        # Get test indices for this fold
        test_idx = all_indices[test_indices == fold]
        # Get training indices (all indices not in test set)
        train_idx = all_indices[test_indices != fold]
        
        if val_split > 0:
            # Split training indices into train and validation
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_split,
                random_state=seed)
            val_dataset = dataset[val_idx]

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        if standardise_x:
            # Get the mean and standard deviation of the training set
            mean, std = get_train_mean_std(train_dataset)
            mean_std['mean'].append(mean)
            mean_std['std'].append(std)

            # Build the standardisation transform
            standardise_tfm = StandardiseNodeAttributes(mean=mean, std=std)
            train_dataset.transform = T.Compose([*train_dataset.transform.transforms, standardise_tfm])
            test_dataset.transform = T.Compose([*test_dataset.transform.transforms, standardise_tfm])

            if val_split > 0:
                val_dataset.transform = T.Compose([*val_dataset.transform.transforms, standardise_tfm])

        # Make the train, validation, and test dataloaders
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

        if val_split > 0:
            val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

    return train_loaders, val_loaders, test_loaders, mean_std

@data_ingredient.capture
def get_tsne_dataloaders(tsne_dataset, num_folds, batch_size, val_split, seed=None):
    '''
    Creates k-fold train/val/test splits for the TSNEDataset.

    Parameters:
    -----------
    tsne_dataset (TSNEDataset): torch dataset (as defined in tsne_benchmark.py) to split.
    num_folds (int): Number of folds to split the dataset into.
    batch_size (int): Batch size for the dataloaders.
    val_split (float): Fraction of the training set to use as validation set.
    seed (int): Random seed for reproducibility of splits.

    Returns:
    --------
    train_loaders, val_loaders, test_loaders, test_indices
    '''
    # Set the seed
    if seed is not None:
        seed_everything(seed)
    batch_size = int(abs(batch_size))

    train_loaders = []
    val_loaders = []
    test_loaders = []  
    test_indices = []

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed) 
    for train_index, test_index in kf.split(tsne_dataset):
        # Split the training set into train and validation
        if val_split > 0:
            train_index, val_index = train_test_split(
                train_index, 
                test_size=val_split, 
                random_state=seed)
            val_dataset = torch.utils.data.Subset(tsne_dataset, val_index)

        # Create the train and test subsets
        train_dataset = torch.utils.data.Subset(tsne_dataset, train_index)
        test_dataset = torch.utils.data.Subset(tsne_dataset, test_index)

        # Create the train, validation, and test dataloaders (torch.utils.data.DataLoader)
        train_loaders.append(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        test_loaders.append(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False))         
        test_indices.append(test_index)
        if val_split > 0:
            val_loaders.append(torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False))

    return train_loaders, val_loaders, test_loaders, test_indices

@data_ingredient.capture
def get_balanced_tsne_dataloaders(tsne_dataset, original_dataset, balance_attrs,
                                  num_folds, batch_size, val_split, seed=None):
    '''
    Creates balanced k-fold train/val/test splits for the TSNEDataset.
    
    Parameters:
    -----------
    tsne_dataset (TSNEDataset): torch dataset (as defined in tsne_benchmark.py) to split.
    original_dataset (BrainGraphDataset): The original dataset to extract attribute values from.
    balance_attrs (list): List of categorical attribute names to balance on, e.g., ['Condition', 'Gender']
    num_folds (int): Number of folds to split the dataset into.
    batch_size (int): Batch size for the dataloaders.
    val_split (float): Fraction of the training set to use as validation set.
    seed (int): Random seed for reproducibility of splits.
    
    Returns:
    --------
    train_loaders, val_loaders, test_loaders, test_indices
    '''
    # Set the seed
    if seed is not None:
        seed_everything(seed)
    batch_size = int(abs(batch_size))
    
    # Extract values for each categorical attribute to be balanced
    attr_values = {}
    for attr_name in balance_attrs:
        attr_idx = original_dataset[0].attr_names.graph.index(attr_name)
        values = [data.graph_attr[0, attr_idx].item() for data in original_dataset]
        attr_values[attr_name] = values
    
    # Create stratification labels by combining all attribute values into a label per sample
    combined_labels = []
    for i in range(len(original_dataset)):
        label = '_'.join(str(attr_values[attr][i]) for attr in balance_attrs)
        combined_labels.append(label)
    
    # Use StratifiedKFold for balanced splitting
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    train_loaders = []
    val_loaders = []
    test_loaders = []
    test_indices = []
    
    for train_index, test_index in skf.split(np.zeros(len(tsne_dataset)), combined_labels):
        # Split the training set into train and validation with stratification
        if val_split > 0:
            # Extract labels for the training set
            train_labels = [combined_labels[i] for i in train_index]
            
            # Perform stratified split for validation
            train_index, val_index = train_test_split(
                train_index,
                test_size=val_split,
                stratify=train_labels,
                random_state=seed
            )
            val_dataset = torch.utils.data.Subset(tsne_dataset, val_index)
        
        # Create the train and test subsets
        train_dataset = torch.utils.data.Subset(tsne_dataset, train_index)
        test_dataset = torch.utils.data.Subset(tsne_dataset, test_index)
        
        # Create the train, validation, and test dataloaders
        train_loaders.append(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        test_loaders.append(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
        test_indices.append(test_index)
        
        if val_split > 0:
            val_loaders.append(torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
    
    return train_loaders, val_loaders, test_loaders, test_indices

@data_ingredient.capture
def get_balanced_kfold_splits(dataset, num_folds, balance_attrs, seed=None):
    '''
    Creates balanced k-fold splits considering categorical attributes.

    Parameters:
    -----------
    dataset (BrainGraphDataset): The dataset to split
    num_folds (int): Number of folds
    balance_attrs (list): List of categorical attribute names to balance on, e.g., ['Condition', 'Gender']
    seed (int): Random seed for reproducibility

    Returns:
    --------
    list of (train_idx, test_idx) tuples for each fold
    '''
    if seed is not None:
        seed_everything(seed)

    # Extract values for each categorical attribute to be balanced
    attr_values = {}
    for attr_name in balance_attrs:
        attr_idx = dataset[0].attr_names.graph.index(attr_name)
        values = [data.graph_attr[0, attr_idx].item() for data in dataset]
        attr_values[attr_name] = values

    # Create stratification labels by combining all attribute values into a label per sample
    combined_labels = []
    for i in range(len(dataset)):
        label = '_'.join(str(attr_values[attr][i]) for attr in balance_attrs)
        combined_labels.append(label)

    # Use StratifiedKFold for balanced splitting
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Get splits
    splits = []
    for train_idx, test_idx in skf.split(np.zeros(len(dataset)), combined_labels):
        splits.append((train_idx, test_idx))

    return splits

@data_ingredient.capture
def get_balanced_kfold_dataloaders(dataset, balance_attrs, 
                                   num_folds, batch_size, val_split, 
                                   standardise_x, 
                                   graph_attrs_to_standardise, seed=None):
    '''
    Returns balanced K-fold train, validation, and test dataloaders.
    
    Parameters:
    -----------
    dataset (BrainGraphDataset): Dataset to split into K-folds
    num_folds (int): Number of folds
    batch_size (int): Batch size for the dataloaders
    val_split (float): Fraction of the training set to use as validation set
    standardise_x (bool): Whether to standardise the node features
    graph_attrs_to_standardise (list): List of graph attribute names to standardise
    balance_attrs (list): List of categorical attribute names to balance on, e.g., ['Condition', 'Gender']
    seed (int): Random seed 
    '''
    # Get balanced splits
    splits = get_balanced_kfold_splits(dataset, num_folds, balance_attrs, seed=seed)
    
    # Initialize containers
    train_loaders = []
    val_loaders = []
    test_loaders = []
    test_indices = []
    mean_std = {'mean': [], 'std': []} if standardise_x else None
    
    # If batch_size is -1, use full batch training (no mini-batching)
    use_full_batch = (batch_size == -1)
    
    for train_index, test_index in splits:
        if val_split > 0:
            # For validation split, maintain balance in the same attributes
            train_data = dataset[train_index]
            train_attrs = {attr: [data.graph_attr[0, train_data[0].attr_names.graph.index(attr)].item() 
                                for data in train_data] 
                         for attr in balance_attrs}
            
            # Create stratification labels for validation split
            train_labels = []
            for i in range(len(train_data)):
                label = tuple(train_attrs[attr][i] for attr in balance_attrs)
                train_labels.append(label)
            
            # Perform stratified split for validation
            train_index, val_index = train_test_split(
                train_index,
                test_size=val_split,
                stratify=[train_labels[i] for i in range(len(train_index))],
                random_state=seed
            )
            val_dataset = dataset[val_index]
        
        train_dataset = dataset[train_index]
        test_dataset = dataset[test_index]
        
        # Node attribute standardisation ---------------------------------------
        if standardise_x:
            # Get the mean and standard deviation of the training set
            mean, std = get_train_mean_std(train_dataset)
            mean_std['mean'].append(mean)
            mean_std['std'].append(std)

            # Build the standardisation transform
            standardise_tfm = StandardiseNodeAttributes(mean=mean, std=std)
            train_dataset.transform = T.Compose([*train_dataset.transform.transforms, standardise_tfm])
            test_dataset.transform = T.Compose([*test_dataset.transform.transforms, standardise_tfm])

            if val_split > 0:
                val_dataset.transform = T.Compose([*val_dataset.transform.transforms, standardise_tfm])

        # Graph attribute standardisation ---------------------------------------
        if len(graph_attrs_to_standardise) > 0:
            graph_attrs_stats = get_graph_attrs_stats_dict(train_dataset, graph_attrs_to_standardise)
            standardise_graph_tfm = StandardiseGraphAttributes(stats=graph_attrs_stats)
            train_dataset.transform = T.Compose([*train_dataset.transform.transforms, standardise_graph_tfm])
            test_dataset.transform = T.Compose([*test_dataset.transform.transforms, standardise_graph_tfm])
            if val_split > 0:
                val_dataset.transform = T.Compose([*val_dataset.transform.transforms, standardise_graph_tfm])
        
        # Create dataloaders
        train_loaders.append(DataLoader(
            train_dataset,
            batch_size=len(train_dataset) if use_full_batch else batch_size,
            shuffle=True))
        test_loaders.append(DataLoader(
            test_dataset,
            batch_size=len(test_dataset) if use_full_batch else batch_size,
            shuffle=False))
        test_indices.append(test_index)
        
        if val_split > 0:
            val_loaders.append(DataLoader(
                val_dataset,
                batch_size=len(val_dataset) if use_full_batch else batch_size,
                shuffle=False))
    
    return train_loaders, val_loaders, test_loaders, test_indices, mean_std
