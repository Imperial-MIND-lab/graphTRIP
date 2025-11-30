'''
BrainGraphDataset class for loading and processing brain graph data.

License: BSD 3-Clause
Date: 2024-10-30
Author: Hanna M. Tolle
'''

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

sys.path.append('../')
from utils.annotations import *
from utils.files import *
from preprocessing.metrics import get_edge_feature_names, get_node_feature_names


# Data structure for node, edge, and graph attributes -----------------------------------

@dataclass
class Attrs:
    node: List[str] = field(default_factory=list)
    edge: List[str] = field(default_factory=list)
    graph: List[str] = field(default_factory=list)

    def add_clinical_graph_attrs(self, study:str):
        if study=='psilodep2':
            self.graph += ['QIDS_Before', 'BDI_Before', 'Condition', 'Gender', 'Stop_SSRI']
        elif study=='psilodep1':
            self.graph += ['Gender', 'Age', 'HAMD_Before', 'QIDS_Before', 'LOTR_Before', 'BDI_Before', 'Condition', 'Stop_SSRI']
        else:
            print(f'Clinical features for {study} unknown. \n'
                  'Edit datasets.Attrs.add_clinical_graph_attrs() to add your study.')

    def is_identical(self, attrs: Any) -> bool:
        '''Check if attrs matches the fields of self.'''
        if not isinstance(attrs, Attrs):
            return False
        return self.node == attrs.node and self.edge == attrs.edge and self.graph == attrs.graph


# Graph data transformations ----------------------------------------------------------
# BaseTransforms take a data object, and return a modified copy of it.
    
class LowerAbsThresholdAdjacency(BaseTransform):

    def __init__(self, threshold: float, edge_info: str):
        self.threshold = threshold
        self.edge_info = edge_info

    def forward(self, data: Data) -> Data:
        # Find the edge attribute to be thresholded
        attr_idx = get_list_idx(data.attr_names.edge, self.edge_info)
        # Get boolean indices according to thresholding
        bool_idx = torch.abs(data.edge_attr[:, attr_idx]) > self.threshold
        # Apply boolean indices to edge indices and attributes
        data.edge_index = data.edge_index[:, bool_idx]
        data.edge_attr = data.edge_attr[bool_idx, :]
        return data
    
class DensityThresholdAdjacency(BaseTransform):
    """
    Thresholds edges based on a target network density.
    For a given target density, it calculates the appropriate threshold value
    that will result in approximately that density when applied to the absolute
    values of the edge weights.
    
    Parameters:
    -----------
    density : float
        Target network density (between 0 and 1)
    edge_info : str
        Name of the edge attribute to threshold
    """
    def __init__(self, density: float, edge_info: str):
        assert 0 < density < 1, "Density must be between 0 and 1"
        self.density = density
        self.edge_info = edge_info

    def forward(self, data: Data) -> Data:
        # Find the edge attribute to be thresholded
        attr_idx = get_list_idx(data.attr_names.edge, self.edge_info)
        
        # Get absolute edge weights
        abs_weights = torch.abs(data.edge_attr[:, attr_idx])
        
        # Calculate number of edges to keep based on target density
        num_edges_to_keep = int(self.density * len(abs_weights))
        
        # Get threshold value that will keep the desired number of edges
        if num_edges_to_keep > 0:
            # Get the threshold value that will keep the desired number of edges
            threshold = torch.kthvalue(abs_weights, len(abs_weights) - num_edges_to_keep + 1).values
            
            # Get boolean indices for edges to keep
            bool_idx = abs_weights >= threshold
            
            # Apply thresholding
            data.edge_index = data.edge_index[:, bool_idx]
            data.edge_attr = data.edge_attr[bool_idx, :]
        else:
            # If density is too low, keep no edges and print a warning
            print(f'Density is too low for {self.edge_info} edge attribute. Returning empty graph.')
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.edge_attr = torch.empty((0, data.edge_attr.shape[1]), dtype=data.edge_attr.dtype)
            
        return data
    
class ApplyAdjacency(BaseTransform):
    '''
    Applies external adjacency information to each data object,
    and updates the edge indices and attributes accordingly.
    For example: use normative structural connectivity as adjacency.
    '''
    def __init__(self, adj_file: str):
        # Load external adjacency matrix
        adj_file = os.path.join(project_root(), adj_file)
        external_adjacency = torch.tensor(pd.read_csv(adj_file, header=None).values)
        
        # Get indices of non-zero entries in adjacency matrix
        rows, cols = torch.nonzero(external_adjacency, as_tuple=True)
        self.edge_index = torch.stack([rows, cols], dim=0)
        
        # Create mapping from (row, col) to edge index in fully connected graph
        num_nodes = external_adjacency.shape[0]
        self.edge_mapping = rows * num_nodes + cols
        
    def forward(self, data: Data) -> Data:
        # Extract edge attributes using the precomputed mapping
        data.edge_attr = data.edge_attr[self.edge_mapping]
        data.edge_index = self.edge_index
        return data
    
class FilterAttributesWithContext(BaseTransform):
    def __init__(self, desired_attrs: Attrs, 
                 context_attrs: List[str] = None):
        self.desired_attrs = desired_attrs
        self.context_attrs = context_attrs if context_attrs is not None else []

    def forward(self, data: Data) -> Data:
        updated_attr_names = Attrs(node=[], edge=[], graph=[])
        
        # Handle node attributes -----------------------------------------------------
        if len(self.desired_attrs.node) == 0:
            # If no node attributes desired, create empty tensor with correct first dimension
            data.x = torch.empty((data.x.shape[0], 0), dtype=data.x.dtype)
            data.num_node_features = 0
            updated_attr_names.node = []

        elif data.attr_names.node != self.desired_attrs.node:
            feature_to_idx = {feat: idx for idx, feat in enumerate(data.attr_names.node)}
            col_idx = torch.tensor([feature_to_idx[feat] for feat in self.desired_attrs.node])
            data.x = data.x[:, col_idx]
            data.num_node_features = data.x.shape[1]
            updated_attr_names.node = self.desired_attrs.node

        else:
            updated_attr_names.node = data.attr_names.node

        # Handle edge attributes -----------------------------------------------------
        if len(self.desired_attrs.edge) == 0:
            # If no edge attributes desired, create empty tensor with correct first dimension
            data.edge_attr = torch.empty((data.edge_attr.shape[0], 0), dtype=data.edge_attr.dtype)
            data.num_edge_features = 0
            updated_attr_names.edge = []

        elif data.attr_names.edge != self.desired_attrs.edge:
            feature_to_idx = {feat: idx for idx, feat in enumerate(data.attr_names.edge)}
            col_idx = torch.tensor([feature_to_idx[feat] for feat in self.desired_attrs.edge])
            data.edge_attr = data.edge_attr[:, col_idx]
            data.num_edge_features = data.edge_attr.shape[1]
            updated_attr_names.edge = self.desired_attrs.edge

        else:
            updated_attr_names.edge = data.attr_names.edge

        # Handle context attributes (part of graph_attr in the unfiltered data) ---------
        if self.context_attrs:
            feature_to_idx = {feat: idx for idx, feat in enumerate(data.attr_names.graph)}
            col_idx = torch.tensor([feature_to_idx[feat] for feat in self.context_attrs])
            data.context_attr = data.graph_attr[:, col_idx]
            data.context_names = self.context_attrs

        else:
            data.context_attr = torch.empty((1, 0), dtype=data.graph_attr.dtype)
            data.context_names = []

        # Handle graph attributes -----------------------------------------------------
        if len(self.desired_attrs.graph) == 0:
            # If no graph attributes desired, create empty tensor with correct first dimension
            data.graph_attr = torch.empty((1, 0), dtype=data.graph_attr.dtype)
            updated_attr_names.graph = []

        elif data.attr_names.graph != self.desired_attrs.graph:
            feature_to_idx = {feat: idx for idx, feat in enumerate(data.attr_names.graph)}
            col_idx = torch.tensor([feature_to_idx[feat] for feat in self.desired_attrs.graph])
            data.graph_attr = data.graph_attr[:, col_idx]
            updated_attr_names.graph = self.desired_attrs.graph
            
        else:
            updated_attr_names.graph = data.attr_names.graph

        # Update attribute names
        data.attr_names = updated_attr_names

        return data
    
class StandardiseNodeAttributes(BaseTransform):
    """Transform to standardise node attributes using pre-computed mean and std."""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def forward(self, data: Data) -> Data:
        if self.mean is None or self.std is None:
            return data
        data.x = (data.x - self.mean) / self.std
        return data
    
class AddLabel(BaseTransform):
    '''
    Adds lables y to each data object.
    Parameter:
    ---------
    labels (dict): Dictionary that hashes the Patient_id to the label.
    
    Note:
    Patient_id is 1-indexed, as in annotations.csv.
    subject is 0-indexed, as in Data.subject.item().
    '''

    def __init__(self, labels: Dict):
        self.labels = labels

    def forward(self, data: Data) -> Data:
        data.y = torch.tensor(self.labels[data.subject.item()+1])
        return data

class AddSPD(BaseTransform):
    """
    Adds precomputed SPD matrices to each data object.
    Expects a dictionary {patient_id: spd_tensor}.
    """
    def __init__(self, spd_dict: Dict):
        self.spd_dict = spd_dict

    def forward(self, data: Data) -> Data:
        patient_id = data.subject.item() + 1
        spd = self.spd_dict[patient_id]
        data.spd = spd
        return data
    
class SetGraphAttr(BaseTransform):
    '''
    Sets the graph attribute of each data object to a new value.
    '''
    def __init__(self, attr_name: str, 
                 attr_value: float):
        self.attr_name = attr_name
        self.attr_value = attr_value

    def forward(self, data: Data) -> Data:
        attr_idx = get_list_idx(data.attr_names.graph, self.attr_name)
        data.graph_attr[:, attr_idx] = self.attr_value
        return data
    
class StandardiseGraphAttributes(BaseTransform):
    """Transform to standardise graph attributes using pre-computed mean and std."""
    def __init__(self, stats: Union[Dict[str, tuple], pd.DataFrame] = None):
        """
        Parameters:
        -----------
        stats : Dict[str, tuple] or pd.DataFrame, optional
            If dict: Maps graph attribute names to (mean, std) tuples.
                Example: {'QIDS_Before': (15.5, 5.2), 'BDI_Before': (20.3, 8.1)}
            If DataFrame: Must have columns ['feature', 'mean', 'std'] or use feature names
                as index with 'mean' and 'std' columns.
                Example: pd.DataFrame({'mean': [15.5, 20.3], 'std': [5.2, 8.1]}, 
                                     index=['QIDS_Before', 'BDI_Before'])
        
        Note:
        -----
        The graph attributes to standardise are inferred from the keys/index of stats.
        Attributes with std=0 will be skipped to avoid division by zero.
        """
        if stats is None:
            self.stats = {}
        elif isinstance(stats, pd.DataFrame):
            # Convert DataFrame to dict
            if 'feature' in stats.columns:
                # DataFrame has 'feature' column
                self.stats = {row['feature']: (row['mean'], row['std']) 
                             for _, row in stats.iterrows()}
            else:
                # DataFrame has feature names as index
                self.stats = {idx: (row['mean'], row['std']) 
                             for idx, row in stats.iterrows()}
        else:
            self.stats = stats

    def forward(self, data: Data) -> Data:
        if not self.stats:
            return data
        
        for attr_name, (mean, std) in self.stats.items():
            attr_idx = get_list_idx(data.attr_names.graph, attr_name)           
            if std == 0:
                continue  # Skip if std is zero to avoid division by zero
            data.graph_attr[:, attr_idx] = (data.graph_attr[:, attr_idx] - mean) / std
        
        return data
    
class AddNormNodeAttr(BaseTransform):
    '''
    Adds normative (i.e. same for all subjects) node attributes 
    x to each data object.
    Parameter:
    ---------
    features (df): Pandas dataframe with feature names and data. 
    '''

    def __init__(self, features: pd.DataFrame):
        self.features = torch.tensor(features.to_numpy(), dtype=torch.float32)
        self.feature_names = features.columns.to_list()

    def forward(self, data: Data) -> Data:
        # Concatenate data in features with data.x
        data.x = torch.cat((data.x, self.features), dim=1)
        # Update attribute names
        data.attr_names.node += self.feature_names
        return data
    
class AddNormNodeAttr_asConditional(BaseTransform):
    '''
    Adds normative (i.e. same for all subjects) node attributes 
    x to each data object as conditional data.
    Parameter:
    ---------
    features (df): Pandas dataframe with feature names and data. 
    '''

    def __init__(self, features: pd.DataFrame):
        self.features = torch.tensor(features.to_numpy(), dtype=torch.float32)
        self.feature_names = features.columns.to_list()

    def forward(self, data: Data) -> Data:
        # Concatenate data in features with data.x
        data.xc = torch.cat((data.xc, self.features), dim=1)
        return data
    
class AddSubjectNodeAttr_asConditional(BaseTransform):
    '''
    Adds subject-specific node attributes as conditional data to each data object.
    Parameter:
    ---------
    feature_dict (dict): Dictionary that hashes the subject id (0-indexing!) to a numerical value.
    feature_name (str): Name of the feature being added.
    '''

    def __init__(self, feature_dict: Dict, feature_name: str = 'feat'):
        self.feature_dict = feature_dict
        self.feature_name = feature_name

    def forward(self, data: Data) -> Data:
        # Get the subject-specific value
        sub_value = self.feature_dict[data.subject.item()+1]
        # Create a tensor repeating the value for each node
        num_nodes = data.x.shape[0]
        feature_tensor = torch.full((num_nodes, 1), sub_value, dtype=torch.float32)
        # Concatenate with existing conditional features
        data.xc = torch.cat((data.xc, feature_tensor), dim=1)
        return data
    
class AddFCEdgeAttr(BaseTransform):
    '''
    Adds a fully-connected edge attribute to each data object.
    Parameter:
    ---------
    feature (dict): hashes subject id (Data.subject.item()) to an edge_attr torch.tensor.
    '''

    def __init__(self, feature: Dict, feature_name: str):
        self.feature = feature
        self.feature_name = feature_name

    def forward(self, data: Data) -> Data:
        # Get the subject id
        sub = data.subject.item()
        # Concatenate data in features with data.edge_attr
        data.edge_attr = torch.cat((data.edge_attr, self.feature[sub]), dim=1)
        # Update attribute names
        data.attr_names.edge.append(self.feature_name)

        return data
    
class AddSelfLoops(BaseTransform):
    '''
    Sets the self-loops of the adjacency matrix to a desired value.
    For setting the self-loops of multiple edge features, use a
    multidimensional input for `value` with shape [num_edge_attrs,].

    Note: the transform assumes that the current adjacency matrix
    has either no self-loops or the self-loops are all zero. If this
    is not the case, use the `RemoveSelfLoops` transform first.
    '''
    def __init__(self, fill_value: Union[float, torch.Tensor, str]):
        self.fill_value = fill_value

    def forward(self, data: Data) -> Data:
        data.edge_index, data.edge_attr = add_self_loops(data.edge_index, 
                                                         data.edge_attr,
                                                         fill_value=self.fill_value,
                                                         num_nodes=data.num_nodes)
        return data
    
class RemoveSelfLoops(BaseTransform):
    '''
    Removes self-loops from the adjacency matrix.
    '''
    def forward(self, data: Data) -> Data:
        data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
        return data

# Dataset utility functions ---------------------------------------------------------

def make_edge_index(num_nodes):
    '''
    Returns indices of upper triangular matrix excluding the diagonal. 
    The result is returned as a 2D torch tensor.
    '''
    return torch.triu(torch.ones((num_nodes, num_nodes)), diagonal=1).nonzero(as_tuple=False).t()

def get_num_nodes(atlas: str) -> int:
    '''Lookup table for number of nodes/ brain regions, given a brain atlas.'''
    # if atlas is a list, ensure it has length 1 and extract the first element
    if isinstance(atlas, list):
        assert len(atlas) == 1, "If atlas is a list, it must have length 1."
        atlas = atlas[0]

    available = ['schaefer100', 'schaefer200', 'aal']
    if atlas not in available:
        raise ValueError("Invalid name. Valid options are: " + ", ".join(available) + "\n"
                         "Edit datasets.get_num_nodes() to add the number of nodes for your atlas.")

    if atlas == 'schaefer100':
        return 100

    elif atlas == 'schaefer200':
        return 200
    
    elif atlas == 'aal':
        return 116
    
def get_default_prefilter(study: str) -> Dict[str, Any]:
    '''Returns the default prefilter for a given study.'''
    if study == 'psilodep2':
        return {'Exclusion': 0}
    elif study == 'psilodep1':
        return {'Exclusion': 0, 'missing_raw_before': 0}
    else:
        raise ValueError(f'Prefilter not specified for study {study}. \n'
                         'Edit datasets.get_default_prefilter() to add your prefilter.')
    
def get_default_target(study: str) -> str:
    '''Returns the default target for a given study.'''
    if study == 'psilodep2':
        return 'QIDS_Final_Integration'
    elif study == 'psilodep1':
        return 'QIDS_1week'
    else:
        raise ValueError(f'Target not specified for study {study}. \n'
                         'Edit datasets.get_default_target() to add your target.')

# BrainGraphDataset class -----------------------------------------------------------
    
class BrainGraphDataset(InMemoryDataset):
    '''
    Dataset of brain graphs. 

    Upon first initialization, a dataset with all available node, edge and graph features
    will be created and saved to disk. Subsequent post-processing will prune features and 
    define edge indices as specified. If new processing and overwriting the old dataset on
    disk with a newly processed one is desired, for example because new features have be-
    come available, this can be done by setting force_reload=True.

    Dependencies:
    ------------
    - edge.csv, node.csv for each subject in root/raw/study/session/atlas/subject/.
    - annotations.csv in root/raw/study/.
    '''

    def __init__(self, root: str = None, 
                 study: str = 'psilodep2', 
                 session: str = 'before', 
                 atlas: str = 'schaefer100',
                 prefilter: Dict[str, Any] = None, 
                 attrs: Attrs = None, 
                 target: str = None,
                 edge_tfm: BaseTransform = None,
                 transforms: List[BaseTransform] = [], 
                 force_reload: bool = False):
        
        # Specify node, edge, graph features of each data object
        # Default: use all available features
        if attrs is None:
            attrs = Attrs(node = get_node_feature_names(atlas),
                          edge = get_edge_feature_names(),
                          graph = [])
            attrs.add_clinical_graph_attrs(study)

        elif isinstance(attrs, dict):
            attrs = Attrs(node = attrs.get('node', []), 
                          edge = attrs.get('edge', []), 
                          graph = attrs.get('graph', []))

        # Default data directory
        if root is None:
            root = os.path.join(project_root(), 'data')

        # Set defaults depending on the study
        if prefilter is None:
            prefilter = get_default_prefilter(study)

        # Assign attributes
        self.root = root              # Data directory
        self.study = study            # Name of the study*
        self.session = session        # Name of the scanning session*
        self.atlas = atlas            # Name of the brain atlas*
        self.prefilter = prefilter    # Filter subjects based on annotations.csv
        self.attrs = attrs            # Node, edge, graph feature names
        self.edge_tfm = edge_tfm      # Transform for deriving edge indices
        self.target = target          # Graph-level prediction label
        
        # Create and compose transforms
        self.transforms = T.Compose(self._get_transforms() + transforms)

        # Initialise loading, or processing
        super(BrainGraphDataset, self).__init__(root, 
                                                transform=self.transforms, 
                                                force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0])        
        
    @property
    def processed_file_names(self):
        return [f'data_{self.__repr__()}.pt']

    def process(self):
        '''
        Loads all available node, edge, and graph feature information,
        creates a list of Data objects and saves them to disk.
        '''

        # Load annotations file and get subject ids
        annotations = load_annotations(study=self.study, filter=self.prefilter)
        subjects = get_all_ids(annotations)
        subjects = [sub-1 for sub in subjects]

        # Convert categorical columns to numerical labels
        annotations = convert_to_numerical(annotations, get_categorical_annotations(self.study))

        # Get all available node, edge, graph features
        all_attrs = Attrs()
        all_attrs.add_clinical_graph_attrs(self.study)

        datalist = []
        for i, sub in tqdm(enumerate(subjects)):

            # Node features
            features = self._get_node_features(sub)        # Load features
            numROIs = len(features)                        # Get number of brain regions
            all_attrs.node = features.columns.to_list()    # Keep track of attributes
            node_attr = torch.tensor(features.to_numpy(), dtype=torch.float32)

            # Edge indices (fully connectivity)
            rows = torch.arange(numROIs, dtype=torch.long).repeat_interleave(numROIs)
            cols = torch.arange(numROIs, dtype=torch.long).repeat(numROIs)
            edge_index = torch.stack((rows, cols), dim=0)            

            # Edge features
            features = self._get_edge_features(sub)        # Load features
            all_attrs.edge = features.columns.to_list()    # Keep track of attributes
            edge_attr = []
            for feature in all_attrs.edge:
                w = features[feature].values.reshape(numROIs, numROIs)
                edge_attr.append(w[rows, cols].reshape(-1, 1))
            edge_attr = torch.tensor(np.hstack(edge_attr), dtype=torch.float32)

            # Graph features
            graph_attr = []
            for feature in all_attrs.graph:
                graph_attr.append(annotations.loc[annotations['Patient'] == sub+1, feature].values[0])
            graph_attr = torch.tensor(graph_attr, dtype=torch.float32).unsqueeze(0)  # Shape: [1, num_graph_attrs]

            # Placeholder for conditional node features (e.g. 3d coordinates)
            xc = torch.empty((numROIs, 0), dtype=torch.float32)

            # Assemble the brain graph
            datalist.append(Data(x=node_attr, 
                                 edge_index=edge_index, 
                                 edge_attr=edge_attr, 
                                 graph_attr=graph_attr,
                                 attr_names=all_attrs, 
                                 subject=sub,
                                 xc=xc))

        # Collate processed data list and save
        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])      

    def _get_node_features(self, subject):
        feature_filepath = get_filepath(root=self.root, 
                                    study=self.study, 
                                    session=self.session, 
                                    atlas=self.atlas, 
                                    subject=subject)
        feature_file = os.path.join(feature_filepath, 'node.csv')
        if os.path.exists(feature_file):
            features = pd.read_csv(feature_file)
            return features
        else:
            raise FileNotFoundError(f'{feature_file} not found for subject {subject}.')
        
    def _get_edge_features(self, subject):
        feature_filepath = get_filepath(root=self.root, 
                                    study=self.study, 
                                    session=self.session, 
                                    atlas=self.atlas, 
                                    subject=subject)
        feature_file = os.path.join(feature_filepath, 'edge.csv')
        if os.path.exists(feature_file):
            features = pd.read_csv(feature_file)
            return features
        else:
            raise FileNotFoundError(f'{feature_file} not found for subject {subject}.')
        
    def _get_transforms(self):
        # Create list with transforms
        tfm_list = []

        # 1. Transform to re-define edge indices
        if not self.edge_tfm == None:
            tfm_list.append(self.edge_tfm)

        # 2. Transform for filtering attributes
        tfm_list.append(FilterAttributesWithContext(self.attrs))
        
        # 3. Transform for adding prediction labels
        if not self.target==None:
            annotations = load_annotations(study=self.study, filter=self.prefilter)
            labels = pd.Series(annotations[self.target].values, 
                               index=annotations['Patient']).to_dict()
            tfm_list.append(AddLabel(labels))

        return tfm_list
    
    def get_dense_adj(self, idx: int):
        '''Return the dense adjacency matrix of the graph with index idx.'''
        data = self[idx]
        adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr).squeeze(0)
        return adj
        
    def __repr__(self):
        id_parts = [self.study,
                    self.session,
                    self.atlas]
        return '_'.join(id_parts)
    