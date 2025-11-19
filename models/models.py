"""
Neural network models.

License: BSD 3-Clause
Authors: Hanna M. Tolle, Lewis J. Ng
Date: 2024-11-02
"""
import sys
sys.path.append('../')

from typing import Dict, Optional
import torch
from torch_geometric.utils import to_dense_adj, scatter, to_dense_batch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.resolver import activation_resolver
from dataclasses import dataclass
from utils.files import *
from models.utils import get_model_configs, init_model


@dataclass
class Outputs:
    x: torch.Tensor
    rcn_x: torch.Tensor
    edges: torch.Tensor
    rcn_edges: torch.Tensor
    z: torch.Tensor          
    mu: torch.Tensor
    logvar: torch.Tensor

# Helpers ----------------------------------------------------------------------

def get_device():
    '''Get torch device.'''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def L1_reg(model):
    '''L1-regularization.'''
    reg_loss = 0.
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += torch.sum(torch.abs(param))
    return reg_loss

def L2_reg(model):
    '''L2-regularization.'''
    l2_loss = 0.
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_loss += torch.sum(torch.square(param))
    return l2_loss

def elasticnet_reg(model, l1_ratio: float = 0.5):
    '''Elasticnet regularization.'''
    reg_loss = 0.
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss += l1_ratio * torch.sum(torch.abs(param))
            reg_loss += (1 - l1_ratio) * torch.sum(torch.square(param))
    return reg_loss

def make_node_ids(num_nodes: int, batch_size: int):
    '''
    Creates node indices of size ([batch_size, num_nodes, 1]).
    '''
    node_num = torch.arange(num_nodes).view(num_nodes, 1)
    node_num = node_num.repeat(batch_size, 1, 1)
    return node_num

def get_linear_layer_dims(input_dim: int, output_dim: int, num_layers: int):
    '''
    Calculates evenly spaced layer dimensions from input_dim to output_dim.
    '''
    step_size = (output_dim - input_dim) / (num_layers)
    layer_dims = [int(input_dim + i * step_size) for i in range(num_layers + 1)]
    return layer_dims

def get_geometric_layer_dims(input_dim: int, output_dim: int, num_layers: int, factor: float = 0.5):
    '''
    Calculates geometrically spaced layer dimensions from input_dim to output_dim.
    '''
    if input_dim > output_dim:
        # Decrease layer width
        layer_dims = [input_dim]
        for i in range(1, num_layers):
            layer_dims.append(max(int(layer_dims[-1] * factor), output_dim))
        layer_dims.append(output_dim)
    else:
        # Increase layer width
        layer_dims = [input_dim]
        for i in range(1, num_layers):
            layer_dims.append(min(int(layer_dims[-1] / factor), output_dim))
        layer_dims.append(output_dim)
    return layer_dims

def edges2node_features(batch: Data, add_node_id: bool = False):
    '''
    Converts edge features to node features. 
    Graphs must be fully connected! There should only be one edge attribute.

    Parameters:
    ----------
    batch (Data): Batch of data objects from PyTorch Geometric.
    add_node_id (bool): Whether to add node indices as node features.
    '''
    # Unpack batch
    x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
    batch_idx, batch_size = batch.batch, batch.batch_size
    num_nodes = x.shape[0] // batch_size
    features = []

    # Add node indices as node features [batch_size, num_nodes, 1]
    if add_node_id:
        node_ids = make_node_ids(num_nodes, batch_size)
        features.append(node_ids)

    # Reshape node features to [batch_size, num_nodes, num_node_attr]
    num_node_attr = x.shape[1]
    if num_node_attr > 0:
        x = x.view(-1, batch_size, num_node_attr) 
        features.append(x)

    # Reshape edge features to [batch_size, num_nodes, num_nodes]
    if edge_attr.shape[1] > 0:
        edges = to_dense_adj(edge_index, batch_idx, edge_attr).squeeze(-1)
        features.append(edges)  

    return torch.cat(features, dim=2), x, edges

def get_num_triu_edges(num_nodes: int):
    '''Returns the number of upper triangular edges in a fully connected graph.'''
    return num_nodes*(num_nodes-1)//2

def batch_spd(batch: Batch, max_spd_dist: int):
    """
    Creates a dense SPD tensor [B, N_max, N_max] and a mask [B, N_max].
    Only necessary if graphs in one batch have different numbers of nodes.
    Parameters:
    ----------
    batch (Batch): Batch of data objects from PyTorch Geometric.
    max_spd_dist (int): Maximum shortest path distance to consider.

    Returns:
    --------
    spd_dense (torch.Tensor): Dense SPD tensor [B, N_max, N_max].
    mask (torch.Tensor): Mask for valid nodes in each graph [B, N_max].
    """
    num_graphs = batch.num_graphs
    batch_vec = batch.batch
    num_nodes_per_graph = torch.bincount(batch_vec)
    N_max = num_nodes_per_graph.max().item()

    spd_list = []
    mask_list = []

    start = 0
    for n in num_nodes_per_graph:
        n = n.item()
        spd = getattr(batch, 'spd')[start:start+n, start:start+n] \
              if hasattr(batch, 'spd') else None
        if spd is None:
            raise ValueError("batch.spd not found.")
        
        pad_size = (0, N_max - n, 0, N_max - n)
        spd_padded = torch.nn.functional.pad(spd, pad_size, value=max_spd_dist + 1)
        spd_list.append(spd_padded)
        mask = torch.zeros(N_max, dtype=torch.bool)
        mask[:n] = True
        mask_list.append(mask)
        start += n

    spd_dense = torch.stack(spd_list, dim=0)  # [B, N_max, N_max]
    mask = torch.stack(mask_list, dim=0)      # [B, N_max]
    return spd_dense, mask

def get_batched_spd(batch, max_spd_dist):
    """Return SPD tensor [B, N, N] and mask [B, N]."""
    B = batch.num_graphs
    num_nodes = batch.num_nodes // B
    spd = batch.spd

    # Check if all graphs have same number of nodes
    if spd.shape[0] == B * num_nodes and spd.shape[1] == num_nodes:
        spd_dense = spd.view(B, num_nodes, num_nodes)
        mask = torch.ones(B, num_nodes, dtype=torch.bool, device=spd.device)
    else:
        spd_dense, mask = batch_spd(batch, max_spd_dist)
    return spd_dense, mask


# Multi-Layer Perceptron ------------------------------------------------------

class StandardMLP(torch.nn.Module):
    '''
    Multi-Layer Perceptron.
    
    Parameters:
    ----------
    layer_dims (list[int]): list of layer dimensions
    dropout (float): dropout rate
    layernorm (bool): whether to use layer normalization
    '''
    def __init__(self, layer_dims: list[int],
                 dropout: float = 0.0,
                 layernorm: bool = False):
        super().__init__()

        # Create linear layers
        num_layers = len(layer_dims) - 1
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i+1]))

            # Add activation and dropout
            if i != num_layers - 1:
                self.layers.append(torch.nn.LeakyReLU())
                if dropout > 0:
                    self.layers.append(torch.nn.Dropout(p=dropout))

        # Add layer normalization
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(layer_dims[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# Regressor models ----------------------------------------------------------------

class RegressionMLP(StandardMLP):
    '''
    Regression MLP.

    Parameters:
    ----------
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    output_dim (int): output dimension
    num_layers (int): number of layers
    dropout (float): dropout rate
    layernorm (bool): whether to use layer normalization in the final layer
    reg_strength (float): regularization strength (L2)
    mse_reduction (str): reduction method for mean squared error loss
    '''
    def __init__(self, input_dim: int, 
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 4,
                 dropout: float = 0.25,
                 layernorm: bool = False,
                 reg_strength: float = 0.01, 
                 mse_reduction: str = 'sum'):
        layer_dims = [input_dim] + [hidden_dim]*(num_layers-1) + [output_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=layernorm)
        self.reg_strength = reg_strength
        self.mse_reduction = mse_reduction

    def penalty(self):
        return self.reg_strength * elasticnet_reg(self)
    
    def loss(self, ypred, ytrue):
        loss = torch.nn.functional.mse_loss(ypred, ytrue, reduction=self.mse_reduction)
        return loss + self.penalty()

class LogisticRegressionMLP(StandardMLP):
    '''
    Logistic Regression MLP for binary classification.

    Parameters:
    ----------
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    num_layers (int): number of layers
    dropout (float): dropout rate
    layernorm (bool): whether to use layer normalization in the final layer
    reg_strength (float): regularization strength (L2)
    pos_weight (float): weight for positive class in loss calculation
    '''
    def __init__(self, input_dim: int, 
                 hidden_dim: int,
                 output_dim: int = 1,
                 num_layers: int = 4,
                 dropout: float = 0.25,
                 layernorm: bool = False,
                 reg_strength: float = 0.01,
                 pos_weight: float = None, 
                 mse_reduction: str = 'sum'):
        # Output dimension is fixed to 1 for binary classification
        layer_dims = [input_dim] + [hidden_dim]*(num_layers-1) + [output_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=layernorm)
        self.reg_strength = reg_strength
        self.pos_weight = pos_weight
        self.mse_reduction = mse_reduction

    def set_pos_weight(self, y: torch.Tensor):
        '''
        Compute the positive weight based on class distribution.
        
        Parameters:
        ----------
        y (torch.Tensor): Binary labels (0 or 1)
        
        Returns:
        -------
        float: Weight for positive class
        '''
        num_neg = (y == 0).sum().item()
        num_pos = (y == 1).sum().item()
        
        # Avoid division by zero
        if num_pos == 0:
            self.pos_weight = 1.0
        else:
            self.pos_weight = num_neg / num_pos

    def forward(self, x):
        # Apply sigmoid activation to the final layer output
        x = super().forward(x)
        return torch.sigmoid(x)

    def penalty(self):
        return self.reg_strength * L2_reg(self)
    
    def loss(self, ypred, ytrue):
        # Binary cross entropy loss with optional positive class weighting
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=ypred.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                ypred, ytrue, pos_weight=pos_weight, reduction=self.mse_reduction)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                ypred, ytrue, reduction=self.mse_reduction)
        return loss + self.penalty()

# Node embedding models --------------------------------------------------------
# Receive data batches and return node embeddings [nodes_in_batch, node_emb_dim].

class NodeEmbeddingMLP(StandardMLP):
    '''
    Node embedding MLP.
    An MLP that projects node features to a lower-dimensional embedding.
    Edges features are treated as node features. Requires a fully connected adjacency.
    This class re-implements the first part of the Lewis-style Encoder.
    Except that the final layer does not have a nonlinearity and no dropout.

    Parameters:
    ----------
    num_nodes (int): number of nodes in the graph
    num_node_attr (int): number of node features
    num_edge_attr (int): number of edge features
    node_emb_dim (int): dimension of node embeddings
    num_layers (int): number of layers
    hidden_dim (int): hidden dimension
    add_node_id (bool): whether to add node indices as node features
    dropout (float): dropout rate
    reg_strength (float): regularization strength (L2)
    '''
    def __init__(self, 
                 num_nodes: int, 
                 num_node_attr: int, 
                 num_edge_attr: int,
                 node_emb_dim: int, 
                 num_layers: int = 4,
                 hidden_dim: int = 256,
                 add_node_id: bool = True,
                 dropout: float = 0.25,
                 reg_strength: float = 0.01):
        # MLP module
        extra_dim = 1 if add_node_id else 0
        input_dim = num_nodes*num_edge_attr + num_node_attr + extra_dim
        layer_dims = [input_dim] + [hidden_dim]*(num_layers-1) + [node_emb_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=False)

        # Parameters
        self.reg_strength = reg_strength
        self.add_node_id = add_node_id
        self.input_dim = input_dim - extra_dim
        self.node_emb_dim = node_emb_dim
    
    def forward(self, batch: Data):
        '''Returns node embeddings and target decoder outputs.'''
        features = self._get_features(batch)
        node_embeddings = super().forward(features) # [batch_size, num_nodes, node_emb_dim]
        return node_embeddings.view(-1, node_embeddings.shape[2]) # [batch_size*num_nodes, node_emb_dim]
    
    def _get_features(self, batch: Data):
        '''
        Returns inputs to the forward pass and target decoder outputs.

        The NodeEmbeddingMLP requires all graphs to have the same number of nodes,
        and the adjacency matrix to be fully connected. It essentially treats the 
        edges features as node features, and it does not have a notion of a 
        node's neighbourhood. The graph is a collection of isolated nodes. Thus,
        we reshape the input features to [batch_size, num_nodes, num_node_attr], so
        that the MLP weights can be shared across nodes. No neighbourhood aggregation
        is performed. Instead, we slice up the edge features and turn them into
        node features.
        '''
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_idx, batch_size = batch.batch, batch.batch_size
        num_nodes = x.shape[0] // batch_size
        features = []

        # Add node indices as node features [batch_size, num_nodes, 1]
        if self.add_node_id:
            node_ids = make_node_ids(num_nodes, batch_size)
            features.append(node_ids)

        # Reshape node features to [batch_size, num_nodes, num_node_attr]
        num_node_attr = x.shape[1]
        if num_node_attr > 0:
            x = x.view(batch_size, num_nodes, num_node_attr) 
            features.append(x)

        # Reshape edge features to [batch_size, num_nodes, num_nodes]
        if edge_attr.shape[1] > 0:
            edges = to_dense_adj(edge_index, batch_idx, edge_attr).squeeze(-1)
            features.append(edges)
            
        return torch.cat(features, dim=2)
    
    def penalty(self):
        return self.reg_strength * L2_reg(self)
    
class NodeEmbeddingGATv2Conv(torch.nn.Module):
    '''
    Node embedding GATv2Conv.
    A Graph Attention Network that learns node embeddings that encode relevant graph structure.
    Uses GATv2Conv layers which are supposed to be more expressive than GATConv.

    Parameters:
    ----------
    num_node_attr (int): number of node features in the input
    num_edge_attr (int): number of edge features in the input
    hidden_dim (int): dimension of hidden layers
    node_emb_dim (int): dimension of node embeddings
    num_layers (int): number of attention layers
    num_heads (int): number of attention heads per layer
    dropout (float): dropout rate
    reg_strength (float): regularization strength (L2)
    layernorm (bool): whether to use layer normalization
    '''
    def __init__(self, num_node_attr: int,
                 num_cond_attrs: int,
                 num_edge_attr: int,
                 hidden_dim: int,
                 node_emb_dim: int,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.25,
                 reg_strength: float = 0.01,
                 layernorm: bool = True):
        super().__init__()

        # Validate hidden_dim is compatible with num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        # Create GAT layers
        self.convs = torch.nn.ModuleList()
        
        # First layer (input -> hidden)
        self.convs.append(GATv2Conv(
            in_channels=num_node_attr + num_cond_attrs,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout, 
            edge_dim=num_edge_attr))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=num_edge_attr))

        # Final layer (hidden -> output, single head)
        self.convs.append(GATv2Conv(
            in_channels=hidden_dim,
            out_channels=node_emb_dim,
            heads=1,  
            dropout=0.0,
            edge_dim=num_edge_attr))

        # Activation, dropout and layer norm layers
        self.activation = torch.nn.LeakyReLU()
        self.layernorm = torch.nn.LayerNorm(hidden_dim) if layernorm else None

        # Parameters
        self.reg_strength = reg_strength
        self.node_emb_dim = node_emb_dim
        self.input_dim = num_node_attr
        
    def forward(self, batch: Data):
        x, edge_index = batch.x, batch.edge_index
        x = torch.cat([x, batch.xc], dim=1) # add conditional attributes
        edge_attr = getattr(batch, 'edge_attr', None)

        if self.layernorm is not None:
            # Layer norm after activation
            for conv in self.convs[:-1]:
                x = conv(x, edge_index, edge_attr)
                x = self.layernorm(self.activation(x))
        else:
            # No layer norm
            for conv in self.convs[:-1]:
                x = conv(x, edge_index, edge_attr)
                x = self.activation(x)
                
        return self.convs[-1](x, edge_index, edge_attr)  # [num_nodes_in_batch, node_emb_dim]
    
    def penalty(self):
        return self.reg_strength * L2_reg(self)
    
class NodeEmbeddingGATv2Conv_withSkip(NodeEmbeddingGATv2Conv):
    '''
    Node embedding GATv2Conv with skip connections.
    '''
    def forward(self, batch: Data):
        x, edge_index = batch.x, batch.edge_index
        x = torch.cat([x, batch.xc], dim=1) # add conditional attributes
        edge_attr = getattr(batch, 'edge_attr', None)
        
        # Store initial features
        x_prev = x
        if self.layernorm is not None:
            # Layer norm after activation
            for conv in self.convs[:-1]:
                x_out = conv(x, edge_index, edge_attr)
                x_out = self.layernorm(self.activation(x_out))
                # Add skip connection if dimensions match
                if x_out.shape[-1] == x_prev.shape[-1]:
                    x = x_out + x_prev
                else:
                    x = x_out
                x_prev = x
        else:
            # No layer norm
            for conv in self.convs[:-1]:
                x_out = conv(x, edge_index, edge_attr)
                x_out = self.activation(x_out)
                # Add skip connection if dimensions match
                if x_out.shape[-1] == x_prev.shape[-1]:
                    x = x_out + x_prev
                else:
                    x = x_out
                x_prev = x
                
        return self.convs[-1](x, edge_index, edge_attr)  # [num_nodes_in_batch, node_emb_dim]

class NodeEmbeddingGraphormer(torch.nn.Module):
    """
    Graphormer-style node-embedding model.
    Incorporates shortest-path distance (SPD) and edge features as additive
    attention biases. Does not consider centrality/ node degree for now.

    Inputs per batch:
        batch.x             [num_nodes, node_feat_dim]
        batch.edge_attr     [num_edges, edge_feat_dim]
        batch.edge_index    [2, num_edges]
        batch.spd           [num_nodes, num_nodes]  (precomputed SPD matrix)
        batch.batch         [num_nodes]  (graph IDs)
    """

    def __init__(self,
                 num_node_attr: int,
                 num_cond_attrs: int,
                 num_edge_attr: int,
                 hidden_dim: int,   # usually higher than input_dim
                 node_emb_dim: int,
                 max_spd_dist: int = 10,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 reg_strength: float = 0.01,
                 layernorm: bool = True):
        super().__init__()

        # Validate hidden_dim is compatible with num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        # Parameters
        self.hidden_dim = hidden_dim
        self.node_emb_dim = node_emb_dim
        self.reg_strength = reg_strength
        self.input_dim = num_node_attr + num_cond_attrs
        self.max_spd_dist = max_spd_dist
        
        # Embeddings
        self.input_proj = torch.nn.Linear(self.input_dim, hidden_dim)
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_edge_attr, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_heads)
        )
        self.dist_encoder = torch.nn.Embedding(max_spd_dist + 2, num_heads)  # +1 for padding/inf

        # Transformer layers
        self.layers = torch.nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, dropout, layernorm)
            for _ in range(num_layers)
        ])

        self.output_proj = torch.nn.Linear(hidden_dim, node_emb_dim)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = getattr(batch, 'edge_attr', None)
        x = torch.cat([x, batch.xc], dim=1) # add conditional attributes
        x = self.input_proj(x)

        # Dense batching
        x_dense, x_mask = to_dense_batch(x, batch.batch)  # [B, N, D]
        B, N, D = x_dense.shape

        # SPD embeddings
        spd = getattr(batch, 'spd', None)
        assert spd is not None, "batch.spd must be provided for Graphormer"
        spd_dense, spd_mask = get_batched_spd(batch, self.max_spd_dist) # [B, N_max, N_max]
        spd_dense = spd_dense.clamp(max=self.dist_encoder.num_embeddings - 1)
        spd_bias = self.dist_encoder(spd_dense.long())  # [B, N_max, N_max, num_heads]
        spd_bias = spd_bias.permute(0, 3, 1, 2)         # [B, num_heads, N_max, N_max]

        # Make sure masks are equal
        if not torch.equal(x_mask, spd_mask):
            mask = x_mask & spd_mask
        else:
            mask = x_mask

        # Edge feature bias
        edge_bias = torch.zeros_like(spd_bias)
        if edge_attr is not None:
            edge_bias_values = self.edge_encoder(edge_attr) # [E, num_heads]

            # Convert global edge_index to per-graph local indices
            # Compute per-graph node ranges so we can map to [0..N_g-1]
            node_counts = torch.bincount(batch.batch)
            starts = torch.cumsum(torch.cat([torch.tensor([0], device=node_counts.device), 
                                             node_counts[:-1]]), dim=0) # [B]

            for g in range(B):
                g_mask = (batch.batch[edge_index[0]] == g)
                e_idx = edge_index[:, g_mask]               # global indices
                start = starts[g]
                e_idx_local = e_idx - start                 # map to [0..N_g-1]
                e_attr = edge_bias_values[g_mask]           # [E_g, num_heads]
                # scatter into the [B,num_heads,N,N] bias tensor
                edge_bias[g, :, e_idx_local[0], e_idx_local[1]] = e_attr.T  # [num_heads, E_g]

        # Combine attention biases
        attn_bias = spd_bias + edge_bias # [B, num_heads, N_max, N_max]

        # Transformer layers
        for layer in self.layers:
            x_dense = layer(x_dense, mask, attn_bias)

        # Output projection
        x_out = x_dense[mask]
        x_out = self.output_proj(x_out)
        return x_out

    def penalty(self):
        return self.reg_strength * L2_reg(self)

class GraphormerLayer(torch.nn.Module):
    """Single Graphormer layer with edge + distance bias."""
    def __init__(self, embed_dim: int, 
                 num_heads: int, 
                 dropout: float = 0.1, 
                 layernorm: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = EdgeAwareMultiheadAttention(embed_dim, num_heads, dropout)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim)
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim) if layernorm else torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(embed_dim) if layernorm else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask, attn_bias):
        attn_out, _ = self.attn(x, mask=mask, edge_bias=attn_bias)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class EdgeAwareMultiheadAttention(torch.nn.Module):
    """Multihead attention with additive per-head bias."""
    def __init__(self, embed_dim: int, 
                 num_heads: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim) 
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim) 
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim) 
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None, edge_bias=None):
        B, N, D = x.shape
        H, d_h = self.num_heads, self.head_dim

        # Compute queries, keys, and values for x
        Q = self.q_proj(x).view(B, N, H, d_h).transpose(1, 2)  # [B,H,N,d_h]
        K = self.k_proj(x).view(B, N, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, d_h).transpose(1, 2)

        # Compute raw attention scores for each node pair
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale # [B,H,N,N]

        # Add edge bias to attention scores, if provided
        if edge_bias is not None:
            attn_scores = attn_scores + edge_bias  # [B,H,N,N]

        # Mask out attention scores for padded nodes
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask[:, None, None, :], float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1) # [B,H,N,N]
        attn_weights = self.dropout(attn_weights)

        # Compute output
        out = torch.matmul(attn_weights, V)  # [B,H,N,d_h]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out) # mix information across heads
        return out, attn_weights


# Encoders ---------------------------------------------------------------------
'''
Encoders take node embeddings [batch_size, num_nodes, node_emb_dim] and return 
latent means and log variance vectors [batch_size, latent_dim*2].
'''

class DenseEncoder(StandardMLP):
    '''
    Dense L2-regularized encoder of a Variational Autoencoder.
    This class re-implements the encoder part of the Lewis-style VAE.

    Parameters:
    ----------
    input_dim (int): input dimension
    latent_dim (int): latent dimension
    num_layers (int): number of layers
    dropout (float): dropout rate
    reg_strength (float): regularization strength (L2)
    '''
    def __init__(self, input_dim: int, 
                 latent_dim: int, 
                 num_layers: int = 2, 
                 dropout: float = 0.25,
                 reg_strength: float = 0.01):
        layer_dims = [input_dim] + [4*latent_dim]*(num_layers-1) + [2*latent_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=False)
        self.reg_strength = reg_strength
        self.latent_dim = latent_dim

    def forward(self, embeddings):
        return torch.chunk(super().forward(embeddings), 2, dim=1) # mu, logvar
    
    def penalty(self):
        return self.reg_strength * L2_reg(self)
    
class DenseOneLayerEncoder(torch.nn.Module):
    '''
    Dense encoder of a Variational Autoencoder.
    This encoder has only one layer, no dropout, and should be L2-regularized.

    Parameters:
    ----------
    input_dim (int): input dimension
    latent_dim (int): latent dimension
    reg_strength (float): regularization strength (L2)
    '''
    def __init__(self, input_dim: int, 
                 latent_dim: int,
                 reg_strength: float = 0.01):
        super().__init__()
        self.layer = torch.nn.Linear(input_dim, latent_dim*2)
        self.reg_strength = reg_strength
        self.latent_dim = latent_dim

    def forward(self, embeddings):
        return torch.chunk(self.layer(embeddings), 2, dim=1) # mu, logvar
    
    def penalty(self):
        return self.reg_strength * L2_reg(self)
    
# Pooling / Readout -----------------------------------------------------------------
# Receives input [nodes_in_batch, node_emb_dim] and returns [batch_size, -1].

class ConcatPooling(torch.nn.Module):
    '''
    Concatenation "pooling".
    This is not permutation invariant and requires graphs to be
    homogeneous (i.e. all graphs must have the same number of nodes,
    and some notion of node ordering).
    '''
    HANDLES_CONTEXT = False
    @classmethod
    def can_handle_context(cls):
        return cls.HANDLES_CONTEXT
    
    def __init__(self, pooling_dim: int, 
                 num_nodes: int):
        super().__init__()  
        self.output_dim = pooling_dim * num_nodes
    
    def forward(self, node_embeddings, batch_index):
        batch_size = batch_index.max().item() + 1
        return node_embeddings.view(batch_size, -1)
    
    def penalty(self):
        return 0.
    
class GlobalAttentionPooling(torch.nn.Module):
    '''
    Global attention pooling encoder.
    A single-layer MLP + softmax that learns attention weights for each node,
    which are then used to compute the pooled graph representation as a weighted
    average of the node embeddings.

    Parameters:
    ----------
    output_dim (int): dimension of node or graph embeddings
    reg_strength (float): regularization strength (L2)
    reduce (str): reduction method for scatter (mean, sum, etc.)
    '''
    HANDLES_CONTEXT = False
    @classmethod
    def can_handle_context(cls):
        return cls.HANDLES_CONTEXT
    
    def __init__(self, pooling_dim: int, 
                 reg_strength: float = 0.01,
                 reduce: str = 'mean'):
        super().__init__()
        # Attention layer
        self.attention = torch.nn.Linear(pooling_dim, 1)
        torch.nn.init.xavier_uniform_(self.attention.weight, gain=1.0)
        torch.nn.init.zeros_(self.attention.bias)

        # Parameters
        self.reg_strength = reg_strength
        self.reduce = reduce
        self.output_dim = pooling_dim

    def forward(self, z, batch_index):
        '''
        Parameters:
        ----------
        node_embeddings (torch.Tensor): node embeddings [nodes_in_batch, node_emb_dim]
        batch_index (torch.Tensor): assigns each node to a graph [nodes_in_batch]
        '''
        # Compute attention weights
        scores = self.attention(z)  # [nodes_in_batch, 1]
        scores = scores - scores.max(dim=0, keepdim=True)[0]  # log-sum-exp trick for stability
        weights = torch.nn.functional.softmax(scores, dim=0)

        # Weighted average over nodes in each graph using batch_index
        pooled = scatter(weights * z, batch_index, dim=0, reduce=self.reduce)
        return pooled
    
    def get_attention_weights(self, z, batch_index=None):
        # Compute attention weights
        scores = self.attention(z)  # [num_nodes, 1]
        scores = scores - scores.max(dim=0, keepdim=True)[0]  # log-sum-exp trick for stability
        weights = torch.nn.functional.softmax(scores, dim=0)  # [num_nodes, 1]
        return weights
    
    def penalty(self):
        return self.reg_strength * L2_reg(self)

class AttentionNetPooling(torch.nn.Module):
    '''
    Attention network pooling.
    Like GlobalAttentionPooling, but attention weights are computed
    by a 1-layer MLP. Can handle latent inputs, in which case attention 
    weights are computed from concatenated latent vectors and shared features. 
    The weighted average (pooling result) is computed over the latent vectors.
    '''
    HANDLES_CONTEXT = True
    @classmethod
    def can_handle_context(cls):
        return cls.HANDLES_CONTEXT
    
    def __init__(self, pooling_dim: int, 
                 num_context_attrs: int,
                 hidden_dim: int,
                 reg_strength: float = 0.01,
                 reduce: str = 'mean'):
        super().__init__()
        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(pooling_dim + num_context_attrs, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1))
        self.reg_strength = reg_strength
        self.reduce = reduce
        self.output_dim = pooling_dim

    def forward(self, z, batch_index):      
        # Compute attention weights
        scores = self.attention_net(z)                    
        scores = scores - scores.max(dim=0, keepdim=True)[0]  # log-sum-exp trick for stability
        weights = torch.nn.functional.softmax(scores, dim=0)  # [num_nodes, 1]
        
        # Compute weighted average of z (without context attributes)
        pooled = scatter(weights * z[:, :self.output_dim], batch_index, dim=0, reduce=self.reduce)
        return pooled
    
    def get_attention_weights(self, z, batch_index=None):
        # Compute attention weights
        scores = self.attention_net(z)  # [num_nodes, 1]
        scores = scores - scores.max(dim=0, keepdim=True)[0]  # log-sum-exp trick for stability
        weights = torch.nn.functional.softmax(scores, dim=0)  # [num_nodes, 1]
        return weights
    
    def penalty(self):
        return self.reg_strength * L2_reg(self)

class GraphTransformerPooling(torch.nn.Module):
    """
    Graph transformer-based pooling layer.
    Applies self-attention among all nodes in each graph (multi-head),
    producing contextualised node embeddings that are then mean- or
    attention-pooled into a graph-level vector.
    """
    HANDLES_CONTEXT = False
    @classmethod
    def can_handle_context(cls):
        return cls.HANDLES_CONTEXT
    
    def __init__(self, pooling_dim: int, 
                 num_heads: int = 4,
                 ff_hidden_dim: int = None, 
                 dropout: float = 0.1,
                 reg_strength: float = 0.01,
                 reduce: str = 'mean'):
        super().__init__()
        self.pooling_dim = pooling_dim
        self.num_heads = num_heads
        self.reduce = reduce
        self.output_dim = pooling_dim
        self.dropout = torch.nn.Dropout(dropout)

        # Multi-head self-attention
        self.attn = torch.nn.MultiheadAttention(pooling_dim, 
            num_heads, dropout=dropout, batch_first=True)

        # Feedforward network (position-wise)
        ff_hidden_dim = ff_hidden_dim or 2 * pooling_dim
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(pooling_dim, ff_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_hidden_dim, pooling_dim),
        )
        self.reg_strength = reg_strength
        self.norm1 = torch.nn.LayerNorm(pooling_dim)
        self.norm2 = torch.nn.LayerNorm(pooling_dim)

    def forward(self, z, batch_index):
        # Convert node features to dense batch form: [B, N_max, D]
        z_dense, mask = to_dense_batch(z, batch_index)
        # z_dense: [batch_size, num_nodes, embed_dim]
        # mask: [batch_size, num_nodes]

        # Self-attention: contextualise node embeddings within each graph
        attn_out, _ = self.attn(z_dense, z_dense, z_dense,
                                key_padding_mask=~mask)  # mask=False -> attend only to valid nodes
        z_dense = self.norm1(z_dense + self.dropout(attn_out))

        # Feed-forward transformation
        ff_out = self.ff(z_dense)
        z_dense = self.norm2(z_dense + self.dropout(ff_out))

        # Pooling: mean or attention over valid nodes
        if self.reduce == 'mean':
            pooled = (z_dense * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        elif self.reduce == 'sum':
            pooled = (z_dense * mask.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError("reduce must be 'mean' or 'sum'")

        return pooled

    def get_attention_weights(self, z, batch_index):
        """
        Returns the self-attention weights (averaged over heads)
        for each graph in the batch.

        Parameters
        ----------
        z : torch.Tensor
            Node embeddings [num_nodes, pooling_dim]
        batch_index : torch.Tensor
            Batch vector mapping each node to its graph [num_nodes]

        Returns
        -------
        attn_avg : torch.Tensor
            Attention weight matrices averaged over heads [batch_size, num_nodes, num_nodes],
            where N = max number of nodes in the batch.
        mask : torch.Tensor
            Boolean mask indicating valid nodes [batch_size, num_nodes]
        """

        # Multi-head self-attention
        z_dense, mask = to_dense_batch(z, batch_index) 
        _, attn_weights = self.attn(
            z_dense, z_dense, z_dense,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=False  # keep per-head attention
        ) # [batch_size, num_heads, num_nodes, num_nodes]

        # Average over heads
        attn_avg = attn_weights.mean(dim=1)  # [batch_size, num_nodes, num_nodes]

        # Mask out invalid nodes (padding)
        attn_avg = attn_avg * mask[:, None, :].float() * mask[:, :, None].float()

        # Compute inbound attention weights
        inbound = attn_avg.sum(dim=-2)
        inbound = inbound * mask.float() / mask.sum(dim=-2, keepdim=True).float()
        # outbound = attn_avg.sum(dim=-1)
        # outbound = outbound * mask.float() / mask.sum(dim=-1, keepdim=True).float()

        return inbound
    
    def penalty(self):
        return self.reg_strength * L2_reg(self)

# Decoders ---------------------------------------------------------------------
'''
Decoders take latent variables and return original node features or edge features.
'''

class Decoder(StandardMLP):
    '''
    Decoder of a Variational Autoencoder.

    Parameters:
    ----------
    latent_dim (int): latent dimension
    output_dim (int): output dimension
    num_layers (int): number of layers
    dropout (float): dropout rate
    reg_strength (float): regularization strength (L2)
    act (str): activation function (identity, tanh, sigmoid, leaky_relu)
    '''
    def __init__(self, latent_dim: int, 
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.25,
                 reg_strength: float = 0.01,
                 act: str = 'identity'):
        layer_dims = [latent_dim] + [4*latent_dim]*(num_layers-1) + [output_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=False)
        self.reg_strength = reg_strength
        if act == 'identity':
            self.act = lambda x: x
        else:
            self.act = activation_resolver(act)

    def forward(self, z):
        return self.act(super().forward(z))

    def penalty(self):
        return self.reg_strength * L2_reg(self)

# Permutation invariant decoders -----------------------------------------------

class MLPNodeDecoder(StandardMLP):
    '''
    Decoder that reconstructs node features as a multi-layer perceptron.

    Forward pass (weights shared across nodes):
    - takes z [nodes_in_batch, latent_dim]
    - returns rcn_x [nodes_in_batch, output_dim]
    '''
    def __init__(self, 
                 latent_dim: int, 
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.25,
                 reg_strength: float = 0.01,
                 act: str = 'identity'):
        layer_dims = [latent_dim] + [hidden_dim]*(num_layers-1) + [output_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=False)
        self.reg_strength = reg_strength
        if act == 'identity':
            self.act = lambda x: x
        else:
            self.act = activation_resolver(act)

    def forward(self, z):
        # z shape: [nodes_in_batch, latent_dim + extra_dim]
        return self.act(super().forward(z))

    def penalty(self):
        return self.reg_strength * L2_reg(self)

class InnerProductEdgeDecoder(torch.nn.Module):
    '''
    Decoder that reconstructs edge features as the inner product 
    of latent node embeddings + non-linearity. 
    Adapted from (Kipf & Welling, 2016).
    '''
    def __init__(self, act: str = 'tanh'):
        super().__init__()
        self.act = activation_resolver(act)
    
    def forward(self, z, edge_idx):
        # z shape: [nodes_in_batch, latent_dim + extra_dim]
        # edge_idx shape: [2, edges_in_batch]
        # Returns shape: [edges_in_batch, 1]
        return self.act((z[edge_idx[0]] * z[edge_idx[1]]).sum(dim=1)).unsqueeze(1)
    
    def penalty(self):
        return 0.

class MLPEdgeDecoder(StandardMLP):
    def __init__(self, latent_dim: int,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 reg_strength: float = 0.01,
                 dropout: float = 0.25,
                 act: str = 'tanh'):
        layer_dims = [latent_dim*2] + [hidden_dim]*(num_layers-1) + [1]
        super().__init__(layer_dims, dropout=dropout, layernorm=False)
        self.reg_strength = reg_strength
        self.act = activation_resolver(act)

    def forward(self, z, edge_idx):
        zi = z[edge_idx[0]]
        zj = z[edge_idx[1]]
        node_pairs = torch.cat([zi, zj], dim=1)
        return self.act(super().forward(node_pairs))

    def penalty(self):
        return self.reg_strength * L2_reg(self)

# Variational graph autoencoders -----------------------------------------------
'''
Variational graph autoencoders take torch_geometric batches and return
reconstructed node features and edge features. They are composed of:
- Node embedding model
- Encoder
- Decoders for node features and edge features
- Possibly a regression-MLP, for learning task-specific latent variables
'''    

class GraphLevelVGAE(torch.nn.Module):
    '''
    Graph-level VGAE.
    VGAE that learns node embeddings, which are concatenated to form
    a graph embedding. The graph embedding is then encoded by a VAE.
    Samples z from the GraphLevelVGAE represent graphs and are vectors.
    GraphLevelVGAE requires homogeneous graphs, i.e. all graphs must have the same
    number of nodes and node ordering.

    Parameters:
    ----------
    params (dict): shared parameters
    node_emb_model_cfg (dict): node embedding model model_type and params
    pooling_cfg (dict): pooling model model_type and params
    encoder_cfg (dict): encoder model model_type and params
    edge_decoder_cfg (dict): edge decoder model model_type and params
    node_decoder_cfg (dict): node decoder model model_type and params
    '''
    def __init__(self, params: dict,
                 node_emb_model_cfg: dict,
                 pooling_cfg: dict,
                 encoder_cfg: dict, 
                 edge_decoder_cfg: Optional[dict] = None,
                 node_decoder_cfg: Optional[dict] = None):
        super().__init__()
        # Get shared parameters
        self.params = get_model_configs(self.__class__.__name__, **params)

        # Make sure that submodule configs match params
        self._build_modules(node_emb_model_cfg, 
                            pooling_cfg, 
                            encoder_cfg, 
                            edge_decoder_cfg, 
                            node_decoder_cfg)
        
    def _build_modules(self, node_emb_model_cfg, 
                       pooling_cfg, 
                       encoder_cfg, 
                       edge_decoder_cfg, 
                       node_decoder_cfg):
        # Node embedding model
        updated_params = self._get_module_params(node_emb_model_cfg)
        self.node_emb_model = init_model(node_emb_model_cfg['model_type'], updated_params)

        # Pooling layer
        updated_params = self._get_module_params(pooling_cfg)
        updated_params['pooling_dim'] = self.params['node_emb_dim'] 
        self.pooling = init_model(pooling_cfg['model_type'], updated_params)
        
        # Encoder
        updated_params = self._get_module_params(encoder_cfg)
        updated_params['input_dim'] = self.pooling.output_dim
        self.encoder = init_model(encoder_cfg['model_type'], updated_params)

        # Decoders
        self.edge_decoder, self.node_decoder = None, None
        if edge_decoder_cfg is not None and 'model_type' in edge_decoder_cfg:
            if edge_decoder_cfg['model_type'] == 'CorrMatDecoder':
                assert self.params['num_edge_attr'] == 1, "CorrMatDecoder can only decode 1 edge attribute"
            updated_params = self._get_module_params(edge_decoder_cfg)
            num_triu_edges = self.params['num_nodes']*(self.params['num_nodes']-1)//2
            updated_params['output_dim'] = num_triu_edges*self.params['num_edge_attr']
            self.edge_decoder = init_model(edge_decoder_cfg['model_type'], updated_params)

        if node_decoder_cfg is not None and 'model_type' in node_decoder_cfg:
            updated_params = self._get_module_params(node_decoder_cfg)
            updated_params['output_dim'] = self.params['num_nodes']*self.params['num_node_attr']
            self.node_decoder = init_model(node_decoder_cfg['model_type'], updated_params)

        self.modules = [self.node_emb_model, 
                        self.pooling,
                        self.encoder, 
                        self.edge_decoder, 
                        self.node_decoder]
        self.readout_dim = self.params['latent_dim']
            
    def _get_module_params(self, submodule_cfg):
        '''
        Get all required and optional parameters for a submodule.
        '''
        # Get all required and optional parameters
        all_params = get_model_configs(submodule_cfg['model_type'], **self.params)
        # These parameters must be consistent across all submodules
        must_be_consistent = ['num_nodes', 
                              'num_edge_attr', 
                              'num_node_attr',
                              'num_graph_attr',
                              'latent_dim',
                              'node_emb_dim',
                              'num_cond_attrs']
        # Override default optional parameters with user inputs
        for k, v in submodule_cfg['params'].items():
            if k in all_params:
                if k not in must_be_consistent and v is not None:
                    all_params[k] = v
            else:
                raise ValueError(f"Unknown parameter '{k}' for model type '{submodule_cfg['model_type']}'")
        return all_params
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, batch: Data):
        '''
        Parameters:
        ----------
        batch (Data): Batch of data objects from PyTorch Geometric.
        '''
        # Save a copy of the original inputs for the decoder
        x, edges = self._get_decoder_labels(batch)
       
        # Encode graph features
        node_embeddings = self.node_emb_model(batch)                   # [nodes_in_batch, node_emb_dim]
        graph_embedding = self.pooling(node_embeddings, batch.batch)   # [batch_size, -1]
        mu, logvar = self.encoder(graph_embedding)                     # 2 x [batch_size, latent_dim]
        z = self.reparameterize(mu, logvar)                            # [batch_size, latent_dim]

        # Decode graph features
        rcn_x, rcn_edges = None, None
        if self.node_decoder is not None:
            rcn_x = self.node_decoder(z)             # [batch_size, num_nodes*num_node_attr] 
            x = x.view(rcn_x.shape)                  # x: [batch_size*num_nodes, num_node_attr] -> [batch_size, num_nodes*num_node_attr]
        if self.edge_decoder is not None:
            rcn_edges = self.edge_decoder(z)         # [batch_size, num_triu_edges]

        # Wrap outputs
        out = Outputs(x=x, 
                      rcn_x=rcn_x, 
                      edges=edges, 
                      rcn_edges=rcn_edges, 
                      z=z, mu=mu, logvar=logvar)
        return out
    
    def _get_decoder_labels(self, batch: Data):
        # Get x (node features) in the right shape
        x = batch.x.clone()   # [nodes_in_batch, num_node_attr]

        # Get triu edges (edge features) in the right shape
        num_nodes = batch.num_nodes // batch.batch_size
        triu_idx = torch.triu(torch.ones((num_nodes, num_nodes)), diagonal=1).nonzero(as_tuple=False).t()

        edges = batch.edge_attr.clone()                                        # [edges_in_batch, num_edge_attr]
        edges = to_dense_adj(batch.edge_index, batch.batch, edges).squeeze(-1) # [batch_size, num_nodes, num_nodes]
        triu_edges = edges[:, triu_idx[0], triu_idx[1]]                        # [batch_size, num_triu_edges]

        return x, triu_edges
    
    def readout(self, z, context = None, batch_idx = None):
        return z
    
    def decode(self, z, triu_idx=None):
        '''
        Decodes the latent sample (num_nodes x latent_dim) 
        into node features and edge features.
        '''
        rcn_x, rcn_edges = None, None
        if self.node_decoder is not None:
            rcn_x = self.node_decoder(z)
        if self.edge_decoder is not None:
            rcn_edges = self.edge_decoder(z)
        return rcn_x, rcn_edges
    
    def penalty(self):
        reg_loss = sum(m.penalty() for m in self.modules if m is not None)     
        return reg_loss
    
    def loss(self, out):
        # Reconstruction loss
        rcn_loss = 0.
        if out.rcn_x is not None:
            rcn_loss += torch.nn.functional.mse_loss(out.rcn_x, out.x, reduction='sum')
        if out.rcn_edges is not None:
            rcn_loss += torch.nn.functional.mse_loss(out.rcn_edges, out.edges, reduction='sum')

        # KL divergence
        # KL divergence between VAE's learned distribution N(mu, var) and prior N(0,1)
        # Formula: KL(N(mu,var) || N(0,1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        # out.logvar is log(var), so var = exp(logvar)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + out.logvar - out.mu.pow(2) - out.logvar.exp(), dim=1))

        # Regularization
        reg_loss = self.penalty()

        return rcn_loss + kl_loss + reg_loss
    
class NodeLevelVGAE(torch.nn.Module):
    '''
    Node-level VGAE.
    VGAE that learns node embeddings, which are then encoded by a VAE.
    Samples z from the NodeLevelVGAE represent nodes and are matrices,
    where each row corresponds to the latent vector of a node. Finally,
    a readout (pooling) layer aggregates latent node vectores into a 
    graph embedding for the graph-level regression task.

    Parameters:
    ----------
    node_emb_model (torch.nn.Module): learns node embeddings
    encoder (torch.nn.Module): encodes the node embeddings
    pooling (torch.nn.Module): aggregates node embeddings into a graph embedding
    edge_decoder (torch.nn.Module): reconstructs edge features
    node_decoder (torch.nn.Module): reconstructs node features
    '''
    def __init__(self, params: dict,
                 node_emb_model_cfg: dict,
                 encoder_cfg: dict, 
                 pooling_cfg: dict,
                 node_decoder_cfg: Optional[dict] = None,
                 edge_decoder_cfg: Optional[dict] = None,
                 edge_idx_decoder_cfg: Optional[dict] = None):
        super().__init__()  
        # Get shared parameters
        self.params = get_model_configs(self.__class__.__name__, **params)
        self.decode_edge_idx = False
        if edge_idx_decoder_cfg:
            self.decode_edge_idx = True

        # Make sure that submodule configs match params
        self._build_modules(node_emb_model_cfg, 
                            pooling_cfg, 
                            encoder_cfg, 
                            edge_decoder_cfg,
                            edge_idx_decoder_cfg,
                            node_decoder_cfg)
        
    def _build_modules(self, node_emb_model_cfg, 
                       pooling_cfg, 
                       encoder_cfg, 
                       edge_decoder_cfg,
                       edge_idx_decoder_cfg,
                       node_decoder_cfg):
        # Node embedding model
        updated_params = self._get_module_params(node_emb_model_cfg)
        self.node_emb_model = init_model(node_emb_model_cfg['model_type'], updated_params)

        # Encoder
        updated_params = self._get_module_params(encoder_cfg)
        updated_params['input_dim'] = self.params['node_emb_dim']
        self.encoder = init_model(encoder_cfg['model_type'], updated_params)

        # Pooling/readout layer
        updated_params = self._get_module_params(pooling_cfg)
        updated_params['pooling_dim'] = self.params['latent_dim']
        
        # Add num_context_attrs if pooling layer handles it
        pooling_class = globals()[pooling_cfg['model_type']]
        if pooling_class.can_handle_context():
            updated_params['num_context_attrs'] = self.params['num_context_attrs']
        elif self.params['num_context_attrs'] > 0:
            raise ValueError(f"Pooling layer '{pooling_cfg['model_type']}' cannot handle context attributes. " 
                             f"Set 'num_context_attrs' to 0 or change pooling layer.")
        self.pooling = init_model(pooling_cfg['model_type'], updated_params)

        # Decoders
        self.edge_decoder, self.edge_idx_decoder, self.node_decoder = None, None, None
        if edge_decoder_cfg:
            if not edge_decoder_cfg['model_type'].endswith('EdgeDecoder'):
                raise ValueError(f"Edge decoder must be an 'EdgeDecoder', got '{edge_decoder_cfg['model_type']}'")
            updated_params = self._get_module_params(edge_decoder_cfg)
            self.edge_decoder = init_model(edge_decoder_cfg['model_type'], updated_params)

        if edge_idx_decoder_cfg:
            assert edge_decoder_cfg, "Edge decoder must be specified if edge index decoder is specified"
            updated_params = self._get_module_params(edge_idx_decoder_cfg)
            self.edge_idx_decoder = init_model(edge_idx_decoder_cfg['model_type'], updated_params)

        if node_decoder_cfg:
            updated_params = self._get_module_params(node_decoder_cfg)
            updated_params['output_dim'] = self.params['num_node_attr']
            self.node_decoder = init_model(node_decoder_cfg['model_type'], updated_params)

        self.modules = [self.node_emb_model, 
                        self.encoder, 
                        self.pooling, 
                        self.edge_decoder, 
                        self.edge_idx_decoder,
                        self.node_decoder]
        self.readout_dim = self.pooling.output_dim
            
    def _get_module_params(self, submodule_cfg):
        '''
        Get all required and optional parameters for a submodule.
        '''
        # Get all required and optional parameters
        all_params = get_model_configs(submodule_cfg['model_type'], **self.params)
        # These parameters must be consistent across all submodules
        must_be_consistent = ['num_nodes', 
                              'num_edge_attr', 
                              'num_node_attr',
                              'num_graph_attr',
                              'latent_dim',
                              'node_emb_dim',
                              'num_cond_attrs']
        # Override default optional parameters with user inputs
        for k, v in submodule_cfg['params'].items():
            if k in all_params:
                if k not in must_be_consistent and v is not None:
                    all_params[k] = v
            else:
                raise ValueError(f"Unknown parameter '{k}' for model type '{submodule_cfg['model_type']}'")
        return all_params

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, batch: Data):
        '''
        Parameters:
        ----------
        batch (Data): Batch of data objects from PyTorch Geometric.
        '''
        # Save a copy of the original inputs for the decoder
        x, edges, triu_idx = self._get_decoder_labels(batch)
       
        # Encode node features
        node_embeddings = self.node_emb_model(batch)       # [nodes_in_batch, node_emb_dim]
        mu, logvar = self.encoder(node_embeddings)         # 2 x [nodes_in_batch, latent_dim]
        z = self.reparameterize(mu, logvar)                # [nodes_in_batch, latent_dim]

        # Decode node features
        rcn_x = None
        if self.node_decoder is not None:
            rcn_x = self.node_decoder(z)                   # [nodes_in_batch, num_node_attr]

        # Decode edge features
        rcn_edges = None
        if self.edge_decoder is not None:
            rcn_edges = self.edge_decoder(z, triu_idx)     # [num_triu_edges, num_edge_attr]

            # Decode edge indices
            if self.decode_edge_idx:
                rcn_edge_idx = self.edge_idx_decoder(z, triu_idx) # [num_triu_edges, 1] between 0 and 1
                rcn_edges = rcn_edge_idx * rcn_edges
        
        # Wrap outputs
        out = Outputs(x=x, 
                      rcn_x=rcn_x, 
                      edges=edges, 
                      rcn_edges=rcn_edges, 
                      z=z, mu=mu, logvar=logvar)
        return out
    
    def decode(self, z, triu_idx):
        '''
        Decodes the latent sample (num_nodes x latent_dim) 
        into node features and edge features.
        '''
        rcn_x, rcn_edges = None, None
        if self.node_decoder is not None:
            rcn_x = self.node_decoder(z)
        if self.edge_decoder is not None:
            rcn_edges = self.edge_decoder(z, triu_idx)
            if self.decode_edge_idx:
                rcn_edge_idx = self.edge_idx_decoder(z, triu_idx)
                rcn_edges = rcn_edge_idx * rcn_edges
        return rcn_x, rcn_edges
    
    def readout(self, z, context, batch_idx):
        '''Pools across nodes and returns a graph embeddings.'''
        z_with_context = torch.cat([z, context], dim=1)
        return self.pooling(z_with_context, batch_idx)
    
    def _get_decoder_labels(self, batch: Data):
        # Node features
        x = batch.x.clone()                          # [nodes_in_batch, num_node_attr]

        # Edge features, if all graphs have the same edge indices
        if not self.decode_edge_idx:
            edges = batch.edge_attr.clone()          # [edges_in_batch, num_edge_attr]
            edge_idx = batch.edge_index              # [2, edges_in_batch]
            triu_mask = edge_idx[0] < edge_idx[1]    # [edges_in_batch]
            triu_edges = edges[triu_mask]            # [num_triu_edges, num_edge_attr]
            triu_idx = edge_idx[:, triu_mask]        # [2, num_triu_edges]

        # Edges features, if graphs have different edge indices
        else:
            triu_edges, triu_idx = self._get_edge_decoder_labels(batch)

        return x, triu_edges, triu_idx
    
    def _get_edge_decoder_labels(self, batch: Data):
        '''
        Get zero-filled triu edges and indices for fully connected graphs,
        if graphs have different adjacency matrices (but the same number of nodes).
        '''
        # Get edge features
        edges = batch.edge_attr.clone()  # [edges_in_batch, num_edge_attr]

        # Convert to dense adjacency (zero-fills missing edges)
        # [batch_size, num_nodes, num_nodes, num_edge_attr]
        dense_edges = to_dense_adj(batch.edge_index, batch.batch, edges)

        # Get triu indices for each graph in batch
        num_nodes = dense_edges.size(1)
        triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)

        # Extract upper triangular edges from each graph and concatenate
        # [batch_size * num_triu_edges, num_edge_attr]
        triu_edges = dense_edges[:, triu_idx[0], triu_idx[1]].reshape(-1, edges.shape[1])

        # Get triu indices for graphs inside batch
        offset = 0
        triu_indices = []
        for i in range(batch.num_graphs):
            idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
            triu_indices.append(idx + offset)
            offset += num_nodes
        triu_idx = torch.cat(triu_indices, dim=1).to(edges.device)

        return triu_edges, triu_idx
    
    def freeze(self, exclude_pooling: bool = False):
        '''Freeze all modules except for the readout layer.'''
        for module in self.modules:
            if module is not None and (not exclude_pooling or module != self.pooling):
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        '''Unfreeze all modules.'''
        for module in self.modules:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
    
    def penalty(self):
        reg_loss = sum(m.penalty() for m in self.modules if m is not None)     
        return reg_loss
    
    def loss(self, out):
        # Reconstruction loss
        rcn_loss = 0.
        if out.rcn_x is not None:
            rcn_loss += torch.nn.functional.mse_loss(out.rcn_x, out.x, reduction='sum')
        if out.rcn_edges is not None:
            rcn_loss += torch.nn.functional.mse_loss(out.rcn_edges, out.edges, reduction='sum')

        # KL divergence
        # KL divergence between VAE's learned distribution N(mu, var) and prior N(0,1)
        # Formula: KL(N(mu,var) || N(0,1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        # out.logvar is log(var), so var = exp(logvar)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + out.logvar - out.mu.pow(2) - out.logvar.exp(), dim=1))

        # Regularization
        reg_loss = self.penalty()

        return rcn_loss + kl_loss + reg_loss
    