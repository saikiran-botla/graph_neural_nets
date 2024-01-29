from typing import List

import torch
from torch_geometric.data import Data

from .constants import CONVS, ACTIVATION_LAYERS

class Namespace():
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.update_default_arguments()
        self.__dict__.update(kwargs)

    def update_default_args(self):
        if self.model_name == 'simgnn':
            self.update_simgnn_default_args()
        elif self.model_name == 'gmn_embed' or self.model_name == 'gmn_match':
            self.update_gmn_default_args()
        elif self.model_name == 'graphsim':
            self.update_graphsim_default_args()
        elif self.model_name == 'isonet':
            self.update_isonet_default_args()
        elif self.model_name == 'neuromatch':
            self.update_neuromatch_default_args()
        else:
            raise NotImplementedError("The model name is incorrect, please use the correct model name")

    def update_simgnn_default_args(self):
        self.__dict__.update(ntn_slices        = 16,
                             filters           = [64, 32, 16],
                             mlp_neurons       = [32,16,8,4],
                             hist_bins         = 16,
                             conv              = 'GCN',
                             activation        = 'tanh',
                             activation_slope  = None,
                             include_histogram = True)

    def update_gmn_default_args(self):
        self.__dict__.update(edge_feature_dim            = None,
                             enc_edge_hidden_sizes       = None,
                             message_net_init_scale      = 0.1,
                             node_update_type            = 'residual',
                             use_reverse_direction       = True,
                             reverse_dir_param_different = True,
                             attention_sim_metric        = 'euclidean',
                             layer_norm                  = False)

    def update_graphsim_default_args(self):
        pass

    def update_isonet_default_args(self):
        pass

    def update_neuromatch_default_args(self):
        self.__dict__.update(conv_type = 'SAGEConv',
                             dropout   = 0.0,
                             skip      = 'learnable')

class GraphPair(Data):
    """ 
    Args:
        edge_index_s (torch.Tensor): Edge Index of the Source / Query Graph
        edge_index_t (torch.Tensor): Edge Index of the Target / Corpus Graph
        x_s (torch.Tensor): Feature Matrix of the Source / Query Graph
        x_t (torch.Tensor): Feature Matrix of the Target / Corpus Graph

    Shapes:
        - **input:**
         node features :math:`(|\mathcal{V}|, F_{in})`,
         edge indices :math:`(2, |\mathcal{E}|)`,
       
    :rtype: :class:`torch_geometric.data.Data`
    """
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t, ged, norm_ged, graph_sim):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.ged = ged
        self.norm_ged = norm_ged
        self.graph_sim = graph_sim

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_1":
            return self.x1.size(0)
        elif key == "edge_index_2":
            return self.x2.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(x_s = {self.x_s.shape}, edge_index_s = {self.edge_index_s.shape}, '
                f'x_t = {self.x_t.shape}, edge_index_t = {self.edge_index_t.shape}, '
                f'graph_sim = {self.graph_sim.shape})')

def setup_linear_nn(input_dim: int, hidden_sizes: List[int]):
    r"""
    """
    mlp = torch.nn.ModuleList()
    _in = input_dim
    for i in range(len(hidden_sizes)):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        _in = _out
    
    return mlp

def setup_LRL_nn(input_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu"):
    r"""
    """
    # XXX: Better to leave this up to MLP class?
    mlp = []
    _in = input_dim
    _activation = ACTIVATION_LAYERS[activation]
    for i in range(len(hidden_sizes) - 1):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        mlp.append(_activation())
        _in = _out
    mlp.append(torch.nn.Linear(_in, hidden_sizes[-1]))
    mlp = torch.nn.Sequential(*mlp)
    
    return mlp

def setup_conv_layers(input_dim, conv_type, filters):
    r"""
    """
    convs = torch.nn.ModuleList()
    _conv = CONVS[conv_type]
    num_layers = len(filters)
    _in = input_dim
    for i in range(num_layers):
        _out = filters[i]
        convs.append(_conv(in_channels=_in, out_channels=_out))
        _in = _out

    return convs

# def cudavar(x):
#     return x.cuda() if cuda.is_available() else x