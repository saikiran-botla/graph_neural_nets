from sgmatch.utils.utility import Namespace

# Importing Graph Similarity Models
from .NeuroMatch import SkipLastGNN
from .SimGNN import SimGNN
from .GMN import GMNEmbed, GMNMatch
from .GraphSim import GraphSim
from .ISONET import ISONET

class graphMatcher():
    r"""
    A Wrapper Class for all the Graph Similarity / Matching models implemented in the library
    
    Args:
        av (Namespace): Object of :class:`Namespace` containing arguments to be passed to models
    
    Returns:
        The initialized model selected by the user through the 'model_name' key in dict 'args'
    """
    def __init__(self, av: Namespace):
        self.av = av
        return self.graph_match_model(self.av)

    def graph_match_model(self, av: Namespace):
        self.model = None
        if av.model_name == 'NeuroMatch':
            self.model = SkipLastGNN(self, input_dim = av.input_dim, hidden_dim = av.hidden_dim, output_dim = av.output_dim, 
                                     num_layers = av.num_layers, conv_type = av.conv_type, dropout = av.dropout,
                                     skip = av.skip)
                                     
        elif av.model_name == 'SimGNN':
            self.model = SimGNN(self, input_dim = av.input_dim, ntn_slices = av.ntn_slices, filters = av.filters,
                                mlp_neurons = av.mlp_neurons, hist_bins = av.hist_bins, conv = av.conv, 
                                activation = av.activation, activation_slope = av.activation_slope, 
                                include_histogram = av.include_histogram)

        elif av.model_name == 'GMNEmbed':
            self.model = GMNEmbed(self, node_feature_dim = av.node_feature_dim,
                                  enc_node_hidden_sizes = av.enc_edge_hidden_sizes, 
                                  prop_node_hidden_sizes = av.prop_node_hidden_sizes,
                                  prop_message_hidden_sizes = av.prop_message_hidden_sizes,
                                  aggr_gate_hidden_sizes = av.aggr_gate_hidden_sizes,
                                  aggr_mlp_hidden_sizes = av.aggr_mlp_hidden_sizes,
                                  edge_feature_dim = av.edge_feature_dim,
                                  enc_edge_hidden_sizes = av.enc_edge_hidden_sizes,
                                  message_net_init_scale = av.message_net_init_scale, 
                                  node_update_type = av.node_update_type, 
                                  use_reverse_direction = av.use_reverse_direction, 
                                  reverse_dir_param_different = av.reverse_dir_param_different, 
                                  layer_norm = av.layer_norm)

        elif av.model_name == 'GMNMatch':
            self.model = GMNMatch(self, node_feature_dim = av.node_feature_dim,
                                  enc_node_hidden_sizes = av.enc_edge_hidden_sizes, 
                                  prop_node_hidden_sizes = av.prop_node_hidden_sizes,
                                  prop_message_hidden_sizes = av.prop_message_hidden_sizes,
                                  aggr_gate_hidden_sizes = av.aggr_gate_hidden_sizes,
                                  aggr_mlp_hidden_sizes = av.aggr_mlp_hidden_sizes,
                                  edge_feature_dim = av.edge_feature_dim,
                                  enc_edge_hidden_sizes = av.enc_edge_hidden_sizes,
                                  message_net_init_scale = av.message_net_init_scale, 
                                  node_update_type = av.node_update_type, 
                                  use_reverse_direction = av.use_reverse_direction, 
                                  reverse_dir_param_different = av.reverse_dir_param_different, 
                                  attention_sim_metric= av.attention_sim_metric,
                                  layer_norm = av.layer_norm)

        elif av.model_name == 'ISONET':
            self.model = ISONET(node_feature_dim = av.node_feature_dim, 
                                enc_node_hidden_sizes = av.enc_node_hidden_sizes,
                                prop_node_hidden_sizes = av.prop_node_hidden_sizes,
                                prop_message_hidden_sizes = av.prop_message_hidden_sizes,
                                edge_feature_dim = av.edge_feature_dim,
                                enc_edge_hidden_sizes = av.enc_edge_hidden_sizes,
                                message_net_init_scale = av.message_net_init_scale,
                                node_update_type = av.node_update_type,
                                use_reverse_direction = av.use_reverse_direction, 
                                reverse_dir_param_different = av.reverse_dir_param_different,
                                layer_norm = av.layer_norm,
                                lrl_hidden_sizes = av.lrl_hidden_sizes,
                                temp = av.temp, 
                                eps = av.eps,
                                noise_factor = av.noise_factor,
                                gs_num_iters = av.gs_num_iters)
        
        elif av.model_name == 'GraphSim':
            self.model = GraphSim(input_dim = av.input_dim,
                                  gnn = av.gnn,
                                  gnn_filters = av.gnn_filters,
                                  conv_filters = av.conv_filters,
                                  mlp_neurons = av.mlp_neurons,
                                  padding_correction = av.padding_correction,
                                  resize_dim = av.resize_dim,
                                  resize_mode = av.resize_mode,
                                  gnn_activation = av.gnn_activation,
                                  mlp_activation = av.mlp_activation,
                                  gnn_dropout_p = av.gnn_dropout_p,
                                  activation_slope = av.activation_slope)

        else:
            raise NotImplementedError("The model name is incorrect, please use the correct model name")

        return self.model