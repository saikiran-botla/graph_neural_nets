from typing import List, Optional, Dict
import torch
from torch.functional import Tensor
from torch_geometric.utils import to_dense_adj

from ..modules.encoder import MLPEncoder
from ..modules.scoring import GumbelSinkhornNetwork
from ..modules.propagation import GraphProp
from ..utils.utility import setup_LRL_nn

class ISONET(torch.nn.Module):
    r"""
    End-to-End implementation of the ISONET model from the `"Interpretable Neural Subgraph Matching for Graph Retrieval" 
    <https://ojs.aaai.org/index.php/AAAI/article/view/20784>`_ paper.

    Args:
        node_feature_dim (int): Input dimension of node feature embedding vectors. 
        enc_node_hidden_sizes ([int]): Number of hidden neurons in each linear layer
            for transforming the node features.
        prop_node_hidden_sizes ([int]): Number of hidden neurons in each linear layer of 
            node update MLP :obj:`f_node`. :obj:`node_feature_dim` is appended as
            the size of the final linear layer to maintain node embedding dimensionality
        prop_message_hidden_sizes ([int]): Number of hidden neurons in each linear layer of 
            message computation MLP :obj:`f_node`. Note that the message vector dimensionality 
            (:obj:`prop_message_hidden_sizes[-1]`) may not be equal to :obj:`node_feature_dim`.
        edge_feature_dim (int, optional): Input dimension of node feature embedding vectors.
            (default: :obj:`None`)
        enc_edge_hidden_sizes ([int], optional): Number of hidden neurons in each linear layer
            for transforming the edge features.
            (default: :obj:`None`)
        message_net_init_scale (float, optional): Initialisation scale for the message net output vectors.
            (default: :obj:`0.1`)
        node_update_type (str, optional): Type of update applied to node feature vectors (:obj:`"GRU"` or 
            :obj:`"MLP"` or :obj:`"residual"`). 
            (default: :obj:`"GRU"`)
        use_reverse_direction (bool, optional): Flag for whether or not to use the reverse message 
            aggregation for propagation step.
            (default: :obj:`True`)
        reverse_dir_param_different (bool, optional): Flag for whether or not message computation 
            model parameters should be shared by forward and reverse messages in propagation step.
            (default: :obj:`True`)
        layer_norm (bool, optional): Flag for applying layer normalization in propagation step.
            (default: :obj:`False`)
        lrl_hidden_sizes ([int], optional): List containing the sizes for LRL network to pass edge features
            of input graphs.
            (default: :obj:`[16,16]`)
        temp (float, optional): Temperature parameter in the Gumbel-Sinkhorn Network.
            (default: :obj:`0.1`)
        eps (float, optional): Small value for numerical stability and precision in the Gumbel-Sinkhorn Network.
            (default: :obj:`1e-20`)
        noise_factor (float, optional): Parameter which controls the magnitude of the effect of sampled Gumbel Noise.
            (default: :obj:`1`)
        gs_num_iters (int, optional): Number of iterations of Sinkhorn Row and Column scaling (in practice, 
            as little as 20 iterations are needed to achieve decent convergence for N~100).
            (default: :obj:`20`)
    """
    def __init__(self, node_feature_dim: int, enc_node_hidden_sizes: List[int], 
                prop_node_hidden_sizes: List[int], prop_message_hidden_sizes: List[int],
                edge_feature_dim: Optional[int] = None, enc_edge_hidden_sizes: Optional[List[int]] = None,
                message_net_init_scale: float = 0.1, node_update_type: str = 'GRU', 
                use_reverse_direction: bool = True, reverse_dir_param_different: bool = True, layer_norm: bool = False, 
                lrl_hidden_sizes: List[int] = [16, 16], temp: float = 0.1, eps: float = 1e-20, 
                noise_factor: float = 1, gs_num_iters: int = 20):
        super(ISONET, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Encoder Module        
        self.enc_node_layers = enc_node_hidden_sizes
        self.enc_edge_layers = enc_edge_hidden_sizes
        
        # Propagation Module
        self.prop_node_layers = prop_node_hidden_sizes
        self.prop_message_layers = prop_message_hidden_sizes

        self.message_net_init_scale = message_net_init_scale # Unused
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction
        self.reverse_dir_param_different = reverse_dir_param_different

        self.layer_norm = layer_norm
        self.prop_type = "embedding"

        # Gumbel-Sinkhorn Network Parameters
        self.lrl_hidden_sizes = lrl_hidden_sizes
        self.temp = temp
        self.eps = eps
        self.noise_factor = noise_factor
        self.gs_num_iters = gs_num_iters

        #added this function to setup layers and reset_parameters
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # Used the same nomenclature present in the ISONET paper
        self._init = MLPEncoder(self.node_feature_dim, self.enc_node_layers, edge_feature_dim=self.edge_feature_dim, 
                                   edge_hidden_sizes=self.enc_edge_layers)
        
        self._message_agg_comb = GraphProp(self.enc_node_layers[-1], self.prop_node_layers, self.prop_message_layers, 
                               edge_feature_dim=self.edge_feature_dim, message_net_init_scale=self.message_net_init_scale,
                               node_update_type=self.node_update_type, use_reverse_direction=self.use_reverse_direction,
                               reverse_dir_param_different=self.reverse_dir_param_different, layer_norm=self.layer_norm,
                               prop_type=self.prop_type)
        
        self.LRL = setup_LRL_nn(input_dim=self.prop_message_layers[-1], hidden_sizes=self.lrl_hidden_sizes)

        self.gumbel_sinkhorn = GumbelSinkhornNetwork(self.temp, self.eps, self.noise_factor, self.gs_num_iters)
        
    def reset_parameters(self):
        self._init.reset_parameters()
        self._message_agg_comb.reset_parameters()
        for layer in self.LRL[::2]:
            layer.reset_parameters()

    def embed_edges(self, node_features: Tensor, edge_index: Tensor, edge_features: Optional[Tensor] = None, num_prop: int = 5):
        from_idx = edge_index[:,0] if len(edge_index.shape) == 3 else edge_index[0]
        to_idx = edge_index[:,1] if len(edge_index.shape) == 3 else edge_index[1]

        if edge_features is not None:
            node_features, edge_features = self._init(node_features, edge_features)
        else:
            node_features = self._init(node_features)
        
        # This calculates h_u(K)
        for _ in range(num_prop):
            # TODO: Can include a list keeping track of propagation layer outputs
            node_features = self._message_agg_comb(node_features, from_idx, to_idx, edge_features)

        # Computes r_(u,v)_(K)
        edge_message = self._message_agg_comb._compute_messages(node_features, from_idx, to_idx,
                                                                        self._message_agg_comb.message_net,
                                                                        edge_features=edge_features)
        #reverse_edge_message = self._message_agg_comb._compute_messages(node_features, to_idx, from_idx,
                                                                        #self._message_agg_comb.message_net,
                                                                        #edge_features=edge_features)
        return edge_message


    def forward(self, node_features_q: Tensor, node_features_c: Tensor, edge_index_q: Tensor, edge_index_c: Tensor,
                edge_features_q: Optional[Tensor] = None, edge_features_c: Optional[Tensor] = None, 
                batch_q: Optional[Tensor] = None, batch_c: Optional[Tensor] = None, num_prop: int = 5):
        
        # Computes r_(u,v)_(K)
        edge_features_q = self.embed_edges(node_features_q, edge_index_q, edge_features_q, num_prop)
        edge_features_c = self.embed_edges(node_features_c, edge_index_c, edge_features_c, num_prop)


        # Once we have the Node and Edge Embeddings, we create R_q and R_c Matrices
        if len(edge_index_q.shape)==3 and len(edge_index_c.shape)==3:
            # Finding out the maximum num of edges in any graph - query / corpus in the batch
            max_num_edges = max([edge_index.shape[1].item() for edge_index in edge_index_q])
            max_num_edges = max(max_num_edges, max([edge_index.shape[1].item() for edge_index in edge_index_c]))

            edge_features_q_batched = torch.stack([torch.functional.pad(x, pad=(0,0,0,max_num_edges-x.shape[0]))\
                                                   for x in edge_features_q])
            edge_features_c_batched = torch.stack([torch.functional.pad(x, pad=(0,0,0,max_num_edges-x.shape[0]))\
                                                   for x in edge_features_c])
        else:
            edge_features_q_batched = to_dense_adj(edge_index=edge_index_q, batch=batch_q, edge_attr=edge_features_q)
            edge_features_c_batched = to_dense_adj(edge_index=edge_index_c, batch=batch_c, edge_attr=edge_features_c)

        
        # Passing R_q and R_c through the LRL and the Gumbel-Sinkhorn Network
        edge_features_q_batched_lrl = self.LRL(edge_features_q_batched)
        edge_features_c_batched_lrl = self.LRL(edge_features_c_batched)
        print(edge_features_q_batched_lrl.shape,edge_features_c_batched_lrl.shape)
        print(edge_features_q_batched.shape,edge_features_c_batched.shape)

        soft_permutation_matrix = self.gumbel_sinkhorn(torch.matmul(edge_features_q_batched_lrl, 
                                                                    edge_features_c_batched_lrl.permute(0,3,1,2)))

        # Calculating the Distance between corpus and query graph using the Soft Permutation Matrix
        d_cq = torch.nn.ReLU(edge_features_q_batched - torch.matmul(soft_permutation_matrix, edge_features_c_batched))
        d_cq = torch.sum(d_cq, dim=(1,2))
        return d_cq


    def __repr__(self) -> str:
        # TODO: Update __repr__ with information
        return super().__repr__()