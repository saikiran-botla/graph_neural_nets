from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torch.nn import Linear, Dropout, Sequential, ModuleList
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch, unbatch, degree
from torch_scatter import scatter_mean, scatter_add

from ..utils.utility import setup_linear_nn, setup_conv_layers, setup_LRL_nn
from ..utils.constants import ACTIVATION_LAYERS, ACTIVATIONS

class GraphSim(torch.nn.Module):
    r"""
    End to end implementation of GraphSim from the `"Learning-based Efficient Graph Similarity Computation via Multi-Scale
    Convolutional Set Matching" <https://arxiv.org/abs/1809.04440>`_ paper.

    NOTE: Model assumes that node features of input graph data are arranged according to Breadth-First Search of the graph
    
    TODO: Provide description of implementation and differences from paper if any

    Args:
        input_dim (int): Input dimension of node feature vectors.
        gnn (str, optional): Type of Graph Neural Network to use to embed the node features (:obj:`"Neuro-PNA"` or 
            :obj:`"PNA"` or :obj:`"GCN"` or :obj:`"GAT"`or :obj:`"SAGE"` or :obj:`"GIN"`
            or :obj:`"graph"` or :obj:`"gated"`).
            (default: :obj:`'GCN'`)
        gnn_filters ([int], optional): Number of hidden neurons in each layer of
            the GNN for embedding input node features. 
            (default: :obj:`[64,32,16]`)
        conv_filters (torch.nn.ModuleList, optional): List of Convolution Filters
            to be applied to each similarity matrix generated from each GNN pass. 
            (default: :obj:`None`)
        mlp_neurons ([int], optional): Number of hidden neurons in each layer of decoder MLP
            (default: :obj:`[32,16,8,4,1]`)
        padding_correction (bool, optional): Flag whether to include padding correction as specified in the paper
            which is voided due to batching of graphs
            (default: :obj:`True`)
        resize_dim (int, optional): Dimension to resize the similarity image matrices to. 
            (default: :obj:`10`)
        resize_mode (str, optional): Interpolation method to resize the similarity images
            (:obj:`nearest'` | :obj:`'linear'` | :obj:`'bilinear'` | :obj:`'bicubic'` | :obj:`'trilinear'` | 
            :obj:`'area'` | :obj:`'nearest-exact'`).
            (default: :obj:`'bilinear'`)
        gnn_activation (str, optional): Activation to be used in the GNN layers
            (default: :obj:`relu`)
        mlp_activation (str, optional): Activation to be used in the MLP decoder layers
            (default: :obj:`relu`)
        activation_slope (int, optional): Slope of negative part in case of :obj:`"leaky_relu"` activation
            (default: :obj:`0.1`)
    """ 
    def __init__(self, input_dim: int, gnn: str = "GCN", gnn_filters: List[int] = [64, 32, 16], conv_filters: ModuleList = None, 
                 mlp_neurons: List[int] = [32,16,8,4,1], padding_correction: bool = True, resize_dim: int = 10, 
                 resize_mode = "bilinear", gnn_activation: str = "relu", mlp_activation: str = "relu", gnn_dropout_p: float = 0.5,
                 activation_slope: Optional[float] = 0.1):
        super(GraphSim, self).__init__()
        # GNN Arguments
        self.input_dim = input_dim
        self.gnn_type = gnn
        self.gnn_filters = gnn_filters
        self.gnn_activation = gnn_activation
        self.gnn_dropout_p = gnn_dropout_p
        
        # Similarity Matrix Arguments
        self.padding_correction = padding_correction
        self.sim_mat_dim = resize_dim
        self.resize_mode = resize_mode

        # Convolution Layer
        # XXX: Should users be allowed to pass torch.nn.Sequential layers for Conv directly?
        # XXX: Do we need to make additional Image Conv setup utility methods?
        self.conv_filters = conv_filters

        # MLP Layer which takes Convolution Output as Input
        self.mlp_neurons = mlp_neurons
        self.mlp_activation = mlp_activation

        self.setup_layers()
        # self.reset_parameters()

    def setup_layers(self):
        # GCN Layers 
        self.gnn_layers = setup_conv_layers(self.input_dim, self.gnn_type, filters=self.gnn_filters)

        
        # Fully Connected Layer
        W = torch.randn(2, 1, self.sim_mat_dim, self.sim_mat_dim) # Dummy Matrix
        W = self.conv_filters(W).view(2,-1)
        self.mlp = setup_LRL_nn(input_dim=W.shape[1], hidden_sizes=self.mlp_neurons, activation=self.mlp_activation)
        del W

        # Scoring Layer to get the final Graph Similarity between (0,1)
        self.scoring_layer = setup_linear_nn(input_dim=self.mlp_neurons[-1], hidden_sizes=[1,])

    def reset_parameters(self):
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameter()

        # TODO: Test correctness
        self.conv_filters.reset_parameters()
        self.mlp.reset_parameters()
        

    def forward(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor, edge_index_j: Tensor, batch_i:Tensor, batch_j:Tensor):
        """
         Forward pass with graphs.
         :param x_i (Tensor): A (N_1+N_2+...+N_B, D) tensor containing 'i' Graphs Features.
         :param x_j (Tensor): A (N_1+N_2+...+N_B, D) tensor containing 'j' Graphs Features
         :param edge_index_i (Tensor) : A (2, num_edges) tensor containing edges of Graphs in 'i'
         :param edge_index_j (Tensor) : A (2, num_edges) tensor containing edges of Graphs in 'j'
         :param batch_i (Tensor) : A (B,) tensor containing information of the graph each node belongs to
         :param batch_j (Tensor) : A (B,) tensor containing information of the graph each node belongs to
         :return score (Tensor): Similarity score.
         """
        # Tensor of number of nodes in each graph
        N_i, N_j = degree(batch_i), degree(batch_j) # Size (B,)
        N_i_j = torch.maximum(N_i, N_j) # (B,)
        B = batch_i.shape[0]

        # Converting Input Nodes to Similarity Matrices
        sim_matrix_list = []
        gnn_activation = ACTIVATION_LAYERS[self.gnn_activation]
        for layer_num, gnn_layer in enumerate(self.gnn_layers):
            # Pass through GNN
            x_i = gnn_layer(x_i,edge_index_i) #updating
            x_j = gnn_layer(x_j,edge_index_j) #updating

            if layer_num != len(self.gnn_layers)-1:
                x_i = gnn_activation(x_i) # Default is a ReLU activation
                x_i = Dropout(x_i, p=self.gnn_dropout_p, training=self.training)
                x_j = gnn_activation(x_j)
                x_j = Dropout(x_j, p=self.gnn_dropout_p, training=self.training)

            # Generate Similarity Matrix after (layer_num + 1)th GNN Embedding Pass
            h_i, mask_i = to_dense_batch(x_i, batch_i) # (B, N_max, D), {0,1}^(B, N_max) - 1 if true node, 0 if padded 
            h_j, mask_j = to_dense_batch(x_j, batch_j) # (B, N_max, D), {0,1}^(B, N_max) - 1 if true node, 0 if padded
            sim_matrix = torch.matmul(h_i, h_j.permute(0,2,1)) # (B, N_max_i, D) * (B, D, N_max_j) -> (B, N_max_i, N_max_j)

            # XXX: Can we just collect Similarity Matrices in this pass and perform other operations outside this loop?
            # Correcting Similarity Matrix Size as per Paper's Specifications
            if self.padding_correction:
                N_max_batch_i, N_max_batch_j = sim_matrix.shape[0], sim_matrix.shape[1] 
                pads_i, pads_j = N_i_j - N_max_batch_i, N_i_j - N_max_batch_j
                repadded_sim_matrices = list(map(lambda x, pad_i, pad_j: F.pad(x,(0,pad_i,0,pad_j)), 
                                                list(sim_matrix), pads_i, pads_j))
                resized_sim_matrices = list(map(lambda x: F.interpolate(x.unsqueeze(0), size=self.sim_mat_dim,
                                                                        mode=self.resize_mode).squeeze(0), repadded_sim_matrices))
                batched_resized_sim_matrices = torch.stack(resized_sim_matrices)
            else:
                batched_resized_sim_matrices = F.interpolate(sim_matrix, size=self.sim_mat_dim, mode=self.resize_mode)
            sim_matrix_list.append(batched_resized_sim_matrices) # [(B, N_reduced, N_reduced)]

        # Passing similarity images through Conv2d and MLP to get similarity score
        
        # sim_matrix_batch = torch.stack(sim_matrix_list, dim=-1) # (B, N_reduced, N_reduced, N_gnn_layers)
        # XXX: Can we use Group Convolutions instead of Looping over Convolved Multi-Scale Sim Matrices
        #sim_matrix_img_batch = sim_matrix_batch.permute(0,3,1,2)
        image_embedding_list = list(map(lambda x, conv_layer: conv_layer(x.unsqueeze(0)).squeeze(0), 
                                        sim_matrix_list, self.conv_filters)) # [(C,H,W),]
        similarity_scores = torch.stack(image_embedding_list).view(B,-1) # (B, C*H*W)

        # Passing Input to MLP
        similarity_scores = self.mlp(similarity_scores)
        similarity_scores = self.scoring_layer(similarity_scores)
        similarity_scores = torch.nn.Sigmoid(similarity_scores)
        
        return similarity_scores.view(-1)
    
class GraphSim_v2(torch.nn.Module):
    r"""
    A more efficient implementation of GraphSim from the `"Learning-based Efficient Graph Similarity Computation via Multi-Scale
    Convolutional Set Matching" <https://arxiv.org/abs/1809.04440>`_ paper.

    Uses the grouped convolution layer in :object:`PyTorch`to speed up the embedding of heirarchical similarity
    image matrices by parallelizing computations. Prefer using this variant over version 1 if the convolution
    network architecture is the same for all similarity image matrices.
    
    TODO: Provide description of implementation and differences from paper if any and update argument description

    Args:
        input_dim (int): Input dimension of node feature vectors.
        gnn (str, optional): Type of Graph Neural Network to use to embed the node features (:obj:`"Neuro-PNA"` or 
            :obj:`"PNA"` or :obj:`"GCN"` or :obj:`"GAT"`or :obj:`"SAGE"` or :obj:`"GIN"`
            or :obj:`"graph"` or :obj:`"gated"`).
            (default: :obj:`'GCN'`)
        gnn_filters ([int], optional): Number of hidden neurons in each layer of
            the GNN for embedding input node features. 
            (default: :obj:`[64,32,16]`)
        conv_filters (torch.nn.ModuleList, optional): List of Convolution Filters
            to be applied to each similarity matrix generated from each GNN pass. 
            (default: :obj:`None`)
        mlp_neurons ([int], optional): Number of hidden neurons in each layer of decoder MLP
            (default: :obj:`[32,16,8,4,1]`)
        padding_correction (bool, optional): Flag whether to include padding correction as specified in the paper
            which is voided due to batching of graphs
            (default: :obj:`True`)
        resize_dim (int, optional): Dimension to resize the similarity image matrices to. 
            (default: :obj:`10`)
        resize_mode (str, optional): Interpolation method to resize the similarity images
            (:obj:`nearest'` | :obj:`'linear'` | :obj:`'bilinear'` | :obj:`'bicubic'` | :obj:`'trilinear'` | 
            :obj:`'area'` | :obj:`'nearest-exact'`).
            (default: :obj:`'bilinear'`)
        gnn_activation (str, optional): Activation to be used in the GNN layers
            (default: :obj:`relu`)
        mlp_activation (str, optional): Activation to be used in the MLP decoder layers
            (default: :obj:`relu`)
        activation_slope (int, optional): Slope of negative part in case of :obj:`"leaky_relu"` activation
            (default: :obj:`0.1`)
    """ 
    def __init__(self, input_dim: int, gnn: str = "GCN", gnn_filters: List[int] = [64, 32, 16], conv_filters: ModuleList = None, 
                 mlp_neurons: List[int] = [32,16,8,4,1], padding_correction: bool = True, resize_dim: int = 10, 
                 resize_mode = "bilinear", gnn_activation: str = "relu", mlp_activation: str = "relu", gnn_dropout_p: float = 0.5,
                 activation_slope: Optional[float] = 0.1):
        super(GraphSim, self).__init__()
        # GNN Arguments
        self.input_dim = input_dim
        self.gnn_type = gnn
        self.gnn_filters = gnn_filters
        self.gnn_activation = gnn_activation
        self.gnn_dropout_p = gnn_dropout_p
        
        # Similarity Matrix Arguments
        self.padding_correction = padding_correction
        self.sim_mat_dim = resize_dim
        self.resize_mode = resize_mode

        # Convolution Layer
        # XXX: Should users be allowed to pass torch.nn.Sequential layers for Conv directly?
        # XXX: Do we need to make additional Image Conv setup utility methods?
        self.conv_filters = conv_filters

        # MLP Layer which takes Convolution Output as Input
        self.mlp_neurons = mlp_neurons
        self.mlp_activation = mlp_activation

        self.setup_layers()
        # self.reset_parameters()

    def setup_layers(self):
        # GCN Layers 
        self.gnn_layers = setup_conv_layers(self.input_dim, self.gnn_type, filters=self.gnn_filters)

        # Fully Connected Layer
        W = torch.randn(2, 1, self.sim_mat_dim, self.sim_mat_dim) # Dummy Matrix
        W = self.conv_filters(W).view(2,-1)
        self.mlp = setup_LRL_nn(input_dim=W.shape[1], hidden_sizes=self.mlp_neurons, activation=self.mlp_activation)
        del W

        # Scoring Layer to get the final Graph Similarity between (0,1)
        self.scoring_layer = setup_linear_nn(input_dim=self.mlp_neurons[-1], hidden_sizes=[1,])

    def reset_parameters(self):
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameter()

        # TODO: Test correctness
        self.conv_filters.reset_parameters()
        self.mlp.reset_parameters()
        

    def forward(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor, edge_index_j: Tensor, batch_i:Tensor, batch_j:Tensor):
        """
         Forward pass with graphs.
         :param x_i (Tensor): A (N_1+N_2+...+N_B, D) tensor containing 'i' Graphs Features.
         :param x_j (Tensor): A (N_1+N_2+...+N_B, D) tensor containing 'j' Graphs Features
         :param edge_index_i (Tensor) : A (2, num_edges) tensor containing edges of Graphs in 'i'
         :param edge_index_j (Tensor) : A (2, num_edges) tensor containing edges of Graphs in 'j'
         :param batch_i (Tensor) : A (B,) tensor containing information of the graph each node belongs to
         :param batch_j (Tensor) : A (B,) tensor containing information of the graph each node belongs to
         :return score (Tensor): Similarity score.
         """
        # Tensor of number of nodes in each graph
        N_i, N_j = degree(batch_i), degree(batch_j) # Size (B,)
        N_i_j = torch.maximum(N_i, N_j) # (B,)
        B = batch_i.shape[0]

        # Converting Input Nodes to Similarity Matrices
        sim_matrix_list = []
        gnn_activation = ACTIVATION_LAYERS[self.gnn_activation]
        for layer_num, gnn_layer in enumerate(self.gnn_layers):
            # Pass through GNN
            x_i = gnn_layer(x_i)
            x_j = gnn_layer(x_j)

            if layer_num != len(self.gnn_layers)-1:
                x_i = gnn_activation(x_i) # Default is a ReLU activation
                x_i = Dropout(x_i, p=self.gnn_dropout_p, training=self.training)
                x_j = gnn_activation(x_j)
                x_j = Dropout(x_j, p=self.gnn_dropout_p, training=self.training)

            # Generate Similarity Matrix after (layer_num + 1)th GNN Embedding Pass
            h_i, mask_i = to_dense_batch(x_i, batch_i) # (B, N_max, D), {0,1}^(B, N_max) - 1 if true node, 0 if padded 
            h_j, mask_j = to_dense_batch(x_j, batch_j) # (B, N_max, D), {0,1}^(B, N_max) - 1 if true node, 0 if padded
            sim_matrix = torch.matmul(h_i, h_j.permute(0,2,1)) # (B, N_max_i, D) * (B, D, N_max_j) -> (B, N_max_i, N_max_j)

            # XXX: Can we just collect Similarity Matrices in this pass and perform other operations outside this loop?
            # Correcting Similarity Matrix Size as per Paper's Specifications
            if self.padding_correction:
                N_max_batch_i, N_max_batch_j = sim_matrix.shape[0], sim_matrix.shape[1] 
                pads_i, pads_j = N_i_j - N_max_batch_i, N_i_j - N_max_batch_j
                repadded_sim_matrices = list(map(lambda x, pad_i, pad_j: F.pad(x,(0,pad_i,0,pad_j)), 
                                                list(sim_matrix), pads_i, pads_j))
                resized_sim_matrices = list(map(lambda x: F.interpolate(x.unsqueeze(0), size=self.sim_mat_dim,
                                                                        mode=self.resize_mode).squeeze(0), repadded_sim_matrices))
                batched_resized_sim_matrices = torch.stack(resized_sim_matrices)
            else:
                batched_resized_sim_matrices = F.interpolate(sim_matrix, size=self.sim_mat_dim, mode=self.resize_mode)
            sim_matrix_list.append(batched_resized_sim_matrices) # [(B, N_reduced, N_reduced)]

        # Passing similarity images through Conv2d and MLP to get similarity score
        
        # sim_matrix_batch = torch.stack(sim_matrix_list, dim=-1) # (B, N_reduced, N_reduced, N_gnn_layers)
        # XXX: Can we use Group Convolutions instead of Looping over Convolved Multi-Scale Sim Matrices
        #sim_matrix_img_batch = sim_matrix_batch.permute(0,3,1,2)
        image_embedding_list = list(map(lambda x, conv_layer: conv_layer(x.unsqueeze(0)).squeeze(0), 
                                        sim_matrix_list, self.conv_filters)) # [(C,H,W),]
        similarity_scores = torch.stack(image_embedding_list).view(B,-1) # (B, C*H*W)

        # Passing Input to MLP
        similarity_scores = self.mlp(similarity_scores)
        similarity_scores = self.scoring_layer(similarity_scores)
        similarity_scores = torch.nn.Sigmoid(similarity_scores)
        
        return similarity_scores.view(-1)