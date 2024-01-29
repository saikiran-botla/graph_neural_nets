from typing import Optional, List

import torch
from torch.functional import Tensor

class NeuralTensorNetwork(torch.nn.Module):
    r"""
    Neural Tensor Network layer from the 
    `"SimGNN: A Neural Network Approach to Fast Graph Similarity Computation"
    <https://arxiv.org/pdf/1808.05689.pdf>`_ paper

    TODO: Include latex formula for NTN interaction score computation
    
    Args:
        input_dim: Input dimension of the graph-level embeddings slices. That is, number of slices (K) the weight tensor possesses. Often 
            interpreted as the number of entity-pair (in this use case - pairwise
            node) relations the data might possess.
        activation: Non-linearity applied on the computed output of the layer
    """
    def __init__(self, input_dim: int, slices: int = 16, activation: str = "tanh"):
        super(NeuralTensorNetwork, self).__init__()
        self.input_dim = input_dim
        self.slices = slices # K: hyperparameter
        self.activation = activation

        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self):
        # XXX: Will arranging weight tensor as (k, d, d) cause problems in batching at dim 0?
        self.weight_tensor = torch.nn.Parameter(torch.Tensor(self.slices, self.input_dim, self.input_dim))
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.slices, 2 * self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.slices, 1))
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_tensor)
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)
    
    def forward(self, h_i: Tensor, h_j: Tensor) -> torch.Tensor:
        r"""
        Args:
            h_i: Graph-level Embedding of the Source/Query graph 
            h_j: Graph-level Embedding of the Corpus/Target graph

        Returns:
            (K, 1) graph-graph interaction score vector
        """
        scores = torch.matmul(h_i.unsqueeze(-2).unsqueeze(-2), self.weight_tensor)
        scores = torch.matmul(scores, h_j.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
        scores += torch.matmul(self.weight_matrix, torch.cat([h_i, h_j], dim=-1).unsqueeze(-1))
        scores += self.bias
        
        # TODO: need to remove this from here and include it in a function in utils
        if self.activation == "tanh":
            _activation = torch.nn.functional.tanh
        elif self.activation == "sigmoid":
            _activation = torch.nn.functional.sigmoid
        
        scores = _activation(scores)

        return scores
    
class GumbelSinkhornNetwork(torch.nn.Module):
    r"""
    Implementation of the Gumbel-Sinkhorn Network from the `"Learning Latent Permutations with Gumbel-Sinkhorn Networks" 
    <https://arxiv.org/pdf/1802.08665.pdf>`_ paper.

    Args:
        temp (float, optional): Temperature parameter for softmax distribution. Lower the value causes 
            softmax probabilities to approach a one-hot vector. Thus, this is a differentiable way to 
            approach a categorical distribution.
            (default: :obj:`0.1`)
        eps (float, optional): Small value for numerical stability and precision.
            (default: :obj:`1e-20`)
        noise_factor (float, optional): Parameter which controls the magnitude of the effect of sampled Gumbel Noise
            (default: :obj:`1`)
        n_iters (int, optional): Number of iterations of Sinkhorn Row and Column scaling (in practice, as little as 20
            iterations are needed to achieve decent convergence for N~100).
            (default: :obj:`20`)
    """
    def __init__(self, temp: float = 0.1, eps: float = 1e-20, noise_factor: float = 1, n_iters: int = 20):
        super().__init__()
        self.temp = temp
        self.eps = eps
        self.noise_factor = noise_factor
        self.n_iters = n_iters

    def sample_from_gumbel(self, shape:List[int], eps:int=1e-20):
        U = torch.rand(shape).float()
        return -torch.log(eps - torch.log(U + eps))

    def forward(self, log_alpha:Tensor) -> torch.Tensor:
        r"""
        Args:
            log_alpha (torch.Tensor): A Tensor with elements being the logarithms of assignment probabilites

        Shapes:
            - **input:**
             log_alpha: 2D Tensor of shape :math:`(N, N)` or 3D tensor of shape :math:`(\mbox{batch_size}, N, N)`
            - **output:** Doubly Stochastic Matrix :math:`(\mbox{batch_size}, N, N)`
            
        Returns:
            A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are converted to 
            3D tensors with batch_size equals to 1)

        :rtype: :class:`torch.Tensor`
        """
        batch_size = log_alpha.size()[0]
        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)
        
        # Sampling from Gumbel Distribution
        shape = [batch_size, n, n]
        noise = self.sample_from_gumbel(shape, self.eps)*self.noise_factor
        log_alpha = log_alpha + noise

        # Iteration 0 of GS Network
        log_alpha = log_alpha/self.temp

        for i in range(self.n_iters):
            # Row Scaling
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
            # Column Scaling
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        return torch.exp(log_alpha)

def similarity(h_i, h_j, mode:str = "cosine"):
    # BUG: similarity may not return the correct product
    # BUG: cosine returns 0-1 normalised similarity and others return unnormalized distance
    if mode == "cosine":
        return torch.nn.functional.cosine_similarity(h_i, h_j, dim=-1)
    if mode == "euclidean":
        return torch.cdist(h_i, h_j, p=2)
    if mode == "manhattan":
        return torch.cdist(h_i, h_j, p=1)
    if mode == "hamming":
        return torch.cdist(h_i, h_j, p=0)
    if mode == "hinge":
        prod = 1 - torch.matmul(h_i, h_j)
        return torch.max(torch.zeros(prod.shape), prod)
    if mode == "min_sum":
        return torch.sum(torch.min(h_i, h_j), dim=-1)