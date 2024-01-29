from sgmatch.models.ISONET import ISONET
import torch
from torch_geometric.utils.random import erdos_renyi_graph

N, D = 3, 3
mlp_layers = [32,16,8]

model = GMNEmbed(node_feature_dim=D, 
                enc_node_hidden_sizes=mlp_layers,
                prop_node_hidden_sizes=mlp_layers,
                prop_message_hidden_sizes=mlp_layers,
                aggr_gate_hidden_sizes=mlp_layers,
                aggr_mlp_hidden_sizes=mlp_layers)
x = torch.randn(N, D)
e = erdos_renyi_graph(N, 0.5)

output = model(x, e)

# Graph representation is a 
assert output.shape == torch.Size([mlp_layers[-1]])