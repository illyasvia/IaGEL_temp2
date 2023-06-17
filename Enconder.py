import torch
import torch_geometric.nn
from torch import nn
from torch.nn import Flatten, GRU, Sequential, Linear, ReLU
from torch_geometric.nn import GCN, GAT, GraphSAGE, APPNP, AttentionalAggregation, GraphMultisetTransformer, PairNorm, \
    Set2Set, SumAggregation, MeanAggregation, GIN
import torch.nn.functional as F


def get_Conv(input_feature, hidden_features, output_feature, operator, dropout, num_layer):
    if operator == 'GAT':
        return GAT(in_channels=input_feature, hidden_channels=hidden_features, out_channels=output_feature, heads=4,
                   dropout=dropout, num_layers=num_layer)
    elif operator == 'GCN':
        return GCN(in_channels=input_feature, hidden_channels=hidden_features, out_channels=output_feature,
                   dropout=dropout, num_layers=num_layer)
    elif operator == 'GraphSAGE':
        return GraphSAGE(in_channels=input_feature, hidden_channels=hidden_features, out_channels=output_feature,
                         heads=3, dropout=dropout, num_layers=num_layer)
    elif operator == 'GIN':
        return GIN(in_channels=input_feature, hidden_channels=hidden_features, out_channels=output_feature,
                   dropout=dropout, num_layers=num_layer)


class Encoder(torch.nn.Module):
    def __init__(self, input_feature, hidden_features, output_feature, dropout, operator):
        super().__init__()
        # self.linear = nn.Linear(input_feature, 128)
        self.conv = get_Conv(input_feature, hidden_features, output_feature, operator, dropout, 3)
        self.appnp = APPNP(1, 0.2)
        # self.aggr = GraphMultisetTransformer(output_feature, output_feature * 2, output_feature, layer_norm=True)
        # self.aggr = Set2Set(output_feature, 3)
        self.aggr = MeanAggregation()
        self.norm = PairNorm()

    def forward(self, data, device, aggr=False):
        data = data.to(device)
        try:
            x = self.conv(data.x, data.edge_index, edge_weight=data.edge_weight)
        except TypeError:
            x = self.conv(data.x, data.edge_index)
        try:
            x = self.appnp(x, data.edge_index, data.edge_attr)
        except IndexError:
            x = self.appnp(x, data.edge_index)
        x = F.leaky_relu(x)
        if aggr:
            # x = self.aggr(x, index=torch.zeros(x.shape[0], dtype=torch.long).to(self.device))
            x = self.aggr(x, index=torch.zeros(x.shape[0], dtype=torch.long).to(device))
            x = F.leaky_relu(x)
        # x = self.norm(x)
        return x
