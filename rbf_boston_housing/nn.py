from rbf_boston_housing.rbf import RBF
import torch.nn as nn


class RBFNetwork(nn.Module):

    def __init__(self, layer_widths, dropout_rate):
        super(RBFNetwork, self).__init__()
        self.rbf_layer = RBF(layer_widths, layer_widths)
        self.linear_layer = nn.Linear(layer_widths, 1)
        self.dropout = nn.Dropout(dropout_rate)
        # self.rbf_layer = nn.ModuleList()
        # self.linear_layers = nn.ModuleList()
        # for i in range(len(layer_widths) - 1):
        #     self.rbf_layers.append(RBF(layer_widths[i], layer_centres[i]))
        #     self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i + 1]))

    def forward(self, x):
        out = x
        out = self.rbf_layer.forward(out)
        out = self.dropout(out)
        out = self.linear_layer(out)
        return out
