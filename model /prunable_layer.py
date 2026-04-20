import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.gate_scores, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def gates(self):
        return torch.sigmoid(self.gate_scores)

    def forward(self, x):
        pruned_weight = self.weight * self.gates()
        return F.linear(x, pruned_weight, self.bias)
