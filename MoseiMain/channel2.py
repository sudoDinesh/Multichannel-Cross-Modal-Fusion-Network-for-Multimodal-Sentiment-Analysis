import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class AuxiliaryModalRedundancyReduction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AuxiliaryModalRedundancyReduction, self).__init__()
        self.acoustic_linear_Q = nn.Linear(input_dim, output_dim)
        self.acoustic_linear_K = nn.Linear(input_dim, output_dim)
        self.acoustic_linear_V = nn.Linear(input_dim, output_dim)

    def forward(self, acoustic_features):
        if acoustic_features.dim() == 2:
            acoustic_features = acoustic_features.unsqueeze(1) 

        acoustic_queries = self.acoustic_linear_Q(acoustic_features)
        acoustic_keys = self.acoustic_linear_K(acoustic_features)
        acoustic_values = self.acoustic_linear_V(acoustic_features)

        attention_scores = F.softmax(torch.matmul(acoustic_queries, acoustic_keys.transpose(-1, -2)) / acoustic_keys.size(-1) ** 0.5, dim=-1)
        attended_values = torch.matmul(attention_scores, acoustic_values)
        F2 = attended_values

        return F2