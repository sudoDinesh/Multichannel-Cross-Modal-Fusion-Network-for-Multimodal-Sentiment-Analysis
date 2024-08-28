import torch
import torch.nn as nn
import numpy as np
import os

class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossModalAttention, self).__init__()
        self.text_linear = nn.Linear(input_dim, output_dim)
        self.audio_linear = nn.Linear(16, output_dim)
        self.query_linear = nn.Linear(output_dim, output_dim)
        self.key_linear = nn.Linear(output_dim, output_dim)
        self.value_linear = nn.Linear(output_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output_dim = output_dim

    def forward(self, text_features, acoustic_features):
        acoustic_features = acoustic_features.unsqueeze(1).expand(-1, text_features.size(1), -1)

        # Step 1: Embedding
        ft = self.text_linear(text_features)
        fa = self.audio_linear(acoustic_features)

        # Step 2: Linear transformations for Query, Key, Value
        Q_t = self.query_linear(ft)
        K_a = self.key_linear(fa)
        V_a = self.value_linear(fa)

        # Step 3: Cross-modal Attention
        attention_scores = self.softmax(torch.matmul(Q_t, K_a.transpose(-2, -1)))
        attention_output = torch.matmul(attention_scores, V_a)

        # Step 4: Residual connection
        F1 = ft + attention_output

        return F1
