import torch
import torch.nn as nn
import numpy as np
import os

class TextGuidedInformationInteractiveLearning(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super(TextGuidedInformationInteractiveLearning, self).__init__()
        self.query_linear = nn.Linear(output_dim, output_dim)
        self.key_linear = nn.Linear(input_dim, output_dim)
        self.value_linear = nn.Linear(input_dim, output_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads)
        self.output_linear = nn.Linear(output_dim, output_dim)

    def forward(self, dynamic_multimodal_features, static_text_features):
        # Step 1: Linear transformations for Query, Key, Value
        Q_d = self.query_linear(dynamic_multimodal_features)
        K_s = self.key_linear(static_text_features)
        V_s = self.value_linear(static_text_features)

        # Step 2: Multihead Attention
        attn_output, _ = self.multihead_attention(Q_d, K_s, V_s)

        # Step 3: Final Linear transformation and residual connection
        F3 = self.output_linear(attn_output) + Q_d

        return F3
