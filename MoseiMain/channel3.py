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

def load_features(feature_dir):
    feature_files = sorted(os.listdir(feature_dir))
    features = []
    for file in feature_files:
        if file.endswith('.npy'):  # Load only .npy files
            file_path = os.path.join(feature_dir, file)
            try:
                data = np.load(file_path, allow_pickle=True)
                features.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e} on Channel 3")
    features = np.stack(features, axis=0)  # Stack into a single tensor
    return torch.tensor(features, dtype=torch.float32)
