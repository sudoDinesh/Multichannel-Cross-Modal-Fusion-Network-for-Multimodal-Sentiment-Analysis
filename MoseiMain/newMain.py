import torch
import torch.nn as nn
import pandas as pd
from transformers import BertModel, BertTokenizer
from channel1 import CrossModalAttention
from channel2 import AuxiliaryModalRedundancyReduction
from channel3 import TextGuidedInformationInteractiveLearning
import numpy as np
import os

# Replace with your actual paths
text_path = r"/Users/dinesh/College/final proj/attempt3/features/text"
audio_path = r"/Users/dinesh/College/final proj/attempt3/features/audio"
csv_path = r"/Users/dinesh/College/final proj/attempt3/updatedMoseiData/new_mosei.csv"

input_dim = 768  # Adjust as needed
output_dim = 128  # Adjust as needed
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load the labels from the CSV file
labels_df = pd.read_csv(csv_path)
labels_tensor = torch.tensor(labels_df[['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']].values, dtype=torch.float32).to(device)


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
                print(f"Error loading {file_path}: {e}")
    features = np.stack(features, axis=0)  # Stack into a single tensor
    return torch.tensor(features, dtype=torch.float32)


text_features = load_features(text_path)
acoustic_features = load_features(audio_path)

text_features = text_features.to(device)
acoustic_features = acoustic_features.to(device)

model = CrossModalAttention(input_dim=input_dim, output_dim=output_dim).to(device)
F1 = model(text_features, acoustic_features)

model = AuxiliaryModalRedundancyReduction(input_dim=input_dim, output_dim=output_dim).to(device)
F2 = model(acoustic_features)

model = TextGuidedInformationInteractiveLearning(input_dim=input_dim, output_dim=output_dim).to(device)
F3 = model(F1, text_features)