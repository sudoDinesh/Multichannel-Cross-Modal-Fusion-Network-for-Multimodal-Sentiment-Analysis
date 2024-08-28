from itertools import chain
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertModel, BertTokenizer
from channel1 import CrossModalAttention
from channel2 import AuxiliaryModalRedundancyReduction
from channel3 import TextGuidedInformationInteractiveLearning
import numpy as np
import os

text_path = r"/Users/dinesh/College/final proj/attempt3/features/text"
audio_path = r"/Users/dinesh/College/final proj/attempt3/features/audio"
csv_path = r"/Users/dinesh/College/final proj/attempt3/updatedMoseiData/new_mosei.csv"

input_dim = 768  
output_dim = 128  
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

labels_df = pd.read_csv(csv_path)
labels_tensor = torch.tensor(labels_df[['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']].values, dtype=torch.float32).to(device)


def load_features(feature_dir):
    feature_files = sorted(os.listdir(feature_dir))
    features = []
    for file in feature_files:
        if file.endswith('.npy'):  
            file_path = os.path.join(feature_dir, file)
            try:
                data = np.load(file_path, allow_pickle=True)
                features.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    features = np.stack(features, axis=0)  
    return torch.tensor(features, dtype=torch.float32)


text_features = load_features(text_path)
acoustic_features = load_features(audio_path)

text_features = text_features.to(device)
acoustic_features = acoustic_features.to(device)

model1 = CrossModalAttention(input_dim=input_dim, output_dim=output_dim).to(device)
F1 = model1(text_features, acoustic_features)

model2 = AuxiliaryModalRedundancyReduction(input_dim=16, output_dim=output_dim).to(device)
F2 = model2(acoustic_features)

model3 = TextGuidedInformationInteractiveLearning(input_dim=input_dim, output_dim=output_dim).to(device)
F3 = model3(F1, text_features)


print(F1.shape)
print(F2.shape)
print(F3.shape)



# Combine outputs using element-wise addition
combined_output = F1 + F2 + F3

# Load the pre-trained language model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
plm = BertModel.from_pretrained(model_name).to(device)


classes = 6

# Define the classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

classifier = Classifier(input_dim, classes).to(device)

# Define model and optimizer
model4 = nn.Sequential(plm, classifier)
optimizer = torch.optim.Adam(
    chain(model1.parameters(), model2.parameters(), model3.parameters(), model4.parameters()), 
    lr=0.001
)
criterion = nn.CrossEntropyLoss()
