import torch
import torch.nn as nn
import pandas as pd
from transformers import BertModel, BertTokenizer
from channel1 import CrossModalAttention
from channel2 import AuxiliaryModalRedundancyReduction
from channel3 import TextGuidedInformationInteractiveLearning
import numpy as np
import os
from itertools import chain

# Paths and device configuration
text_path = r"/Users/dinesh/College/final proj/attempt3/features/text"
audio_path = r"/Users/dinesh/College/final proj/attempt3/features/audio"
csv_path = r"/Users/dinesh/College/final proj/attempt3/updatedMoseiData/new_mosei.csv"
input_dim = 768  
output_dim = 128  
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load labels
labels_df = pd.read_csv(csv_path)
labels_tensor = torch.tensor(labels_df[['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']].values, dtype=torch.float32).to(device)

# Load features
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

text_features = load_features(text_path).to(device)
acoustic_features = load_features(audio_path).to(device)

# Initialize models
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

# Add a linear layer to transform combined_output to BERT's expected dimension
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Initialize transformer and adjust dimensions
feature_transformer = FeatureTransformer(input_dim=output_dim, output_dim=768).to(device)
transformed_output = feature_transformer(combined_output)

# Load pre-trained BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
plm = BertModel.from_pretrained(model_name).to(device)

# Define the custom model
class CustomModel(nn.Module):
    def __init__(self, bert_model, input_dim, num_classes):
        super(CustomModel, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()  # For multilabel classification

    def forward(self, x):
        # BERT expects input_ids and attention_mask
        # Here we use inputs_embeds to pass the precomputed embeddings
        outputs = self.bert(inputs_embeds=x)
        x = outputs.last_hidden_state.mean(dim=1)  # Pooling BERT outputs
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Create the custom model instance
model4 = CustomModel(plm, 768, 6).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(
    chain(model1.parameters(), model2.parameters(), model3.parameters(), feature_transformer.parameters(), model4.parameters()), 
    lr=0.001
)
criterion = nn.BCEWithLogitsLoss()

# Pass transformed output through the custom model
op = model4(transformed_output)



# # Training and evaluation
# def train_model(model, criterion, optimizer, features, labels):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(features)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     return loss.item()

# # Training loop (example)
# epochs = 5
# for epoch in range(epochs):
#     loss = train_model(model4, criterion, optimizer, combined_output, labels_tensor)
#     print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

# # Threshold tuning function
# def tune_thresholds(y_true, y_pred_probs):
#     best_thresholds = []
    
#     # Iterate over each emotion (output dimension)
#     for i in range(y_pred_probs.shape[1]):
#         best_f1 = 0
#         best_threshold = 0.5
        
#         # Test different thresholds from 0.1 to 0.9
#         for threshold in np.arange(0.1, 1.0, 0.1):
#             y_pred = (y_pred_probs[:, i] >= threshold).astype(int)
#             f1 = f1_score(y_true[:, i].cpu().numpy(), y_pred)
            
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold
        
#         best_thresholds.append(best_threshold)
    
#     return best_thresholds

# # After training, evaluate and tune thresholds
# model4.eval()
# with torch.no_grad():
#     outputs = model4(combined_output)
#     best_thresholds = tune_thresholds(labels_tensor, outputs)

# print("Optimal thresholds for each emotion:", best_thresholds)