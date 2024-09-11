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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

torch.autograd.set_detect_anomaly(True)

# Paths and device configuration
text_path = r"/Users/dinesh/College/final proj/attempt3/features/text"
audio_path = r"/Users/dinesh/College/final proj/attempt3/features/audio"
csv_path = r"/Users/dinesh/College/final proj/attempt3/updatedMoseiData/processed_mosei.csv"
input_dim = 768  
output_dim = 128  
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load labels and preprocess
labels_df = pd.read_csv(csv_path)
sentiment_labels = labels_df['sentiment_label'].tolist()

# Encode sentiment labels to class indices
label_encoder = LabelEncoder()
label_encoder.fit(['negative', 'neutral', 'positive'])
sentiment_indices = label_encoder.transform(sentiment_labels)
sentiment_tensor = torch.tensor(sentiment_indices, dtype=torch.long).to(device)

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
model2 = AuxiliaryModalRedundancyReduction(input_dim=16, output_dim=output_dim).to(device)
model3 = TextGuidedInformationInteractiveLearning(input_dim=input_dim, output_dim=output_dim).to(device)

# Add a linear layer to transform combined_output to BERT's expected dimension
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Initialize transformer and adjust dimensions
feature_transformer = FeatureTransformer(input_dim=output_dim, output_dim=768).to(device)

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
    
    def forward(self, x):
        # Use inputs_embeds to pass precomputed embeddings to BERT
        outputs = self.bert(inputs_embeds=x)
        x = outputs.last_hidden_state.mean(dim=1)  # Pooling BERT outputs
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Apply softmax to get class probabilities
        return nn.Softmax(dim=1)(x)

num_classes = 3
# Create the custom model instance
model4 = CustomModel(plm, 768, num_classes).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(
    chain(model1.parameters(), model2.parameters(), model3.parameters(), feature_transformer.parameters(), model4.parameters()), 
    lr=0.001
)
criterion = nn.CrossEntropyLoss()

# Training and evaluation
def train_model(model, feature_transformer, criterion, optimizer, text_features, acoustic_features, labels):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass through the first three channels
    F1 = model1(text_features, acoustic_features)
    F2 = model2(acoustic_features)
    F3 = model3(F1, text_features)

    # Combine outputs using element-wise addition
    combined_output = F1 + F2 + F3

    # Transform features
    transformed_output = feature_transformer(combined_output)
    
    # Forward pass through BERT-based model
    outputs = model(transformed_output)
    
    # Compute loss
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.item(), outputs

# Function to calculate evaluation metrics and print sample labels and predictions
def evaluate_model(outputs, labels, sample_indices):
    # Convert probabilities to class predictions using argmax
    preds = torch.argmax(outputs, dim=1).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    # Print sample labels and predictions
    for i in sample_indices:
        label = label_encoder.inverse_transform([labels[i]])[0]
        prediction = label_encoder.inverse_transform([preds[i]])[0]
        print(f"Sample {i} - Label: {label}, Prediction: {prediction}")
    
    # Count number of records for each class
    class_counts = {label: 0 for label in label_encoder.classes_}
    for pred in preds:
        class_label = label_encoder.inverse_transform([pred])[0]
        class_counts[class_label] += 1
    
    # Print class counts
    print("Class counts for predictions:")
    for class_label, count in class_counts.items():
        print(f"{class_label}: {count}")

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return accuracy, precision, recall, f1

# Training loop
epochs = 5
sample_indices = [0, 1, 2]  # Adjust as needed to print different sample indices
for epoch in range(epochs):
    loss, outputs = train_model(model4, feature_transformer, criterion, optimizer, text_features, acoustic_features, sentiment_tensor)
    accuracy, precision, recall, f1 = evaluate_model(outputs, sentiment_tensor, sample_indices)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
