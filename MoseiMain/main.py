import torch
import torch.nn as nn
import pandas as pd
from transformers import BertModel, BertTokenizer
from f1 import predict as predict_f1
from f2 import predict_acoustic_features as predict_f2
from f3 import predict_third_channel as predict_f3

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

# Extract features using the three channels
f1_output = predict_f1(text_path, audio_path, input_dim, output_dim)
f2_output = predict_f2(audio_path, 16, output_dim)
f3_output = predict_f3(f1_output, text_path, input_dim, output_dim)
print(f1_output.shape)
print(f2_output.shape)
print(f3_output.shape)
# Ensure all outputs have the same shape
f1_output = f1_output.squeeze(1)
f2_output = f2_output.squeeze(1)
f3_output = f3_output.squeeze(1)

# Combine outputs using element-wise addition
combined_output = f1_output + f2_output + f3_output

# Load the pre-trained language model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
plm = BertModel.from_pretrained(model_name).to(device)

# Define the classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

classifier = Classifier(input_dim, output_dim).to(device)

# Define model and optimizer
model = nn.Sequential(plm, classifier)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10  # Adjust the number of epochs as desired
batch_size = 32  # Adjust batch size as needed

for epoch in range(num_epochs):
    for i in range(0, len(labels_tensor), batch_size):
        batch_inputs = combined_output[i:i+batch_size]
        batch_labels = labels_tensor[i:i+batch_size]

        optimizer.zero_grad()
        try:
            outputs = classifier(batch_inputs)
        except RuntimeError as e:
            # Handle potential RuntimeError (e.g., due to input type mismatch)
            if "Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int" in str(e):
                # Convert text data to integer IDs if necessary
                batch_inputs = batch_inputs.long()
                outputs = classifier(batch_inputs)
            else:
                raise e  # Re-raise the error if it's not the expected type

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set (if available)
    # ...

# After training, evaluate on the test set
with torch.no_grad():
    predicted_labels = classifier(combined_output)
    accuracy = (predicted_labels.argmax(dim=1) == labels_tensor.argmax(dim=1)).float().mean()
    print("Test Accuracy:", accuracy)
