import torch
from channel3 import TextGuidedInformationInteractiveLearning, load_features

def predict_third_channel(f1_output, text_feature_path, input_dim, output_dim, device='mps' if torch.backends.mps.is_available() else 'cpu'):

    static_text_features = load_features(text_feature_path)

    static_text_features = static_text_features.to(device)
    f1_output = f1_output.to(device)

    model = TextGuidedInformationInteractiveLearning(input_dim=input_dim, output_dim=output_dim).to(device)
    F3 = model(f1_output, static_text_features)
    return F3
