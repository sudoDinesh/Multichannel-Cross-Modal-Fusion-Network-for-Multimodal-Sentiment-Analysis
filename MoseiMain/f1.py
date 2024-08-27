import torch 
from channel1 import CrossModalAttention, load_features

def predict(text_feature_path, audio_feature_path, input_dim, output_dim, device='mps' if torch.backends.mps.is_available() else 'cpu'):

    text_features = load_features(text_feature_path)
    acoustic_features = load_features(audio_feature_path)

    text_features = text_features.to(device)
    acoustic_features = acoustic_features.to(device)

    model = CrossModalAttention(input_dim=input_dim, output_dim=output_dim).to(device)
    F1 = model(text_features, acoustic_features)
    return F1