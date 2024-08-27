import torch
from channel2 import load_acoustic_features, AuxiliaryModalRedundancyReduction

def predict_acoustic_features(acoustic_feature_dir, input_dim, output_dim, device='mps' if torch.backends.mps.is_available() else 'cpu'):
  acoustic_features = load_acoustic_features(acoustic_feature_dir)
  if acoustic_features is None:
      return None  
  
  acoustic_features = acoustic_features.to(device)
  model = AuxiliaryModalRedundancyReduction(input_dim=input_dim, output_dim=output_dim).to(device)
  F2 = model(acoustic_features)
  return F2