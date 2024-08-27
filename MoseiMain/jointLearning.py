import torch
import torch.nn as nn
import torch.nn.functional as F

class MCFNet(nn.Module):
    def __init__(self, feature_dim, plm_model, hidden_dim=256, num_classes=3):
        super(MCFNet, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.plm = plm_model  # Assuming plm_model is a transformer encoder or similar
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, F1, F2, F3):
        F1 = self.fc(F1)
        F2 = self.fc(F2)
        F3 = self.fc(F3)

        fused_features = F1 + F2 + F3
        plm_output = self.plm(fused_features)
        P_prime = self.mlp(plm_output)
        y_hat = self.classifier(P_prime)
        return y_hat

def load_pretrained_plm(model_name='bert-base-uncased'):
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_name)
