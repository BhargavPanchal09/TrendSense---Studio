"""Versioned content regressor. Preprocessing travels with the saved weights."""
import torch
from torch import nn


class ContentRegressor(nn.Module):
    def __init__(self, width=128, dropout=0.35):
        super().__init__()
        # Buffers are saved with the model and do not receive gradients.
        self.register_buffer('feature_mean', torch.zeros(385))
        self.register_buffer('feature_scale', torch.ones(385))
        self.network = nn.Sequential(
            nn.Linear(385, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.LayerNorm(width // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width // 2, 1),
        )

    def forward(self, features):
        # Train-only standardization gives the small CLIP feature a usable scale.
        x = (features - self.feature_mean) / self.feature_scale
        # Bound predictions to the same range as the target during training too.
        return 100 * torch.sigmoid(self.network(x))


class ContentEnsemble(nn.Module):
    def __init__(self, config, states):
        super().__init__()
        self.members = nn.ModuleList()
        for state in states:
            member = ContentRegressor(**config)
            member.load_state_dict(state)
            self.members.append(member)
        if not self.members:
            raise ValueError('An ensemble needs at least one trained member')

    def forward(self, features):
        return torch.stack([member(features) for member in self.members]).mean(0)


def load_content_model(path, device='cpu'):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact.get('format_version') != 1 or artifact.get('input_features') != 385:
        raise ValueError('Unsupported content checkpoint format')
    return ContentEnsemble(artifact['config'], artifact['states']).to(device).eval()
