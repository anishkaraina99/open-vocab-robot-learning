"""RGB + two mask channels + state, with an optional shared language embedding."""
import torch
from torch import nn
from .bc_model import BehaviorCloningPolicy


class SAMBehaviorCloningPolicy(BehaviorCloningPolicy):
    def __init__(self, robot_state_dim=16, action_dim=7, pretrained=True,
                 language_dim=0, mask_feature_dim=128):
        super().__init__(robot_state_dim, action_dim, pretrained)
        self.language_dim = language_dim
        # Retain a spatial grid: global pooling would discard object location.
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 4 * 4, mask_feature_dim), nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(512 + robot_state_dim + language_dim + mask_feature_dim, 256),
            nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, image, robot_state, masks, language_embedding=None):
        if masks.ndim != 4 or masks.shape[:2] != (image.shape[0], 2) or masks.shape[2:] != image.shape[2:]:
            raise ValueError('Masks must be Bx2xHxW aligned with RGB')
        features = [self.encoder((image - self.image_mean) / self.image_std),
                    (robot_state - self.state_mean) / self.state_std.clamp(min=1e-6),
                    self.mask_encoder(masks.float())]
        if self.language_dim:
            if language_embedding is None or language_embedding.shape != (image.shape[0], self.language_dim):
                raise ValueError('Supply the configured language embedding for every sample')
            features.append(language_embedding)
        elif language_embedding is not None:
            raise ValueError('Set language_dim when constructing a language-conditioned model')
        return self.policy(torch.cat(features, dim=1))
