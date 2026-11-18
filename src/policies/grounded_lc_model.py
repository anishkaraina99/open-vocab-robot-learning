"""Bounding-box and mask variants of the RGB + language ResNet-18 baseline."""
import torch
from torch import nn
from .lc_model import LanguageConditionedPolicy


class GroundedLanguagePolicy(LanguageConditionedPolicy):
    def __init__(self, representation='masks', robot_state_dim=9, language_dim=512,
                 action_dim=7, pretrained=True):
        super().__init__(robot_state_dim, language_dim, action_dim, pretrained)
        if representation not in ('boxes', 'masks'):
            raise ValueError('representation must be boxes or masks')
        self.representation = representation
        self.language_dim = language_dim
        if representation == 'masks':
            self.mask_encoder = nn.Sequential(
                nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                nn.Linear(512, 128), nn.ReLU())
        extra_dim = 10 if representation == 'boxes' else 128
        self.policy = nn.Sequential(
            nn.Linear(512 + robot_state_dim + language_dim + extra_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())

    def forward(self, image, robot_state, language_embedding, grounding):
        if language_embedding.shape != (image.shape[0], self.language_dim):
            raise ValueError('Language embedding dimensions do not match the policy')
        if self.representation == 'boxes':
            if grounding.shape != (image.shape[0], 10):
                raise ValueError('Box features must be Bx10: two normalized XYXY+present boxes')
            extra = grounding
        else:
            if grounding.shape != (image.shape[0], 2, *image.shape[2:]):
                raise ValueError('Masks must be Bx2xHxW aligned with RGB')
            extra = self.mask_encoder(grounding.float())
        visual = self.encoder((image - self.image_mean) / self.image_std)
        state = (robot_state - self.state_mean) / self.state_std.clamp(min=1e-6)
        return self.policy(torch.cat([visual, state, language_embedding, extra], dim=1))


def make_policy(representation='rgb', pretrained=True, **dimensions):
    if representation == 'rgb':
        return LanguageConditionedPolicy(pretrained=pretrained, **dimensions)
    return GroundedLanguagePolicy(representation, pretrained=pretrained, **dimensions)
