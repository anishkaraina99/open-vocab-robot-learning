
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class LanguageConditionedPolicy(nn.Module):
    def __init__(
        self,
        robot_state_dim=9,
        language_dim=512,
        action_dim=7,
        pretrained=True,
    ):
        super().__init__()

        # Visual encoder
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()

        # Combined features:
        # 512 visual + 9 robot state + 512 language = 1033
        self.policy = nn.Sequential(
            nn.Linear(512 + robot_state_dim + language_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        # ImageNet normalization
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # Robot state normalization
        self.register_buffer(
            "state_mean",
            torch.zeros(robot_state_dim),
        )

        self.register_buffer(
            "state_std",
            torch.ones(robot_state_dim),
        )

    def set_state_normalization(self, mean, std):
        self.state_mean.copy_(
            torch.as_tensor(mean, dtype=torch.float32)
        )
        self.state_std.copy_(
            torch.as_tensor(std, dtype=torch.float32)
        )

    def forward(self, image, robot_state, language_embedding):
        # Normalize inputs
        image = (image - self.image_mean) / self.image_std

        robot_state = (
            robot_state - self.state_mean
        ) / self.state_std.clamp(min=1e-6)

        # Extract visual features
        visual_features = self.encoder(image)

        # Concatenate all modalities
        combined = torch.cat(
            [visual_features, robot_state, language_embedding],
            dim=1,
        )

        return self.policy(combined)
