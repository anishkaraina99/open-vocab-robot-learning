
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BehaviorCloningPolicy(nn.Module):
    def __init__(
        self,
        robot_state_dim=16,
        action_dim=7,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()

        self.policy = nn.Sequential(
            nn.Linear(512 + robot_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        # ImageNet normalization for pretrained ResNet
        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

        # Defaults until training statistics are supplied
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

    def forward(self, image, robot_state):
        image = (
            image - self.image_mean
        ) / self.image_std

        robot_state = (
            robot_state - self.state_mean
        ) / self.state_std.clamp(min=1e-6)

        visual_features = self.encoder(image)

        combined = torch.cat(
            [visual_features, robot_state],
            dim=1,
        )

        return self.policy(combined)
