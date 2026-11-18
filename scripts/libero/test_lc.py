import sys
import torch

sys.path.insert(0, "src/policies")

from lc_model import LanguageConditionedPolicy
from encode_text_embeddings import encode_instructions

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

instructions = [
    "Pick up the butter and place it in the basket.",
    "Pick up the milk and place it in the basket."
]

embeddings = encode_instructions(instructions)
print("CLIP embeddings:", embeddings.shape)

model = LanguageConditionedPolicy(
    robot_state_dim=9,
    language_dim=512,
    action_dim=7,
    pretrained=False
).to(device)

images = torch.rand(2, 3, 128, 128).to(device)
robot_states = torch.rand(2, 9).to(device)
embeddings = embeddings.to(device)

with torch.no_grad():
    predictions = model(images, robot_states, embeddings)

print("Predictions:", predictions.shape)
print("Predictions finite:", torch.isfinite(predictions).all().item())
print("Device:", device)