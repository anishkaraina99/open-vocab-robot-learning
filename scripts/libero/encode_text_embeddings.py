
import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

model_name = "openai/clip-vit-base-patch32"

tokenizer = CLIPTokenizer.from_pretrained(model_name)

text_encoder = CLIPTextModelWithProjection.from_pretrained(
    model_name
).to(device)

text_encoder.eval()

# Freeze CLIP parameters
for param in text_encoder.parameters():
    param.requires_grad = False


def encode_instructions(instructions):
    tokens = tokenizer(
        instructions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = text_encoder(**tokens)

        # Projected CLIP text embeddings
        embeddings = outputs.text_embeds

        # L2 normalization
        embeddings = embeddings / embeddings.norm(
            dim=-1, keepdim=True
        )

    return embeddings
