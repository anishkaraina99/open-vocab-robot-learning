"""The baseline's frozen, normalized CLIP text embeddings, loaded lazily."""
import torch

MODEL_NAME = 'openai/clip-vit-base-patch32'
_ENCODERS = {}


def encode_instructions(instructions, device=None):
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    device = str(device or ('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu'))
    if device not in _ENCODERS:
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
        encoder = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(device)
        encoder.eval().requires_grad_(False)
        _ENCODERS[device] = (tokenizer, encoder)
    tokenizer, encoder = _ENCODERS[device]
    tokens = tokenizer(instructions, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.inference_mode():
        embeddings = encoder(**tokens).text_embeds
        return (embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clone()
