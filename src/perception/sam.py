"""Frozen SAM with explicit, original-image XYXY pixel box coordinates."""
import numpy as np
import torch
from PIL import Image, ImageDraw

CHANNELS = ('target', 'destination')


def validate_boxes(boxes, height, width):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape != (2, 4) or not np.isfinite(boxes).all():
        raise ValueError('Expected finite (2, 4) XYXY pixel boxes: target, destination')
    present = np.any(boxes != 0, axis=1)
    for box in boxes[present]:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'Invalid box {box} for image {width}x{height}')
    return boxes, present


class BoxPromptedSAM:
    def __init__(self, checkpoint, model_type='vit_b', device='cpu'):
        from segment_anything import SamPredictor, sam_model_registry
        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device).eval().requires_grad_(False)
        self.predictor = SamPredictor(model)

    @torch.inference_mode()
    def __call__(self, image, boxes):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('SAM requires an HWC uint8 RGB image')
        h, w = image.shape[:2]
        boxes, present = validate_boxes(boxes, h, w)
        masks = np.zeros((2, h, w), dtype=np.uint8)
        scores = np.zeros(2, dtype=np.float32)
        if present.any():
            self.predictor.set_image(np.ascontiguousarray(image))
            for channel in np.flatnonzero(present):
                mask, score, _ = self.predictor.predict(box=boxes[channel], multimask_output=False)
                masks[channel] = mask[0]
                scores[channel] = score[0]
        return masks, scores, present


def save_overlay(image, masks, boxes, output):
    """Original RGB beside colored mask overlay. Green=target; blue=destination."""
    image = np.asarray(image)
    overlay = image.astype(np.float32).copy()
    colors = [(40, 230, 80), (50, 140, 255)]
    for mask, color in zip(masks, colors):
        pixels = mask.astype(bool)
        overlay[pixels] = .55 * overlay[pixels] + .45 * np.array(color)
    panel = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    for box, color in zip(boxes, colors):
        if np.any(box):
            draw.rectangle(tuple(box), outline=color, width=1)
    canvas = Image.new('RGB', (image.shape[1] * 2, image.shape[0]))
    canvas.paste(Image.fromarray(image), (0, 0))
    canvas.paste(panel, (image.shape[1], 0))
    canvas.save(output)
