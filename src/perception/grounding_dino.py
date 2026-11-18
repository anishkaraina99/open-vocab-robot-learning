"""Frozen Grounding DINO with separate target/destination text queries."""
import numpy as np
import torch
from PIL import Image


class GroundingDINO:
    def __init__(self, model_id='IDEA-Research/grounding-dino-tiny', device='cpu',
                 box_threshold=.3, text_threshold=.25):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, disable_custom_kernels=True).to(device).eval().requires_grad_(False)
        self.device = device
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.metadata = {'model_id': model_id, 'revision': self.model.config._commit_hash,
                         'box_threshold': box_threshold, 'text_threshold': text_threshold,
                         'selection': 'highest_score_per_separate_query', 'channels': ['target', 'destination']}

    @torch.inference_mode()
    def __call__(self, image, prompts):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or len(prompts) != 2:
            raise ValueError('Expected uint8 HWC RGB image and two prompts')
        h, w = image.shape[:2]
        texts = [prompt.strip().lower().rstrip('.') + '.' for prompt in prompts]
        inputs = self.processor(images=[Image.fromarray(image)] * 2, text=texts,
                                padding=True, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(h, w)] * 2)
        boxes, scores = np.zeros((2, 4), np.float32), np.zeros(2, np.float32)
        for channel, result in enumerate(results):
            if len(result['scores']):
                index = int(result['scores'].argmax())
                box = result['boxes'][index].cpu().numpy()
                box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
                if box[2] > box[0] and box[3] > box[1]:
                    boxes[channel], scores[channel] = box, float(result['scores'][index])
        return boxes, scores


def box_features(boxes, height, width):
    from .sam import validate_boxes
    boxes, present = validate_boxes(boxes, height, width)
    normalized = boxes / np.array([width, height, width, height], np.float32)
    return np.concatenate([normalized, present[:, None]], axis=1).astype(np.float32).reshape(10)
