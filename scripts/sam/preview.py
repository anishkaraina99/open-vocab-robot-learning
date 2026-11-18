"""Run real SAM on an RGB image with two manually supplied example boxes."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from src.perception.sam import BoxPromptedSAM, save_overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--boxes', required=True, help='JSON file: [[x0,y0,x1,y1], [x0,y0,x1,y1]]; zeros = missing')
    p.add_argument('--checkpoint', default='checkpoints/sam/sam_vit_b_01ec64.pth')
    p.add_argument('--model-type', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='data/sam/preview')
    a = p.parse_args()
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    image = np.array(Image.open(a.image).convert('RGB'))
    boxes = np.asarray(json.loads(Path(a.boxes).read_text()), dtype=np.float32)
    sam = BoxPromptedSAM(a.checkpoint, a.model_type, a.device)
    masks, scores, present = sam(image, boxes)
    np.savez_compressed(output / 'masks.npz', masks=masks, boxes=boxes, scores=scores, present=present)
    save_overlay(image, masks, boxes, output / 'overlay.png')
    for channel, mask in zip(('target', 'destination'), masks):
        Image.fromarray(mask * 255).save(output / f'{channel}.png')
    summary = {'image': a.image, 'box_source': 'manual_debug_only', 'scores': scores.tolist(),
               'present': present.tolist(), 'mask_pixels': masks.sum(axis=(1, 2)).tolist()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
