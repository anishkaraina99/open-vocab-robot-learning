import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.perception.sam import BoxPromptedSAM
from src.perception.mask_cache import build_cache, sha256


def main():
    p = argparse.ArgumentParser(description='Cache frozen SAM outputs for explicit per-frame boxes')
    p.add_argument('--dataset', default='data/processed/can_rgb.hdf5')
    p.add_argument('--boxes', required=True)
    p.add_argument('--output', default='data/processed/can_sam.hdf5')
    p.add_argument('--checkpoint', default='checkpoints/sam/sam_vit_b_01ec64.pth')
    p.add_argument('--model-type', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--preview-dir')
    a = p.parse_args()
    model_info = {'model_type': a.model_type, 'checkpoint_sha256': sha256(a.checkpoint)}
    predictor = BoxPromptedSAM(a.checkpoint, a.model_type, a.device)
    count = build_cache(a.dataset, a.boxes, a.output, predictor, model_info, a.preview_dir)
    print(f'Cached {count} new frames')


if __name__ == '__main__':
    main()
