"""Generate reusable DINO boxes and optional SAM masks on stored LIBERO RGB frames."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from src.perception.grounding_dino import GroundingDINO
from src.perception.sam import BoxPromptedSAM, save_overlay
from src.perception.mask_cache import file_identity, sha256
from src.policies.grounded_libero_dataset import select_demos
from src.policies.libero_images import IMAGE_TRANSFORM, policy_image


def cache_task(source_path, output, names, prompts, detector, sam=None,
               sam_info=None, frame_limit=None, preview_dir=None):
    with h5py.File(source_path, 'r') as source:
        shape = list(source[f'data/{names[0]}/obs/agentview_rgb'].shape[1:])
    metadata = {'schema': 1, 'source': file_identity(source_path), 'image_shape': shape,
                'image_transform': IMAGE_TRANSFORM,
                'detector': detector.metadata, 'prompts': prompts,
                'channels': ['target', 'destination']}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
    counts = {'boxes': 0, 'masks': 0, 'missing_target': 0, 'missing_destination': 0}
    with h5py.File(source_path, 'r') as source, h5py.File(output, 'a') as cache:
        encoded = json.dumps(metadata, sort_keys=True)
        if 'metadata' in cache.attrs and cache.attrs['metadata'] != encoded:
            raise ValueError('Perception settings/source changed: use a new cache directory')
        cache.attrs['metadata'] = encoded
        if sam is not None:
            encoded_sam = json.dumps(sam_info, sort_keys=True)
            if 'sam' in cache.attrs and cache.attrs['sam'] != encoded_sam:
                raise ValueError('SAM checkpoint changed: use a new cache directory')
            cache.attrs['sam'] = encoded_sam
        for name in names:
            images = source[f'data/{name}/obs/agentview_rgb']
            n, h, w, _ = images.shape
            group = cache.require_group(f'data/{name}')
            boxes = group.require_dataset('boxes', (n, 2, 4), dtype='f4')
            scores = group.require_dataset('dino_scores', (n, 2), dtype='f4')
            box_done = group.require_dataset('boxes_complete', (n,), dtype='bool')
            if sam is not None:
                masks = group.require_dataset('masks', (n, 2, h, w), dtype='u1', compression='gzip', chunks=(1, 2, h, w))
                mask_scores = group.require_dataset('sam_scores', (n, 2), dtype='f4')
                mask_done = group.require_dataset('masks_complete', (n,), dtype='bool')
            for t in range(min(n, frame_limit) if frame_limit else n):
                if box_done[t] and (sam is None or mask_done[t]):
                    continue
                image = policy_image(images[t])
                if not box_done[t]:
                    box, confidence = detector(image, prompts)
                    boxes[t], scores[t] = box, confidence
                    cache.flush(); box_done[t] = True; cache.flush()
                    counts['boxes'] += 1
                box = boxes[t]
                missing = ~np.any(box != 0, axis=1)
                counts['missing_target'] += int(missing[0])
                counts['missing_destination'] += int(missing[1])
                if sam is not None and not mask_done[t]:
                    mask, confidence, _ = sam(image, box)
                    masks[t], mask_scores[t] = mask, confidence
                    cache.flush(); mask_done[t] = True; cache.flush()
                    counts['masks'] += 1
                if preview_dir and t < 3:
                    mask = masks[t] if sam is not None else np.zeros((2, h, w), np.uint8)
                    save_overlay(image, mask, box, Path(preview_dir) / f'{source_path.stem}_{name}_{t}.png')
                if t % 50 == 0:
                    print(f'{source_path.stem}/{name}/{t}: {counts}', flush=True)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', default='data/libero/libero_object')
    p.add_argument('--splits', default='configs/libero_dataset_splits.json')
    p.add_argument('--prompts', default='configs/libero_grounding_prompts.json')
    p.add_argument('--cache-dir', default='data/libero/grounding')
    p.add_argument('--representation', choices=['boxes', 'masks'], default='masks')
    p.add_argument('--split', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--train-fraction', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--frame-limit', type=int, help='Explicit smoke-test limit per demonstration')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dino-model', default='IDEA-Research/grounding-dino-tiny')
    p.add_argument('--box-threshold', type=float, default=.3)
    p.add_argument('--text-threshold', type=float, default=.25)
    p.add_argument('--sam-checkpoint', default='checkpoints/sam/sam_vit_b_01ec64.pth')
    p.add_argument('--sam-model', default='vit_b')
    p.add_argument('--preview-dir')
    a = p.parse_args()
    if a.frame_limit is not None and a.frame_limit < 1:
        p.error('frame-limit must be positive')
    splits = json.loads(Path(a.splits).read_text())
    prompts = json.loads(Path(a.prompts).read_text())
    by_task = {}
    for split in a.split:
        for item in select_demos(splits, split, a.train_fraction, a.seed, a.tasks):
            task, name = item.split('/'); by_task.setdefault(task, set()).add(name)
    detector = GroundingDINO(a.dino_model, a.device, a.box_threshold, a.text_threshold)
    sam = BoxPromptedSAM(a.sam_checkpoint, a.sam_model, a.device) if a.representation == 'masks' else None
    sam_info = {'model': a.sam_model, 'checkpoint_sha256': sha256(a.sam_checkpoint)} if sam else None
    for task, names in sorted(by_task.items()):
        result = cache_task(Path(a.dataset_dir) / f'{task}_demo.hdf5',
                            Path(a.cache_dir) / f'{task}.hdf5', sorted(names),
                            prompts[task], detector, sam, sam_info, a.frame_limit, a.preview_dir)
        print(task, result, flush=True)


if __name__ == '__main__':
    main()
