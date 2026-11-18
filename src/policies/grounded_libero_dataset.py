"""Shared trajectory selection and validated perception joins for LIBERO."""
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from .dataset_libero import LIBERODataset
from .libero_images import IMAGE_TRANSFORM
from ..perception.mask_cache import file_identity
from ..perception.grounding_dino import box_features


def select_demos(splits, split, fraction=1., seed=42, tasks=None):
    if not 0 < fraction <= 1:
        raise ValueError('Training fraction must be in (0, 1]')
    selected = [x for x in splits[split] if tasks is None or x.split('/')[0] in tasks]
    if split == 'train' and fraction < 1:
        rng = np.random.default_rng(seed)
        result = []
        # Nested, task-stratified demo subsets for every policy and data budget.
        for task in sorted({x.split('/')[0] for x in selected}):
            demos = sorted(x for x in selected if x.split('/')[0] == task)
            order = rng.permutation(len(demos))
            result.extend(demos[i] for i in order[:max(1, int(len(demos) * fraction))])
        selected = result
    if not selected:
        raise ValueError(f'No demonstrations selected for {split}')
    return selected



class GroundedLIBERODataset(LIBERODataset):
    def __init__(self, dataset_dir, splits_path, split='train', representation='rgb',
                 cache_dir=None, fraction=1., seed=42, tasks=None, frame_limit=None):
        self.dataset_dir = str(dataset_dir)
        self._files, self._caches = {}, {}
        self._pid = os.getpid()
        self.representation = representation
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.samples = []
        self.demos = select_demos(json.loads(Path(splits_path).read_text()), split, fraction, seed, tasks)
        self.cache_metadata = {}
        if representation not in ('rgb', 'boxes', 'masks'):
            raise ValueError('Unknown representation')
        if representation != 'rgb' and cache_dir is None:
            raise ValueError('Grounded policies require a cache directory')
        if frame_limit is not None and frame_limit < 1:
            raise ValueError('frame_limit must be positive')
        for demo_path in self.demos:
            task, name = demo_path.split('/')
            path = Path(dataset_dir) / f'{task}_demo.hdf5'
            with h5py.File(path, 'r') as f:
                demo = f[f'data/{name}']
                n = len(demo['actions'])
                if frame_limit is not None:
                    n = min(n, frame_limit)
                if representation != 'rgb':
                    with h5py.File(self.cache_dir / f'{task}.hdf5', 'r') as cache:
                        meta = json.loads(cache.attrs['metadata'])
                        if meta.get('image_transform') != IMAGE_TRANSFORM:
                            raise ValueError('Perception image transform mismatch; regenerate the cache')
                        if meta['source'] != file_identity(path):
                            raise ValueError(f'Stale perception source for {task}')
                        if representation == 'masks':
                            meta['sam'] = json.loads(cache.attrs['sam'])
                        self.cache_metadata[task] = meta
                        key = 'masks_complete' if representation == 'masks' else 'boxes_complete'
                        if f'data/{name}/{key}' not in cache or not cache[f'data/{name}/{key}'][:n].all():
                            raise ValueError(f'Incomplete {representation} cache for {demo_path}')
                        images = demo['obs/agentview_rgb']
                        if tuple(meta['image_shape']) != images.shape[1:]:
                            raise ValueError('Perception/image shape mismatch')
            self.samples.extend((task, name, t) for t in range(n))

    def __getitem__(self, idx):
        if os.getpid() != self._pid:
            self.close(); self._pid = os.getpid()
        sample = super().__getitem__(idx)
        task, name, t = self.samples[idx]
        sample['frame_index'] = t
        if self.representation != 'rgb':
            if task not in self._caches:
                self._caches[task] = h5py.File(self.cache_dir / f'{task}.hdf5', 'r')
            group = self._caches[task][f'data/{name}']
            if self.representation == 'boxes':
                _, h, w = sample['image'].shape
                grounding = box_features(group['boxes'][t], h, w)
            else:
                grounding = np.array(group['masks'][t], dtype=np.float32)
            sample['grounding'] = torch.from_numpy(grounding)
        return sample

    def close(self):
        super().close()
        for f in getattr(self, '_caches', {}).values():
            f.close()
        self._caches = {}

    def __getstate__(self):
        state = super().__getstate__()
        state['_caches'] = {}
        return state
