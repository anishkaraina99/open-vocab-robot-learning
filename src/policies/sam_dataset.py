"""Processed PickPlaceCan observations joined to a validated mask cache."""
import json
import h5py
import numpy as np
import torch
from .dataset import RobotDataset
from ..perception.mask_cache import file_identity


class SAMRobotDataset(RobotDataset):
    def __init__(self, hdf5_path, splits_path, cache_path, split='train'):
        super().__init__(hdf5_path, splits_path, split)
        self.cache_path = str(cache_path)
        self._cache = None
        with h5py.File(cache_path, 'r') as cache, h5py.File(hdf5_path, 'r') as source:
            metadata = json.loads(cache.attrs['metadata'])
            if metadata['source'] != file_identity(hdf5_path):
                raise ValueError('Mask cache was generated from a different or changed dataset')
            if metadata['channels'] != ['target', 'destination']:
                raise ValueError('Mask channel order mismatch')
            for name in sorted({name for name, _ in self.samples}):
                path = f'data/{name}'
                if path not in cache or not cache[path]['complete'][:].all():
                    raise ValueError(f'Incomplete mask cache for {name}; generate all frames before training')
                images = source[path]['images']
                n, h, w, _ = images.shape
                if cache[path]['masks'].shape != (n, 2, h, w):
                    raise ValueError(f'Mask/image shape mismatch for {name}')

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self._cache is None:
            self._cache = h5py.File(self.cache_path, 'r')
        name, t = self.samples[idx]
        sample['masks'] = torch.from_numpy(np.array(self._cache[f'data/{name}/masks'][t], copy=True)).float()
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = state['_cache'] = None
        return state

    def close(self):
        for attr in ('_file', '_cache'):
            handle = getattr(self, attr, None)
            if handle is not None:
                handle.close()
                setattr(self, attr, None)

    def __del__(self):
        self.close()
