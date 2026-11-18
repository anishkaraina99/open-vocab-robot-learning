"""Resumable cache, keyed by demonstration and original frame index."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .sam import CHANNELS, validate_boxes, save_overlay


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache(dataset, manifest_path, output, predictor, model_info, preview_dir=None):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('coordinate_format') != 'xyxy_pixels' or manifest.get('channels') != list(CHANNELS):
        raise ValueError('Manifest must specify xyxy_pixels and [target, destination] channels')
    if not manifest.get('box_source') or not manifest.get('demos'):
        raise ValueError('Manifest requires box_source and nonempty demos')
    metadata = json.dumps({'schema_version': 1, 'source': file_identity(dataset),
                           'manifest_sha256': sha256(manifest_path), 'model': model_info,
                           'channels': list(CHANNELS), 'box_source': manifest['box_source']}, sort_keys=True)
    if Path(dataset).resolve() == Path(output).resolve():
        raise ValueError('Cache must not overwrite source dataset')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
    completed = 0
    with h5py.File(dataset, 'r') as source, h5py.File(output, 'a') as cache:
        if 'metadata' in cache.attrs and cache.attrs['metadata'] != metadata:
            raise ValueError('Cache provenance differs; choose a new output file')
        cache.attrs['metadata'] = metadata
        for name, frames in manifest['demos'].items():
            images = source[f'data/{name}/images']
            n, h, w, _ = images.shape
            group = cache.require_group(f'data/{name}')
            masks = group.require_dataset('masks', (n, 2, h, w), dtype='u1', chunks=(1, 2, h, w), compression='gzip')
            boxes_out = group.require_dataset('boxes', (n, 2, 4), dtype='f4')
            scores_out = group.require_dataset('scores', (n, 2), dtype='f4')
            present_out = group.require_dataset('present', (n, 2), dtype='bool')
            done = group.require_dataset('complete', (n,), dtype='bool')
            for key, boxes in frames.items():
                t = int(key)
                if str(t) != key or not 0 <= t < n:
                    raise ValueError(f'Invalid frame index {name}/{key}')
                boxes, _ = validate_boxes(boxes, h, w)
                if done[t]:
                    continue
                image = images[t]
                result, scores, present = predictor(image, boxes)
                if result.shape != (2, h, w) or not np.isin(result, [0, 1]).all():
                    raise ValueError('Predictor returned invalid binary masks')
                masks[t], boxes_out[t], scores_out[t], present_out[t] = result, boxes, scores, present
                cache.flush()
                done[t] = True
                cache.flush()
                completed += 1
                if preview_dir:
                    save_overlay(image, result, boxes, Path(preview_dir) / f'{name}_{t:05d}.png')
                print(f'{name}/{t}: cached', flush=True)
    return completed
