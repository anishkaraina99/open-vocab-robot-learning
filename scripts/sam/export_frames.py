"""Export exact processed frames for box annotation; no resizing or flipping."""
import argparse
import json
from pathlib import Path
import h5py
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='data/processed/can_rgb.hdf5')
    p.add_argument('--demo', default='demo_0')
    p.add_argument('--frames', type=int, nargs='+', default=[0])
    p.add_argument('--output', default='data/sam/frames')
    a = p.parse_args()
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.dataset, 'r') as f:
        images = f[f'data/{a.demo}/images']
        for t in a.frames:
            if not 0 <= t < len(images):
                raise ValueError(f'Frame {t} outside {a.demo}: 0..{len(images)-1}')
            path = output / f'{a.demo}_{t:05d}.png'
            Image.fromarray(images[t]).save(path)
            print(path)
    (output / f'{a.demo}_source.json').write_text(json.dumps({
        'dataset': str(Path(a.dataset).resolve()), 'demo': a.demo,
        'frames': a.frames, 'transform': 'none'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
