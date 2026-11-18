"""Download public PickPlaceCan demonstrations and SAM ViT-B without overwriting files."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

ASSETS = [
    ('https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/can/ph/low_dim_v15.hdf5',
     'data/robomimic/v1.5/can/ph/low_dim_v15.hdf5',
     '3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962'),
    ('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
     'checkpoints/sam/sam_vit_b_01ec64.pth',
     'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-only', action='store_true')
    a = p.parse_args()
    for url, filename, expected in ASSETS[:1] if a.dataset_only else ASSETS:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.partial')
            print(f'Downloading {url}', flush=True)
            urlretrieve(url, partial)
            if expected and digest(partial) != expected:
                raise ValueError(f'Checksum mismatch: {partial}')
            partial.replace(path)
        actual = digest(path)
        if expected and actual != expected:
            raise ValueError(f'Existing file checksum mismatch: {path}; remove it before retrying')
        print(f'{path}: SHA256 {actual}')


if __name__ == '__main__':
    main()
