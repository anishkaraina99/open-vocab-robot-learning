"""Download the official LIBERO-object files directly on the training machine."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/libero')
    p.add_argument('--tasks',nargs='+',help='Optional task stems, without _demo.hdf5')
    a=p.parse_args()
    patterns=[f'libero_object/{task}_demo.hdf5' for task in a.tasks] if a.tasks else ['libero_object/*_demo.hdf5']
    snapshot_download(repo_id='yifengzhu-hf/LIBERO-datasets',repo_type='dataset',
                      allow_patterns=patterns,local_dir=a.output,max_workers=2)
    print('Data:',Path(a.output).resolve()/'libero_object')


if __name__=='__main__':
    main()
