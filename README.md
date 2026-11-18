# open-vocab-robot-learning

DL Robotics Project

## Setup

Create a Python 3.10 environment and install dependencies:

```bash
conda create -n openvocab python=3.10
conda activate openvocab
pip install -r requirements.txt
```

Install LIBERO dependencies in your main environment:

```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

cd LIBERO
python -m pip install -e . --no-deps
```

### Installation Notes

The versions of `numba` and `llvmlite` specified in `requirements.txt` may not have prebuilt wheels available for all platforms.

For Intel-based macOS systems (`x86_64`), if installation of the specified versions fails, try installing compatible older versions:

```bash
pip install numba==0.62.1 llvmlite==0.45.1
```

## Test RoboSuite

python scripts/test_pickplace.py

## Download LIBERO

Create a separate Python environment and install LIBERO dependencies:

```bash
conda create -n libero python=3.8.13
conda activate libero

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

pip install -r requirements.txt
pip install -e .
```

Download libero_object dataset:

```bash
python benchmark_scripts/download_libero_datasets.py --datasets libero_object --use-huggingface
```

## Grounded language-conditioned policies

Compare three behavior-cloning policies on LIBERO-Object using shared instructions,
demonstration splits, normalization rules, and evaluation metrics:

| Policy | Inputs |
| --- | --- |
| RGB + language | RGB, robot state, CLIP instruction embedding |
| DINO bounding boxes | Baseline inputs plus target/destination box features |
| DINO + SAM masks | Baseline inputs plus encoded target/destination masks |

Grounding DINO, SAM, and CLIP remain frozen. Train the RGB encoder, action head,
and mask encoder where present. Both grounded policies retain the RGB input.

Follow the [GPU runbook](docs/SAM_GPU_RUNBOOK.md) for setup, perception caching,
training and offline evaluation. The SAM rollout evaluator is not included in this branch. Use the
[Colab Enterprise notebook](notebooks/train_sam_google_cloud.ipynb) for interactive
execution. Run the small integration check before full-dataset processing.

Use the [PickPlaceCan SAM debugging tools](scripts/sam/README.md) for individual
image previews and cache checks. Keep PickPlaceCan results separate from LIBERO.
Recorded LIBERO smoke runs verify software integration only; full training and
closed-loop simulator validation remain outstanding.

## Regular Google Colab

Use [the regular Colab notebook](notebooks/train_sam_colab.ipynb) at
https://colab.research.google.com with a hosted NVIDIA GPU runtime. No Google Cloud
project or runtime template is needed. Build the upload archive with
`python scripts/sam/package_source.py`, upload it when prompted, and run the
six-frame smoke test before enabling `RUN_FULL`. Mount Drive to retain results
and best checkpoints. Datasets and perception caches remain on temporary local
storage; recreate them after runtime deletion. GPU availability is not guaranteed.
