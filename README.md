# open-vocab-robot-learning

DL Robotics Project

## Setup

Create a Python 3.10 environment and install dependencies:

```bash
conda create -n openvocab python=3.10
conda activate openvocab
pip install -r requirements.txt

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

Create a new Python environment and install LIBERO dependencies:

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
