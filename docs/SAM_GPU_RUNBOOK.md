# Grounded behavior cloning on LIBERO: GPU runbook

## Scope and validation status

Use the shared `scripts/libero/run_grounded.py` runner to compare RGB + language,
DINO bounding boxes, and DINO + SAM masks. All variants retain RGB, robot state,
and instruction embeddings. The mask policy adds a trainable mask encoder.
Use committed demonstration splits and compute normalization from selected training
demonstrations only. Do not reuse PickPlaceCan statistics or checkpoints.

Local checks completed on November 17, 2026:

- Real official LIBERO alphabet-soup task downloaded (50 demonstrations).
- DINO + SAM generated target and destination outputs on six frames from three
  distinct demos: train demo_41, validation demo_0, test demo_8.
- Actual pretrained ResNet-18 and frozen CLIP used in one-step training and
  held-out-frame MSE checks for RGB, boxes, and masks.
- Fresh-image DINO/SAM inference matched cached outputs and predicted seven actions.
- Unit/integration tests exercise all three policies, cache joins/resume, nested
  demonstration budgets, gradients, checkpoint loading, and held-out MSE.

These are integration checks only. No full LIBERO training, closed-loop success
rate, or OOD-generalization result is claimed. The SAM closed-loop evaluator and
live-inference wrapper are not included in this branch. Local prototype checks
do not establish a supported rollout workflow. Current `.venv-sam`
contains RoboSuite 1.5.2 for the earlier PickPlaceCan work; upstream LIBERO depends
on RoboSuite 1.4.0 and must be run in a separate compatible environment.

## Configure GPU compute

Use a Linux NVIDIA GPU runtime with persistent storage. For Colab Enterprise:

1. Select a project linked to the intended education or paid billing account.
   Verify the credit balance and expiration separately from account status.
2. Confirm GPU quota before creating a runtime: at least one GPU globally and
   one GPU of the selected type in the runtime region.
3. Create a runtime template in the notebook's region. Select `g2-standard-4`
   with one NVIDIA L4 where available. Allocate at least 50 GB of data disk and
   enable idle shutdown with a 30-minute timeout.
4. Select a network and regional subnet. Enable public internet access for
   package/model downloads. Enable Private Google Access if required by the console.
5. Create and connect to a runtime from the template. Run `nvidia-smi` and verify
   the GPU before installing dependencies or processing the full dataset.
6. Save outputs to durable storage and explicitly stop the runtime after use.
   Idle shutdown and budget alerts do not impose a spending cap.

Start with batch size 32 and reduce it if GPU memory is insufficient. Measure the
small cache run before estimating full-job duration or cost. GPU approval does
not guarantee regional hardware availability. The notebook creates a separate
Python 3.10 environment; the notebook kernel can use the default Python version.

References:

- [Runtime templates](https://docs.cloud.google.com/colab/docs/create-runtime-template)
- [Compute quotas](https://docs.cloud.google.com/compute/resource-usage)
- [Idle shutdown](https://docs.cloud.google.com/colab/docs/idle-shutdown)
- [Grounding DINO API](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/grounding-dino)

## Put the current code on the GPU machine

Transfer the source revision used for the experiment, including uncommitted changes
if applicable. Build an upload archive with `python scripts/sam/package_source.py`.
Upload `data/transfer/sam_project_source.tar.gz` into `/content` in the notebook
Files pane. Set `ARCHIVE` to the uploaded file's absolute path. Re-import the
notebook after notebook changes; uploading source alone does not update its cells.
Keep datasets and pretrained weights out of Git; download them on the GPU runtime.

Run the following from the project root on a Linux GPU machine, in a fresh
Python 3.10 environment. CUDA-capable PyTorch must match the machine's driver.
The following pair matches the team's pinned baseline versions (CUDA 12.1 wheels):

```bash
python3.10 -m venv .venv-grounded
source .venv-grounded/bin/activate
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-sam.txt
python -m pip check
export HF_HOME="$PWD/checkpoints/huggingface"
export TORCH_HOME="$PWD/checkpoints/torch"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
python -m pytest tests/test_sam_pipeline.py tests/test_grounded_libero.py -q
```

The requirements pin PyTorch 2.2.2, torchvision 0.17.2, NumPy 1.26.4, h5py 3.14.0,
Pillow 12.3.0, and Transformers 4.46.3 to the shared baseline. Do not install the full simulator
requirements into the offline training environment. Earlier local CPU checks used
PyTorch 2.15.0 and torchvision 0.29.2; they do not validate the pinned GPU stack. Record `pip freeze` in the final results and use the same
versions for every compared policy. No simulator is needed for caching or offline BC.

## Download inputs and validate a small run

```bash
python scripts/libero/download_object.py
# SAM checkpoint only (avoid downloading unrelated PickPlaceCan data):
mkdir -p checkpoints/sam
curl -fL --retry 3 https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o checkpoints/sam/sam_vit_b_01ec64.pth
```

The official LIBERO-object HDF5 files total about 7.44 GB. Images already exist in
these files; do not use the PickPlaceCan renderer on LIBERO. The state definition is
`[gripper_qpos(2), eef_pos(3), eef_quaternion(4)]`. Rotate raw stored RGB by 180 degrees for all policies and perception. The shared
`policy_image` helper applies this convention; box/mask coordinates refer to the
rotated image. Regenerate caches made before this preprocessing change.

Repeat the small smoke test below using the current image orientation. Earlier
real-model smoke artifacts used unrotated images and are historical checks only:

```bash
python scripts/libero/cache_grounding.py \
  --splits configs/libero_smoke_splits.json --frame-limit 2 \
  --cache-dir data/libero/grounding_smoke --device cuda \
  --preview-dir data/libero/grounding_previews
python scripts/libero/run_grounded.py --policy masks \
  --splits configs/libero_smoke_splits.json --frame-limit 2 \
  --cache-dir data/libero/grounding_smoke --device cuda \
  --epochs 1 --batch-size 2 --run-name libero_masks_smoke
python scripts/libero/run_grounded.py --mode evaluate \
  --splits configs/libero_smoke_splits.json --frame-limit 2 \
  --cache-dir data/libero/grounding_smoke --device cuda \
  --checkpoint checkpoints/libero_masks_smoke/best.pt --run-name libero_masks_smoke
```

Inspect previews for correct target identity, not merely plausible-looking masks.
`configs/libero_grounding_prompts.json` supplies target/destination text per task.
DINO uses a separate query for each channel and selects its highest-scoring box.
Missing detections produce zero channels, never ground-truth simulator masks.
Tune prompts/thresholds using training or validation images only, then freeze
those choices before testing. Several groceries have similar shapes: successful
segmentation does not prove correct semantic selection.

## Cache the complete dataset once

```bash
python scripts/libero/cache_grounding.py --device cuda \
  --cache-dir data/libero/grounding
```

This computes DINO boxes and SAM masks, frozen, for the selected train/val/test
frames. All demonstrations are selected by default. The same cache supplies both
policies. You can first use `--representation boxes`, then rerun without that flag
to add masks while reusing the detections. Reruns skip completed work. Changed
prompts, thresholds, source files, or model revisions require a new cache directory.

Do not add `--frame-limit` to the full cache command. That option explicitly makes
a partial smoke-test cache. Training checks every selected frame is complete.
Copying a cache to another machine changes source identities: regenerate there
instead of silently associating masks with an unverified source file.

## Train the mask policy and evaluate offline

```bash
python scripts/libero/run_grounded.py --policy masks --epochs 5 \
  --device cuda --run-name libero_sam_full
python scripts/libero/run_grounded.py --mode evaluate --device cuda \
  --checkpoint checkpoints/libero_sam_full/best.pt --run-name libero_sam_full
```

Use `--policy rgb` or `--policy boxes` with distinct run names for matched comparisons. The common code preserves ResNet-18, frozen normalized 512-D CLIP,
two 256-unit action-head layers, MSE, the committed demo splits, and fixed
trajectory selection. RGB inputs are retained in both grounded policies.
Box features are two normalized XYXY coordinates plus presence flags (10 values).
The mask CNN encodes target/destination binary channels into 128 features while
retaining spatial information. Statistics use only the selected training data.

Checkpoints contain language embeddings, normalization, configuration, and
perception provenance. Offline evaluation reports overall, per-action, and per-task
MSE; these are not closed-loop success rates.

After the initial full-data experiment is reviewed, run the requested budgets:

```bash
bash scripts/libero/run_budgets.sh
# Compare all policy variants with the same settings:
POLICIES="rgb boxes masks" bash scripts/libero/run_budgets.sh
```

The budget script defaults to the mask policy. Subsets are nested, task-stratified, selected
at demonstration level, and identical across policies. Validation/test remain fixed.
The runner saves the selected demo IDs for audit. Do not infer OOD generalization
from these same-task splits; unseen-object/appearance/distractor conditions remain
separate experiments to design with the team.

## Closed-loop evaluation (not included)

This branch provides perception caching, offline behavior-cloning training, and
held-out action MSE evaluation. The SAM rollout evaluator and live-inference
wrapper are not included. No runnable SAM rollout command is provided here.

Add and validate a separate LIBERO-compatible rollout implementation before
reporting robot success rates. It must compute new DINO/SAM outputs from each
current camera observation, preserve the training image/state conventions, and
use the same horizon, seeds, and held-out initial states for all policy variants.
Do not substitute cached expert-frame masks for live observations.

## Regular Colab alternative

Import `notebooks/train_sam_colab.ipynb` at https://colab.research.google.com.
Select Runtime → Change runtime type → T4 GPU, connect, and run the GPU check.
Use the archive upload cell, install the isolated Python 3.10 environment, and
complete the six-frame check. Mount Drive and inspect saved previews before
setting `RUN_FULL = True` in a new cell. Then run the remaining stages individually.

This route uses standard hosted Colab resources rather than the project's
Colab Enterprise runtimes. The notebook stores datasets and caches locally for
performance, while full-training checkpoints and metrics are written to Drive.
A disconnected/deleted runtime loses local cache progress. Training currently
has no automatic optimizer-resume mode; preserve a run and select a new name
when restarting training. Check available Drive storage before starting.

The regular-Colab notebook has been syntax-checked locally, but must still be
validated by running its smoke test on a hosted GPU. Resource availability and
session duration are not guaranteed; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
