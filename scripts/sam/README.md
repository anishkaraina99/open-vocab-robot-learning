# SAM debugging tools: PickPlaceCan

Use these scripts to debug box-prompted segmentation and mask-conditioned behavior
cloning on PickPlaceCan. For the current LIBERO language-conditioned experiment,
follow [the GPU runbook](../../docs/SAM_GPU_RUNBOOK.md) and `scripts/libero/`.
The PickPlaceCan model also accepts an optional language embedding. DINO and SAM stay frozen; only the
BC policy (including its mask encoder and RGB backbone) is trained.

## Source packaging

Run `python scripts/sam/package_source.py` from the repository root to rebuild
`data/transfer/sam_project_source.tar.gz`. The archive includes source, configuration,
documentation, tests, and notebooks. Datasets, checkpoints, caches, and environments
are excluded.

## Repository baseline

Work starts from `origin/language-guidance` at `c39995b`, which contains `main`
through `3770aec`. That branch already adds a LIBERO loader, projected/normalized
512-dimensional CLIP embeddings, and a language-conditioned ResNet-18 policy.
There is no DINO implementation in that snapshot. The existing PickPlaceCan
baseline still uses a 16-dimensional robot state; LIBERO defaults to 9 dimensions.
Do not interchange their statistics or checkpoints.

## Local setup and downloaded data

From the repository root:

```bash
uv venv .venv-sam --python 3.10
uv pip install --python .venv-sam/bin/python -r requirements-sam.txt
.venv-sam/bin/python scripts/sam/download_assets.py
```

Assets are ignored by Git:

- `data/robomimic/v1.5/can/ph/low_dim_v15.hdf5`: official 200-demo proficient-human can dataset.
- `checkpoints/sam/sam_vit_b_01ec64.pth`: pretrained original SAM ViT-B.

Sources: [RoboMimic dataset](https://huggingface.co/datasets/robomimic/robomimic_datasets/blob/main/v1.5/can/ph/low_dim_v15.hdf5)
and [official SAM](https://github.com/facebookresearch/segment-anything).
The downloader verifies the dataset SHA256 and uses temporary downloads before
renaming. It does not overwrite existing files.

The low-dimensional file includes simulator states, actions, and robot observations;
it does not include the processed RGB images. Generate those with the team's script:

```bash
# Renderer dependencies, if not already installed in this environment:
uv pip install --python .venv-sam/bin/python robosuite==1.5.2 mujoco==3.3.7 'opencv-python<4.12'
.venv-sam/bin/python scripts/robomimic/generate_dataset.py
```

This writes `data/processed/can_rgb.hdf5`. The upstream generator overwrites its
output; do not run it concurrently with caching/training. macOS rendering needs
normal graphics access. The original `.venv` is preserved.

## 1. Preview real SAM on a saved image

```bash
.venv-sam/bin/python scripts/sam/preview.py \
  --image data/sam/frames/demo_0_00000.png \
  --boxes configs/sam/demo_0_frame_0_boxes.json
```

Output: `data/sam/preview/overlay.png`, separate binary PNG masks, `masks.npz`,
and `summary.json`. Left is RGB; right is the green target mask with its box.
Export demo_0 frame 0 with `export_frames.py` before running this command.
These example boxes were specified manually for that processed frame. They
are a SAM integration test, not DINO detections or experimental results.
The destination channel is deliberately zero: a correct destination-region prompt
has not yet been established. SAM quality scores are not measured ground-truth IoU.

## 2. Cache per-frame masks

To inspect an exact processed demonstration frame before annotating a debug box:

```bash
.venv-sam/bin/python scripts/sam/export_frames.py --demo demo_0 --frames 0 40 80
```


Supply per-frame detector outputs using this JSON contract:

```json
{
  "coordinate_format": "xyxy_pixels",
  "channels": ["target", "destination"],
  "box_source": "grounding_dino:record-model-and-prompt-settings-here",
  "demos": {
    "demo_0": {
      "0": [[32, 48, 41, 63], [0, 0, 0, 0]],
      "1": [[32, 48, 41, 63], [0, 0, 0, 0]]
    }
  }
}
```

The coordinates above illustrate the schema, not verified boxes for demo_0.
Use **original stored image pixel coordinates**: `[left, top, right, bottom]`.
Convert DINO's normalized center/width/height boxes before export if necessary.
The image origin is upper left. The team's processed images are already flipped
upright; do not flip them a second time. Channel order must always be target then
destination. `[0,0,0,0]` explicitly means a missing detection. Omitted frame keys
mean **not processed**, not a missing detection. Do not reuse a fixed target box
across a trajectory where the object moves.

```bash
.venv-sam/bin/python scripts/sam/cache_masks.py \
  --boxes data/processed/can_boxes.json \
  --output data/processed/can_sam.hdf5 \
  --device cpu
```

Use `--device cuda` on a GPU machine. For a small manifest, add
`--preview-dir data/sam/cache_previews` to inspect every processed frame.
SAM encodes each frame once, then predicts masks for the available boxes.

The compressed HDF5 cache stores masks, boxes, SAM quality scores, detection flags,
and completion flags indexed by demo/frame. It records source identity, manifest
hash, model type, and checkpoint hash. Rerunning the same command resumes completed
frames. Changed source data, boxes, or checkpoint require a new cache output.
Training rejects incomplete caches or mismatched source files. For training,
provide explicit boxes or explicit missing-detection zeros for every frame in
every selected demonstration. Never fill unprocessed frames with silent zeros.

### Reproduce the verified three-frame demonstration check

```bash
.venv-sam/bin/python scripts/sam/cache_masks.py \
  --boxes configs/sam/demo_0_debug_boxes.json \
  --output data/sam/demo_0_debug_masks.hdf5 \
  --preview-dir data/sam/demo_0_previews
.venv-sam/bin/python scripts/sam/check_cached_batch.py
```

This uses manually annotated target boxes on demo_0 frames 0, 40, and 80.
The destination channel is empty. Only these three frames are cached, so this
cache is deliberately rejected by the full training dataset loader. The batch
check performs one optimization step with randomly initialized RGB weights and
reports gradient flow; its MSE is not an evaluation result. Use a complete DINO
manifest before starting the full comparative experiment.

## 3. Train the mask policy using the shared BC trainer

After a complete cache is available:

```bash
.venv-sam/bin/python scripts/robomimic/train_bc.py \
  --mask-cache data/processed/can_sam.hdf5 \
  --run-name sam_can_debug --epochs 5 --batch-size 32
```

Without `--mask-cache`, the trainer retains the original RGB policy.
`--dataset`, `--splits`, and `--stats` can select explicit files. For an offline
wiring check, use `--no-pretrained --epochs 1 --max-batches 1 --device cpu` and
a separate run name; this is not a trained benchmark baseline.

`SAMBehaviorCloningPolicy` combines 512 ResNet features, 128 mask features, and
robot state, then uses the same two 256-unit MLP layers and Tanh actions as the
baseline. The mask CNN preserves a 4x4 spatial grid before its projection.
To instantiate the optional language-conditioned model:

```python
model = SAMBehaviorCloningPolicy(robot_state_dim=9, language_dim=512)
actions = model(images, robot_states, masks, language_embeddings)
```

The current PickPlaceCan trainer sets `language_dim=0`. The LIBERO path uses `GroundedLanguagePolicy`, `scripts/libero/cache_grounding.py`,
and `scripts/libero/run_grounded.py` for task-specific detections, masks, and
instruction embeddings. The SAM rollout evaluator is not included in this branch; implement and validate
it separately before measuring robot success.
During closed-loop evaluation, compute new DINO/SAM outputs from each current
camera image; cached expert-frame masks cannot be used as rollout observations.

## Checks

```bash
.venv-sam/bin/python -m pytest tests/test_sam_pipeline.py -q
```

Checks cover box validation, missing detections, cache resume and provenance,
incomplete-cache rejection, sample alignment, mask-encoder gradients, and required
language inputs. Unit cache tests use an explicit fake predictor; the preview and
recorded smoke runs use real pretrained SAM.

## Next shared decisions

- Use LIBERO-Object for the implemented language-conditioned comparison; confirm
  the final task and evaluation protocol before full training.
- Share DINO boxes with exact coordinate convention and target/destination order.
- Agree how to prompt destination regions and handle failed detections.
- Keep the same instructions, splits, training budgets, and evaluation states.
- Do not report manual-box debug runs as DINO+SAM or open-vocabulary results.

## Updated LIBERO integration

The language-conditioned DINO/SAM implementation now extends the shared RGB + language baseline.
See [the GPU runbook](../../docs/SAM_GPU_RUNBOOK.md) for the current experiment path.
