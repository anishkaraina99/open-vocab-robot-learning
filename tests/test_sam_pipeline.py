import json
import h5py
import numpy as np
import pytest
import torch
from src.perception.sam import BoxPromptedSAM, validate_boxes
from src.perception.mask_cache import build_cache
from src.policies.sam_dataset import SAMRobotDataset
from src.policies.sam_model import SAMBehaviorCloningPolicy


def test_boxes_reject_invalid_coordinates():
    boxes, present = validate_boxes([[2, 3, 10, 12], [0, 0, 0, 0]], 16, 16)
    assert present.tolist() == [True, False]
    for bad in ([[1, 2, 1, 8], [0]*4], [[0, 0, 17, 8], [0]*4], [[float('nan')]*4]*2):
        with pytest.raises(ValueError):
            validate_boxes(bad, 16, 16)


def test_missing_detection_skips_sam():
    sam = BoxPromptedSAM.__new__(BoxPromptedSAM)
    sam.predictor = None
    masks, scores, present = sam(np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((2, 4)))
    assert not masks.any() and not scores.any() and not present.any()


def test_cache_resume_alignment_and_rejection(tmp_path):
    source = tmp_path / 'rgb.hdf5'
    manifest = tmp_path / 'boxes.json'
    cache = tmp_path / 'masks.hdf5'
    splits = tmp_path / 'splits.json'
    with h5py.File(source, 'w') as f:
        g = f.create_group('data/demo_0')
        g['images'] = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        g['robot_state'] = np.zeros((2, 16), dtype=np.float32)
        g['actions'] = np.ones((2, 7), dtype=np.float32)
    splits.write_text(json.dumps({'train': ['demo_0']}))
    content = {'coordinate_format': 'xyxy_pixels', 'channels': ['target', 'destination'],
               'box_source': 'test', 'demos': {'demo_0': {'0': [[2, 3, 10, 12], [0]*4]}}}
    manifest.write_text(json.dumps(content))
    calls = []
    def predictor(image, boxes):
        calls.append(1)
        masks = np.zeros((2, 16, 16), dtype=np.uint8)
        masks[0, 3:12, 2:10] = 1
        return masks, np.array([.9, 0]), np.array([True, False])
    assert build_cache(source, manifest, cache, predictor, {'model': 'test'}) == 1
    assert build_cache(source, manifest, cache, predictor, {'model': 'test'}) == 0
    assert len(calls) == 1
    with pytest.raises(ValueError, match='Incomplete'):
        SAMRobotDataset(source, splits, cache)
    content['demos']['demo_0']['1'] = [[2, 3, 10, 12], [0]*4]
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='provenance'):
        build_cache(source, manifest, cache, predictor, {'model': 'test'})
    complete = tmp_path / 'complete.hdf5'
    build_cache(source, manifest, complete, predictor, {'model': 'test'})
    dataset = SAMRobotDataset(source, splits, complete)
    assert dataset[1]['masks'][0, 3, 2] == 1
    assert dataset[1]['masks'][1].sum() == 0
    assert dataset[1]['action'].tolist() == [1] * 7
    assert dataset.__getstate__()['_cache'] is None
    dataset.close()
    with h5py.File(source, 'a') as f:
        f.attrs['changed'] = True
    with pytest.raises(ValueError, match='changed dataset'):
        SAMRobotDataset(source, splits, complete)


@pytest.mark.parametrize('language_dim', [0, 512])
def test_policy_learns_mask_features_and_requires_language(language_dim):
    torch.manual_seed(5)
    torch.set_num_threads(2)
    model = SAMBehaviorCloningPolicy(pretrained=False, language_dim=language_dim)
    image = torch.rand(2, 3, 32, 32)
    state = torch.rand(2, 16)
    masks = torch.randint(0, 2, (2, 2, 32, 32)).float()
    language = torch.rand(2, language_dim) if language_dim else None
    action = model(image, state, masks, language)
    assert action.shape == (2, 7) and torch.isfinite(action).all()
    action.square().mean().backward()
    assert model.mask_encoder[0].weight.grad.abs().sum() > 0
    if language_dim:
        with pytest.raises(ValueError, match='language embedding'):
            model(image, state, masks)


def test_shared_trainer_writes_sam_checkpoint(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    source = tmp_path / 'rgb.hdf5'
    boxes = tmp_path / 'boxes.json'
    cache = tmp_path / 'masks.hdf5'
    splits = tmp_path / 'splits.json'
    stats = tmp_path / 'stats.json'
    with h5py.File(source, 'w') as f:
        for name in ('demo_train', 'demo_val'):
            g = f.create_group(f'data/{name}')
            g['images'] = np.zeros((2, 32, 32, 3), dtype=np.uint8)
            g['robot_state'] = np.zeros((2, 16), dtype=np.float32)
            g['actions'] = np.zeros((2, 7), dtype=np.float32)
    splits.write_text(json.dumps({'train': ['demo_train'], 'val': ['demo_val']}))
    stats.write_text(json.dumps({'mean': [0] * 16, 'std': [1] * 16}))
    boxes.write_text(json.dumps({'coordinate_format': 'xyxy_pixels', 'channels': ['target', 'destination'],
        'box_source': 'test', 'demos': {name: {str(t): [[0]*4]*2 for t in range(2)}
                                     for name in ('demo_train', 'demo_val')}}))
    def predictor(image, boxes):
        return np.zeros((2, 32, 32), dtype=np.uint8), np.zeros(2), np.zeros(2, dtype=bool)
    build_cache(source, boxes, cache, predictor, {'model': 'test'})
    script = Path(__file__).resolve().parents[1] / 'scripts/robomimic/train_bc.py'
    result = subprocess.run([sys.executable, str(script), '--dataset', str(source),
        '--splits', str(splits), '--stats', str(stats), '--mask-cache', str(cache),
        '--no-pretrained', '--device', 'cpu', '--epochs', '1', '--batch-size', '2',
        '--max-batches', '1', '--run-name', 'sam_test'], cwd=tmp_path,
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(tmp_path / 'checkpoints/sam_test/best.pt', weights_only=True)
    assert checkpoint['config']['policy_type'] == 'sam'
    assert 'mask_encoder.0.weight' in checkpoint['model_state_dict']
    assert np.isfinite(checkpoint['val_mse'])
