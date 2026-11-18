import json
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
from src.policies.grounded_lc_model import make_policy
from src.policies.grounded_libero_dataset import GroundedLIBERODataset, select_demos
from src.perception.grounding_dino import box_features
from scripts.libero.cache_grounding import cache_task


class Detector:
    metadata = {'model_id': 'test', 'box_threshold': .3}
    def __call__(self, image, prompts):
        return np.array([[2, 3, 10, 12], [0, 0, 0, 0]], np.float32), np.array([.9, 0], np.float32)


class Segmenter:
    def __call__(self, image, boxes):
        masks = np.zeros((2, *image.shape[:2]), np.uint8)
        masks[0, 3:12, 2:10] = 1
        return masks, np.array([.9, 0]), np.array([True, False])


@pytest.fixture
def fixture(tmp_path):
    task = 'pick_up_the_soup_and_place_it_in_the_basket'
    path = tmp_path / f'{task}_demo.hdf5'
    with h5py.File(path, 'w') as f:
        for i in range(3):
            group = f.create_group(f'data/demo_{i}')
            group['obs/agentview_rgb'] = np.full((2, 32, 32, 3), 90 + i, np.uint8)
            group['robot_states'] = np.zeros((2, 9), np.float32)
            group['actions'] = np.zeros((2, 7), np.float32)
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({k:[f'{task}/demo_{i}'] for i,k in enumerate(['train','val','test'])}))
    cache = tmp_path / 'cache'
    cache_task(path, cache / f'{task}.hdf5', [f'demo_{i}' for i in range(3)],
               ['soup','basket'], Detector(), Segmenter(), {'checkpoint_sha256':'test'})
    return tmp_path, task, path, splits, cache


def test_normalized_box_features_and_missing_flag():
    features = box_features([[2,3,10,12],[0,0,0,0]], 20, 40)
    np.testing.assert_allclose(features, [.05,.15,.25,.6,1,0,0,0,0,0])


def test_nested_demo_budgets():
    splits = {'train':[f'{t}/demo_{i}' for t in ['a','b'] for i in range(40)]}
    small = set(select_demos(splits,'train',.1))
    medium = set(select_demos(splits,'train',.25))
    assert len(small) == 8 and len(medium) == 20 and small <= medium


def test_cache_join_and_resume(fixture):
    root, task, path, splits, cache = fixture
    result = cache_task(path, cache/f'{task}.hdf5', ['demo_0'], ['soup','basket'], Detector(), Segmenter(), {'checkpoint_sha256':'test'})
    assert result['boxes'] == result['masks'] == 0
    data = GroundedLIBERODataset(root,splits,representation='masks',cache_dir=cache)
    sample = data[0]
    assert sample['grounding'].shape == (2,32,32) and sample['grounding'][0,3,2] == 1
    assert sample['grounding'][1].sum() == 0
    assert data.__getstate__()['_caches'] == {}
    data.close()
    with h5py.File(cache/f'{task}.hdf5','a') as f:
        f['data/demo_0/masks_complete'][1] = False
    with pytest.raises(ValueError, match='Incomplete'):
        GroundedLIBERODataset(root,splits,representation='masks',cache_dir=cache)
    limited = GroundedLIBERODataset(root,splits,representation='masks',cache_dir=cache,frame_limit=1)
    assert len(limited) == 1
    limited.close()


@pytest.mark.parametrize('representation',['boxes','masks'])
def test_policy_gradients_and_language(representation):
    torch.set_num_threads(2)
    model = make_policy(representation, pretrained=False)
    images, state, language = torch.rand(2,3,32,32),torch.rand(2,9),torch.rand(2,512)
    grounding = torch.rand(2,10) if representation == 'boxes' else torch.rand(2,2,32,32)
    prediction = model(images,state,language,grounding)
    prediction.square().mean().backward()
    assert prediction.shape == (2,7)
    assert model.policy[0].weight.grad.abs().sum() > 0
    if representation == 'masks':
        assert model.mask_encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('representation',['rgb','boxes','masks'])
def test_train_and_evaluate_shared_pipeline(fixture, representation):
    root, task, path, splits, cache = fixture
    embeddings = root/'embeddings.pt'
    torch.save({task:torch.ones(512)/np.sqrt(512)},embeddings)
    script = Path(__file__).resolve().parents[1]/'scripts/libero/run_grounded.py'
    base = [sys.executable,str(script),'--dataset-dir',str(root),'--splits',str(splits),
            '--cache-dir',str(cache),'--batch-size','2','--device','cpu','--run-name',representation]
    result = subprocess.run(base+['--policy',representation,'--epochs','1','--no-pretrained','--embeddings',str(embeddings)],
                            cwd=root,capture_output=True,text=True,timeout=120)
    assert result.returncode == 0,result.stdout+result.stderr
    checkpoint = root/f'checkpoints/{representation}/best.pt'
    result = subprocess.run(base+['--mode','evaluate','--checkpoint',str(checkpoint)],
                            cwd=root,capture_output=True,text=True,timeout=120)
    assert result.returncode == 0,result.stdout+result.stderr
    report = json.loads((root/f'results/{representation}/test_results.json').read_text())
    assert report['samples'] == 2 and np.isfinite(report['mse'])


def test_cache_and_loader_share_rotated_image(tmp_path):
    task = 'pick_up_the_soup_and_place_it_in_the_basket'
    source = tmp_path / f'{task}_demo.hdf5'
    raw = np.zeros((32, 32, 3), dtype=np.uint8)
    raw[2:6, 3:9] = 255
    expected = np.rot90(raw, 2).copy()
    with h5py.File(source, 'w') as f:
        demo = f.create_group('data/demo_0')
        demo['obs/agentview_rgb'] = raw[None]
        demo['robot_states'] = np.zeros((1, 9), np.float32)
        demo['actions'] = np.zeros((1, 7), np.float32)
    class ImageDetector(Detector):
        def __call__(self, image, prompts):
            np.testing.assert_array_equal(image, expected)
            return super().__call__(image, prompts)
    class ImageSegmenter(Segmenter):
        def __call__(self, image, boxes):
            np.testing.assert_array_equal(image, expected)
            mask = (image[..., 0] > 0).astype(np.uint8)
            return np.stack([mask, np.zeros_like(mask)]), np.ones(2), np.ones(2, bool)
    cache = tmp_path / 'cache'
    cache_task(source, cache / f'{task}.hdf5', ['demo_0'], ['soup', 'basket'],
               ImageDetector(), ImageSegmenter(), {'checkpoint_sha256': 'test'})
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'train': [f'{task}/demo_0']}))
    data = GroundedLIBERODataset(tmp_path, splits, representation='masks', cache_dir=cache)
    sample = data[0]
    np.testing.assert_array_equal(sample['image'][0].numpy(), expected[..., 0] / 255)
    np.testing.assert_array_equal(sample['grounding'][0].numpy(), sample['image'][0].numpy())
    data.close()


def test_reject_old_cache_orientation(fixture):
    root, task, path, splits, cache = fixture
    with h5py.File(cache / f'{task}.hdf5', 'a') as f:
        meta = json.loads(f.attrs['metadata'])
        meta['image_transform'] = 'none:matches_baseline_stored_RGB'
        f.attrs['metadata'] = json.dumps(meta)
    with pytest.raises(ValueError, match='image transform mismatch'):
        GroundedLIBERODataset(root, splits, representation='masks', cache_dir=cache)
