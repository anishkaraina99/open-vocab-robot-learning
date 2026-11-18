"""One optimization step on selected cached frames; a wiring check, not evaluation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import torch
from src.perception.mask_cache import file_identity
from src.policies.sam_model import SAMBehaviorCloningPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='data/processed/can_rgb.hdf5')
    p.add_argument('--cache', default='data/sam/demo_0_debug_masks.hdf5')
    p.add_argument('--demo', default='demo_0')
    p.add_argument('--frames', type=int, nargs='+', default=[0, 40, 80])
    p.add_argument('--stats', default='configs/robot_state_stats.json')
    p.add_argument('--output', default='data/sam/batch_check.json')
    a = p.parse_args()
    if len(a.frames) < 2:
        p.error('Use at least two frames for the training-mode BatchNorm check')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    with h5py.File(a.dataset, 'r') as source, h5py.File(a.cache, 'r') as cache:
        if json.loads(cache.attrs['metadata'])['source'] != file_identity(a.dataset):
            raise ValueError('Cache source mismatch')
        demo = source[f'data/{a.demo}']; cached = cache[f'data/{a.demo}']
        if not all(cached['complete'][t] for t in a.frames):
            raise ValueError('Selected frames have not all been cached')
        def tensor(dataset):
            return torch.from_numpy(np.stack([dataset[t] for t in a.frames])).float()
        images = tensor(demo['images']).permute(0, 3, 1, 2) / 255
        states, actions, masks = tensor(demo['robot_state']), tensor(demo['actions']), tensor(cached['masks'])
    model = SAMBehaviorCloningPolicy(robot_state_dim=states.shape[1], action_dim=actions.shape[1], pretrained=False)
    stats = json.loads(Path(a.stats).read_text())
    model.set_state_normalization(stats['mean'], stats['std'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    predicted = model(images, states, masks)
    loss = torch.nn.functional.mse_loss(predicted, actions)
    loss.backward()
    gradient_norm = model.mask_encoder[0].weight.grad.norm().item()
    if not torch.isfinite(loss) or gradient_norm <= 0:
        raise RuntimeError('Nonfinite loss or missing mask gradients')
    optimizer.step()
    result = {'purpose': 'integration_smoke_test_not_benchmark', 'pretrained_rgb': False,
              'demo': a.demo, 'frames': a.frames, 'mask_shape': list(masks.shape),
              'action_shape': list(predicted.shape), 'loss_before_single_step': loss.item(),
              'mask_gradient_norm': gradient_norm, 'optimizer_step_completed': True}
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
