"""Common training / held-out MSE evaluation for RGB, DINO boxes, and DINO+SAM."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.policies.grounded_libero_dataset import GroundedLIBERODataset
from src.policies.grounded_lc_model import make_policy
from src.policies.libero_images import IMAGE_TRANSFORM
from scripts.libero.encode_text_embeddings import encode_instructions


def forward_batch(model, batch, embeddings, representation, device):
    images, states = batch['image'].to(device), batch['robot_state'].to(device)
    language = torch.stack([embeddings[t] for t in batch['task_name']]).to(device)
    if representation == 'rgb':
        return model(images, states, language)
    return model(images, states, language, batch['grounding'].to(device))


def run_epoch(model, loader, embeddings, representation, device, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    total_error, count, per_action, by_task = 0., 0, None, {}
    with torch.set_grad_enabled(optimizer is not None):
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            predicted = forward_batch(model, batch, embeddings, representation, device)
            actions = batch['action'].to(device)
            error = (predicted - actions).square()
            loss = error.mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite BC loss')
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            detached = error.detach().cpu()
            total_error += detached.sum().item(); count += detached.numel()
            summed = detached.sum(0)
            per_action = summed if per_action is None else per_action + summed
            for task, row in zip(batch['task_name'], detached):
                record = by_task.setdefault(task, [0., 0])
                record[0] += row.sum().item(); record[1] += row.numel()
    if not count:
        raise ValueError('No samples were evaluated')
    return {'mse': total_error / count, 'samples': count // len(per_action),
            'per_action_mse': (per_action / (count // len(per_action))).tolist(),
            'per_task_mse': {k: v[0] / v[1] for k, v in by_task.items()}}


def selected_state_statistics(dataset):
    total, total_sq, count = None, None, 0
    # Statistics only from selected training demonstrations/frames, never validation/test.
    by_demo = {}
    for task, name, t in dataset.samples:
        by_demo.setdefault((task, name), []).append(t)
    for (task, name), frames in by_demo.items():
        with h5py.File(Path(dataset.dataset_dir) / f'{task}_demo.hdf5', 'r') as f:
            values = f[f'data/{name}/robot_states'][frames].astype(np.float64)
        if total is None:
            total, total_sq = np.zeros(values.shape[1]), np.zeros(values.shape[1])
        total += values.sum(0); total_sq += np.square(values).sum(0); count += len(values)
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(total_sq / count - mean ** 2, 0)), 1e-3)
    return mean, std


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    p.add_argument('--policy', choices=['rgb', 'boxes', 'masks'], default='masks')
    p.add_argument('--dataset-dir', default='data/libero/libero_object')
    p.add_argument('--splits', default='configs/libero_dataset_splits.json')
    p.add_argument('--cache-dir', default='data/libero/grounding')
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--train-fraction', type=float, default=1.)
    p.add_argument('--frame-limit', type=int)
    p.add_argument('--max-batches', type=int)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--run-name', default='libero_sam')
    p.add_argument('--checkpoint')
    p.add_argument('--embeddings', help='Optional precomputed task->CLIP tensor dictionary (.pt)')
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1 or (a.max_batches is not None and a.max_batches < 1):
        p.error('epochs/batch-size/max-batches must be positive')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    output = Path('results') / a.run_name; output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path('checkpoints') / a.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(a.splits).read_text())
    # Split manifest itself must not leak demonstrations across partitions.
    partitions = [set(splits[k]) for k in ('train', 'val', 'test')]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise ValueError('Overlapping trajectory splits')
    checkpoint = None
    if a.mode == 'evaluate':
        if not a.checkpoint:
            p.error('evaluate requires --checkpoint')
        checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        cfg = checkpoint['config']
        if cfg.get('image_transform') != IMAGE_TRANSFORM:
            raise ValueError('Checkpoint image transform mismatch; retrain with current preprocessing')
        a.policy, a.tasks = cfg['policy'], cfg['tasks']
        if checkpoint['splits'] != splits:
            raise ValueError('Evaluation split manifest differs from training')
    else:
        cfg = vars(a).copy()
    tasks = sorted({x.split('/')[0] for k in ('train','val','test') for x in splits[k]
                    if a.tasks is None or x.split('/')[0] in a.tasks})
    if checkpoint:
        embeddings = checkpoint['language_embeddings']
    elif a.embeddings:
        embeddings = torch.load(a.embeddings, map_location='cpu', weights_only=True)
    else:
        embeddings = dict(zip(tasks, encode_instructions(
            [GroundedLIBERODataset._get_instruction(t) for t in tasks], device).cpu()))
    if any(t not in embeddings or embeddings[t].shape != (512,) for t in tasks):
        raise ValueError('Every task requires a 512-dimensional CLIP embedding')
    def dataset(split):
        return GroundedLIBERODataset(a.dataset_dir, a.splits, split, a.policy, a.cache_dir,
                                    a.train_fraction, a.seed, tasks, a.frame_limit)
    if a.mode == 'evaluate':
        test = dataset('test')
        model = make_policy(a.policy, pretrained=False, **checkpoint['dimensions']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Reject perception settings changed between training and evaluation.
        for task, meta in test.cache_metadata.items():
            expected = checkpoint['perception_metadata'][task]
            keys = ['detector', 'prompts', 'image_transform', 'image_shape']
            if a.policy == 'masks':
                keys.append('sam')
            for key in keys:
                if meta[key] != expected[key]:
                    raise ValueError(f'Perception mismatch: {task}/{key}')
        metrics = run_epoch(model, DataLoader(test, batch_size=a.batch_size, num_workers=a.workers),
                            embeddings, a.policy, device, max_batches=a.max_batches)
        metrics.update(checkpoint=a.checkpoint, frame_limit=a.frame_limit, max_batches=a.max_batches,
                       evaluation='held_out_action_MSE_not_rollout_success')
        (output / 'test_results.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print(json.dumps(metrics, indent=2)); test.close(); return
    train, val = dataset('train'), dataset('val')
    sample = train[0]; train.close()
    dimensions = {'robot_state_dim': sample['robot_state'].numel(), 'action_dim': sample['action'].numel(), 'language_dim': 512}
    model = make_policy(a.policy, pretrained=not a.no_pretrained, **dimensions).to(device)
    mean, std = selected_state_statistics(train); model.set_state_normalization(mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    cfg.update(image_transform=IMAGE_TRANSFORM, tasks=tasks, train_samples=len(train), val_samples=len(val),
               smoke_test=a.frame_limit is not None or a.max_batches is not None,
               language_encoder='openai/clip-vit-base-patch32:projected_L2_normalized',
               normalization='selected_training_demonstrations_only')
    (output / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (output / 'selected_demos.json').write_text(json.dumps({'train':train.demos,'val':val.demos},indent=2)+'\n')
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                              generator=torch.Generator().manual_seed(a.seed))
    val_loader = DataLoader(val, batch_size=a.batch_size, num_workers=a.workers)
    history, best = [], float('inf')
    for epoch in range(a.epochs):
        training = run_epoch(model, train_loader, embeddings, a.policy, device, optimizer, a.max_batches)
        validation = run_epoch(model, val_loader, embeddings, a.policy, device, max_batches=a.max_batches)
        row = {'epoch':epoch+1,'train':training,'val':validation};history.append(row)
        print(json.dumps(row),flush=True)
        if validation['mse'] < best:
            best = validation['mse']
            torch.save({'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                        'language_embeddings':{t:e.cpu().clone() for t,e in embeddings.items()},
                        'config':cfg, 'dimensions':dimensions, 'splits':splits,
                        'perception_metadata':train.cache_metadata, 'epoch':epoch+1, 'val_mse':best}, checkpoint_dir / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history,indent=2)+'\n')
    train.close();val.close()


if __name__ == '__main__':
    main()
