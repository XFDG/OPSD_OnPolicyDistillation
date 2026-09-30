"""Validate optimized production checkpoints or explicitly selected stability checkpoints."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

import torch

from launcher_status import REPO, RUNTIME, selected_profile, source_manifest


def cosine_lr(step):
    return 1e-6 * (1.0 + math.cos(math.pi * step / 1739)) / 2.0


def checkpoint_evidence(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    assert re.fullmatch(r'global_step_\d+', checkpoint.name), checkpoint
    step = int(checkpoint.name.rsplit('_', 1)[1])
    assert 0 < step <= 1739, step
    config = json.loads((checkpoint / 'actor/fsdp_config.json').read_text())
    assert config['world_size'] == 8 and config['FSDP_version'] == 1, config
    ranks = []
    for rank in range(8):
        for kind in ('model', 'optim', 'extra_state'):
            path = checkpoint / 'actor' / f'{kind}_world_size_8_rank_{rank}.pt'
            assert path.is_file() and path.stat().st_size > 0, path
        extra = torch.load(checkpoint / 'actor' / f'extra_state_world_size_8_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        scheduler = extra['lr_scheduler']
        assert scheduler['last_epoch'] == step, (rank, scheduler)
        assert len(scheduler['_last_lr']) == 1
        assert math.isclose(float(scheduler['_last_lr'][0]), cosine_lr(step), rel_tol=1e-8, abs_tol=1e-14)
        assert scheduler['base_lrs'] == [1e-6], scheduler
        rng = extra['rng']
        assert {'cpu', 'cuda', 'numpy', 'random'} <= set(rng), (rank, rng.keys())
        for name in ('cpu', 'cuda'):
            assert isinstance(rng[name], torch.Tensor) and rng[name].dtype == torch.uint8 and rng[name].numel() > 0
        assert isinstance(rng['numpy'], tuple) and isinstance(rng['random'], tuple)
        ranks.append({'rank': rank, 'scheduler_step': step, 'next_lr': float(scheduler['_last_lr'][0]),
                      'rng_components': sorted(rng)})
    loader = torch.load(checkpoint / 'data.pt', map_location='cpu', weights_only=False)
    snapshot = loader['_snapshot']
    cursor = snapshot['_main_snapshot']['_sampler_iter_state']['samples_yielded']
    assert cursor == step * 8, (step, cursor)
    assert snapshot['_snapshot_step'] + loader['_steps_since_snapshot'] == step
    return {'checkpoint': str(checkpoint), 'global_step': step, 'world_size': 8,
            'ranks': ranks, 'samples_yielded': cursor, 'next_lr': cosine_lr(step), 'passed': True}


def verify_resume(checkpoint, run=None, stability=False):
    checkpoint = Path(checkpoint).resolve()
    assert checkpoint.is_relative_to(RUNTIME / 'runs'), checkpoint
    origin_run = checkpoint.parents[2]
    output_kind = checkpoint.parents[1].name
    allowed = {'stability-output', 'stability-resume-output'} if stability else {'train-output'}
    assert output_kind in allowed, f'Resume requires {allowed}, found {output_kind}'
    assert origin_run.parent == RUNTIME / 'runs' and origin_run.name.startswith('tip-'), origin_run
    report = checkpoint_evidence(checkpoint)
    assert report['global_step'] < 1739, 'Checkpoint is already complete'
    recorded = json.loads((origin_run / 'profile.json').read_text())
    assert recorded['profile'] == selected_profile(), 'Resume optimization profile differs'
    assert recorded['source_sha256'] == source_manifest(REPO), 'Resume source hashes differ'
    report['profile'] = recorded['profile']
    report['source_sha256'] = recorded['source_sha256']
    report['stability_checkpoint'] = stability
    if run is not None:
        (Path(run) / 'resume-source.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('run', nargs='?')
    parser.add_argument('--stability', action='store_true')
    args = parser.parse_args()
    report = verify_resume(args.checkpoint, args.run, args.stability)
    print('Optimized resume checkpoint: PASS', json.dumps(report))
