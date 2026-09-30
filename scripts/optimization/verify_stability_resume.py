"""Verify two bounded resumed updates with original dataset/schedule continuity."""
import json
import math
from pathlib import Path
import sys

from verify_resume import verify_resume
from verify_stability import check_metrics, check_original_dataset, check_output, check_profile, read_metrics


def verify_stability_resume(run, source):
    run, source = Path(run), Path(source)
    profile = check_profile(run)
    initial = verify_resume(source, stability=True)
    start = initial['global_step']
    log, steps = read_metrics(run, 'stability-resume')
    assert f'Setting global step to {start}' in log
    assert 'Total training steps: 1739' in log
    assert '=== OPD training completed ===' in log
    check_original_dataset(log)
    for rank in range(8):
        for kind, label in [('model', 'model'), ('optim', 'optimizer'), ('extra_state', 'rng')]:
            name = source / 'actor' / f'{kind}_world_size_8_rank_{rank}.pt'
            assert f'Loaded {label} from {name}' in log, (rank, label, name)
        extra = source / 'actor' / f'extra_state_world_size_8_rank_{rank}.pt'
        assert f'Loaded lr_scheduler from {extra}' in log, (rank, 'scheduler')
    check_metrics(steps, (start + 1, start + 2))
    assert math.isclose(steps[start + 1]['opd/lr'], initial['next_lr'], rel_tol=1e-8, abs_tol=1e-14)
    out, checkpoints = check_output(run, 'stability-resume-output', (start + 1, start + 2), start + 2)
    report = {'passed': True, 'source': initial, 'profile': profile, 'steps': steps,
              'checkpoint_evidence': checkpoints, 'output': str(out), 'validation_sequences': 32,
              'dataset_rows': 13918, 'scheduler_total_training_steps': 1739,
              'scope': 'real-model eight-rank checkpoint restoration and two updates; not full-epoch accuracy'}
    (run / 'stability-resume-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Stability resume: eight-rank model/optimizer/RNG/scheduler restored; two updates, data/LR continuity, 32 validation outputs: PASS')


if __name__ == '__main__':
    verify_stability_resume(*sys.argv[1:])
