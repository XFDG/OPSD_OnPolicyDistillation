"""Verify six updates on the original dataset and original 1739-step schedule."""
import json
import math
from pathlib import Path
import re
import sys

from launcher_status import REPO, ROOT, selected_profile, source_manifest
from verify_resume import checkpoint_evidence, cosine_lr


def read_metrics(run, phase):
    log = (run / f'{phase}-train.log').read_text(errors='replace')
    live = run / 'ray-live.log'
    if live.exists():
        log += '\n' + live.read_text(errors='replace')
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    steps = {}
    for line in log.splitlines():
        match = re.search(r'step:(\d+) - ', line)
        if match:
            values = {key: float(value) for key, value in
                      re.findall(r'([a-zA-Z0-9_/@-]+):([-+0-9.eE]+)', line)}
            if 'training/global_step' in values:
                steps[int(match[1])] = values
    return log, steps


def check_metrics(steps, expected):
    assert set(steps) == set(expected), (steps.keys(), expected)
    for step, metrics in steps.items():
        for key, value in {'rollout/prompts': 8, 'rollout/samples_per_prompt': 16,
                           'rollout/generated_sequences': 128, 'tip/global_rollouts': 128,
                           'tip/enabled': 1, 'tip/global_normalization': 1,
                           'opt/teacher_cache_cuda': 1, 'opt/resident_teacher': 1}.items():
            assert metrics.get(key) == value, (step, key, metrics.get(key))
        assert metrics['training/global_step'] == step
        assert 0.48 < metrics['tip/selected_fraction'] <= 0.5
        assert math.isfinite(metrics['opd/loss']) and metrics['opd/loss'] >= 0
        assert math.isfinite(metrics['opd/grad_norm']) and metrics['opd/grad_norm'] > 0
        assert math.isclose(metrics['opd/lr'], cosine_lr(step - 1), rel_tol=1e-8, abs_tol=1e-14)
        assert all(math.isfinite(value) for key, value in metrics.items() if key.startswith(('opd/', 'tip/', 'timing/', 'perf/')))
        assert metrics['tip/global_selected_tokens'] > 0 and metrics['tip/global_response_tokens'] > 0
        assert not any(key.startswith('profile/') for key in metrics), 'Profiling must be disabled for stability timing'


def check_profile(run):
    recorded = json.loads((run / 'profile.json').read_text())
    assert recorded['profile'] == selected_profile(), 'Stability settings differ from the selected production profile'
    assert recorded['source_sha256'] == source_manifest(REPO), 'Stability source hashes differ'
    return recorded


def check_original_dataset(log):
    import pyarrow.parquet as parquet
    train = ROOT / 'OPSD-B200-assets/data/grpo_processed/train.parquet'
    assert str(train) in log and '/smoke-data/train.parquet' not in log
    assert parquet.read_metadata(train).num_rows == 13918


def check_output(run, kind, checkpoint_steps, final):
    outputs = list((run / kind).glob('Qwen3-4B-*'))
    assert len(outputs) == 1, outputs
    out = outputs[0]
    actual = sorted(int(path.name.rsplit('_', 1)[1]) for path in out.glob('global_step_*'))
    assert actual == list(checkpoint_steps), actual
    evidence = [checkpoint_evidence(out / f'global_step_{step}') for step in checkpoint_steps]
    assert (out / 'latest_checkpointed_iteration.txt').read_text().strip() == str(final)
    validation = [json.loads(line) for line in (out / f'{final}.jsonl').open()]
    assert len(validation) == 32, len(validation)
    assert all(isinstance(row.get('acc'), (int, float)) and math.isfinite(row['acc']) for row in validation)
    return out, evidence


def verify_stability(run):
    run = Path(run)
    profile = check_profile(run)
    log, steps = read_metrics(run, 'stability')
    assert 'Total training steps: 1739' in log
    # The trainer's INFO completion log may be filtered. This shell marker
    # is printed only after Python exits zero; metrics/checkpoints prove step 6.
    assert '=== OPD training completed ===' in log
    check_original_dataset(log)
    check_metrics(steps, range(1, 7))
    out, checkpoints = check_output(run, 'stability-output', (2, 4, 6), 6)
    report = {'passed': True, 'profile': profile, 'steps': steps, 'checkpoint_evidence': checkpoints,
              'output': str(out), 'validation_sequences': 32, 'dataset_rows': 13918,
              'scheduler_total_training_steps': 1739,
              'scope': 'six real-model eight-rank updates and checkpoint/evaluation; not full-epoch accuracy'}
    (run / 'stability-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Stability: six global TIP updates, original schedule/dataset, eight-rank checkpoints, RNG/data and 32 validation outputs: PASS')


if __name__ == '__main__':
    verify_stability(sys.argv[1])
