"""Capture and verify the CPU-cache/offload baseline without profiling syncs."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from launcher_status import REPO, execution_sources, source_manifest
from verify_resume import cosine_lr
from verify_stability import check_original_dataset, check_output, read_metrics


def baseline_profile():
    return {'teacher_cache_device': 'cpu', 'resident_teacher': False,
            'actor_param_offload': True, 'actor_optimizer_offload': True,
            'ref_param_offload': True, 'micro_batch_size_per_gpu': 1,
            'rollout_tp': 2, 'profile': False, 'world_size': 8,
            'gradient_checkpointing': True, 'train_prompts': 8, 'rollout_n': 16,
            'global_rollouts': 128, 'prompt_length': 2048, 'response_length': 8192,
            'opd_max_length': 16384, 'opd_chunk_size': 512, 'loss': 'reverse_kl',
            'tip_keep_ratio': 0.5, 'entropy_clip_quantile': 0.98,
            'learning_rate': 1e-6, 'lr_schedule': 'cosine', 'warmup_steps': 0,
            'weight_decay': 0.1, 'grad_clip': 1.0, 'total_training_steps': 1739,
            'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'thinking': False,
            'precision': 'BF16-forward/FP32-loss', 'rollout_attention': 'fa4',
            'rollout_memory_utilization': 0.6, 'val_n': 16, 'experimental_overlap': False,
            'dataset_rows': 13918, 'stop_after_steps': 6,
            'save_freq': 2, 'test_freq': 0, 'final_validation_sequences': 32,
            'student_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
            'teacher_revision': 'b968826d9c46dd6066d109eabc6255188de91218',
            'verl_revision': '0ddd28933f2d06fbf06d2d4b2cec7da547d596fc'}


def baseline_sources(repo):
    repo = Path(repo)
    sources = source_manifest(repo)
    sources['run_b200_opt.sh'] = hashlib.sha256((repo / 'run_b200_opt.sh').read_bytes()).hexdigest()
    return sources


def verification_changes(original, current):
    """Record changes outside the baseline's captured execution-source set."""
    return {name: {'old': original.get(name), 'new': current.get(name)}
            for name in sorted(set(original) | set(current))
            if original.get(name) != current.get(name)}


def capture_profile(repo, run):
    expected = {'OPD_TEACHER_CACHE_DEVICE': 'cpu', 'OPD_RESIDENT_TEACHER': 'False',
                'OPD_ACTOR_PARAM_OFFLOAD': 'True', 'OPD_ACTOR_OPTIM_OFFLOAD': 'True',
                'OPD_REF_PARAM_OFFLOAD': 'True', 'OPD_PROFILE': 'False',
                'TRAIN_BATCH_SIZE': '8', 'ROLLOUT_N': '16', 'PPO_MICRO_BATCH_SIZE_PER_GPU': '1',
                'TP_SIZE': '2', 'STOP_AFTER_STEPS': '6', 'SAVE_FREQ': '2', 'TEST_FREQ': '0',
                'LEARNING_RATE': '1e-6', 'LR_SCHEDULER_TYPE': 'cosine', 'RESUME_MODE': 'disable',
                'RUN_ID': 'b200-opt-baseline'}
    observed = {key: os.environ.get(key) for key in expected}
    assert observed == expected, (observed, expected)
    assert 'TRAIN_FILE_OVERRIDE' not in os.environ and 'TOTAL_TRAINING_STEPS' not in os.environ
    profile = {'profile': baseline_profile(), 'environment': observed, 'source_sha256': baseline_sources(repo)}
    (Path(run) / 'baseline-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    print('Actual CPU-cache/offload baseline profile captured:', json.dumps(profile['profile'], sort_keys=True))


def verify_baseline(run):
    run = Path(run)
    profile = json.loads((run / 'baseline-profile.json').read_text())
    assert profile['profile'] == baseline_profile(), 'Baseline settings differ'
    original_sources = profile['source_sha256']
    current_sources = baseline_sources(REPO)
    original_execution = execution_sources(original_sources)
    current_execution = execution_sources(current_sources)
    assert original_execution == current_execution, 'Baseline execution source hashes differ'
    changed = verification_changes(original_sources, current_sources)
    assert not execution_sources(changed), 'An execution source changed during baseline re-verification'
    log, steps = read_metrics(run, 'baseline-stability')
    assert 'Total training steps: 1739' in log
    # The trainer's py_logger.info completion line may be suppressed. The
    # shell reaches this marker only after Python returned successfully; the
    # six updates, checkpoint state and validation are independently checked.
    assert '=== OPD training completed ===' in log
    check_original_dataset(log)
    assert set(steps) == set(range(1, 7)), steps.keys()
    for step, metrics in steps.items():
        for key, value in {'rollout/prompts': 8, 'rollout/samples_per_prompt': 16,
                           'rollout/generated_sequences': 128, 'tip/global_rollouts': 128,
                           'tip/enabled': 1, 'tip/global_normalization': 1,
                           'opt/teacher_cache_cuda': 0, 'opt/resident_teacher': 0}.items():
            assert metrics.get(key) == value, (step, key, metrics.get(key))
        assert metrics['training/global_step'] == step
        assert 0.48 < metrics['tip/selected_fraction'] <= 0.5
        assert math.isfinite(metrics['opd/loss']) and metrics['opd/loss'] >= 0
        assert math.isfinite(metrics['opd/grad_norm']) and metrics['opd/grad_norm'] > 0
        assert math.isclose(metrics['opd/lr'], cosine_lr(step - 1), rel_tol=1e-8, abs_tol=1e-14)
        assert all(math.isfinite(value) for key, value in metrics.items()
                   if key.startswith(('opd/', 'tip/', 'timing/', 'perf/')))
        assert metrics['tip/global_selected_tokens'] > 0 and metrics['tip/global_response_tokens'] > 0
        assert not any(key.startswith('profile/') for key in metrics), 'Baseline profiling must be disabled'
    out, checkpoints = check_output(run, 'baseline-output', (2, 4, 6), 6)
    report = {'passed': True, 'profile': profile, 'steps': steps, 'checkpoint_evidence': checkpoints,
              'output': str(out), 'validation_sequences': 32,
              'scheduler_total_training_steps': 1739,
              'execution_source_sha256': original_execution,
              'current_execution_source_sha256': current_execution,
              'verification_sources_changed': changed,
              'scope': 'six full-dataset eight-rank baseline updates without profiling syncs; not full-epoch accuracy'}
    (run / 'baseline-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Baseline: six global TIP updates, original schedule/data, three eight-rank checkpoints, 32 validation outputs, no profiling: PASS')
    return report


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == 'capture-profile':
        capture_profile(args[1], args[2])
    elif len(args) == 1:
        verify_baseline(args[0])
    else:
        raise SystemExit('usage: verify_baseline.py RUN | capture-profile REPO RUN')
