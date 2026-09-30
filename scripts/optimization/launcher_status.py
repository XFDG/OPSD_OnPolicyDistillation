"""Progress, production checkpoint discovery, and accepted-profile checks."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/volume/pt-test/users/zhaoye')
RUNTIME = ROOT / 'OPSD-B200-opt-runtime'
REPO = ROOT / 'OPSD_B200_Optimize'
EXPERIMENT = 'Qwen3-4B-b200-opt-full-TIP-rho0.5-reverse_kl-teacher-Qwen3-8B-nothink-lr1e-6-bs8-n16'


def selected_profile():
    tp = int(os.environ.get('OPD_ROLLOUT_TP', '2'))
    if tp not in (1, 2):
        raise ValueError('OPD_ROLLOUT_TP must be 1 or 2')
    return {'teacher_cache_device': 'cuda', 'resident_teacher': True,
            'actor_param_offload': False, 'actor_optimizer_offload': False,
            'ref_param_offload': False, 'micro_batch_size_per_gpu': 1,
            'rollout_tp': tp, 'profile': False, 'world_size': 8,
            'gradient_checkpointing': True, 'train_prompts': 8, 'rollout_n': 16,
            'global_rollouts': 128, 'prompt_length': 2048, 'response_length': 8192,
            'opd_max_length': 16384, 'opd_chunk_size': 512,
            'loss': 'reverse_kl', 'tip_keep_ratio': 0.5, 'entropy_clip_quantile': 0.98,
            'learning_rate': 1e-6, 'lr_schedule': 'cosine', 'warmup_steps': 0,
            'weight_decay': 0.1, 'grad_clip': 1.0, 'total_training_steps': 1739,
            'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'thinking': False,
            'precision': 'BF16-forward/FP32-loss', 'rollout_attention': 'fa4',
            'rollout_memory_utilization': 0.6, 'val_n': 16, 'experimental_overlap': False,
            'student_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
            'teacher_revision': 'b968826d9c46dd6066d109eabc6255188de91218',
            'verl_revision': '0ddd28933f2d06fbf06d2d4b2cec7da547d596fc'}


def source_manifest(repo):
    repo = Path(repo)
    paths = [repo / 'full.sh', repo / 'run_b200_optimized_full.sh', repo / 'run_b200_opt.sh', repo / 'scripts/opd/train_opd.sh']
    paths += sorted((repo / 'src/opd').rglob('*.py'))
    paths += sorted((repo / 'src/opd/config').rglob('*.yaml'))
    paths += sorted((repo / 'scripts/optimization').glob('*.py'))
    paths += sorted((repo / 'scripts/b200').glob('*.py'))
    paths += [repo / 'scripts/b200/keepalive.sh']
    paths += sorted((repo / 'src/rewards').glob('*.py'))
    paths += sorted((repo / 'tests').glob('*.py'))
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def execution_sources(manifest):
    """Actual baseline execution code; verifier-only changes permit re-audit."""
    return {path: digest for path, digest in manifest.items()
            if path.startswith(('src/', 'scripts/b200/'))
            or path in ('scripts/opd/train_opd.sh', 'run_b200_opt.sh')}


def capture_profile(repo, run):
    report = {'profile': selected_profile(), 'source_sha256': source_manifest(repo)}
    (Path(run) / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Selected profile captured:', json.dumps(report['profile'], sort_keys=True))


def verify_acceptance(repo, stamp):
    accepted = json.loads(Path(stamp).read_text())
    assert accepted.get('passed') is True, 'Profile acceptance is not passed'
    assert accepted.get('profile') == selected_profile(), 'Accepted optimization settings differ'
    assert accepted.get('source_sha256') == source_manifest(repo), 'Accepted source hashes differ'
    verify_evidence(accepted)
    # A missing stamp or any mismatch is terminal for production modes. There
    # is deliberately no environment variable that can bypass this check.
    print('Production profile and source acceptance: PASS')


def verify_evidence(accepted):
    """Bind acceptance to bounded real-model, resume, capacity and speed proofs."""
    kinds = {'baseline': ('baseline-stability', 'baseline-verification.json'),
             'stability': ('stability', 'stability-verification.json'),
             'resume': ('stability-resume', 'stability-resume-verification.json'),
             'long_sequence': ('long-gate', 'long-sequence-verification.json')}
    evidence = accepted['evidence']
    assert set(evidence) == set(kinds), 'Missing optimization acceptance evidence'
    reports = {}
    for name, (phase, filename) in kinds.items():
        item = evidence[name]
        run = Path(item['run_dir']).resolve()
        assert run.parent == RUNTIME / 'runs' and run.name.startswith('tip-'), run
        data = (run / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['report_sha256'], (name, 'proof changed')
        report = json.loads(data)
        assert report.get('passed') is True, (name, 'proof did not pass')
        driver_log = (run / 'driver.log').read_text(errors='replace')
        gpu_log = (run / f'gpu-{phase}.log').read_text(errors='replace')
        if name == 'baseline' and 'launcher exit=0;' not in driver_log:
            # Preserve the original observer failure. A separate CPU audit
            # proves six real updates/checkpoints without rerunning training.
            assert 'launcher exit=1;' in driver_log
            assert f'GPU {phase} finished with exit=1;' in gpu_log
            audit_data = (run / 'baseline-reverification.json').read_bytes()
            assert hashlib.sha256(audit_data).hexdigest() == item['audit_sha256']
            audit = json.loads(audit_data)
            assert audit['passed'] is True and audit['cpu_verification_exit'] == 0
            assert audit['original_launcher_exit'] == audit['original_gpu_wrapper_exit'] == 1
            assert audit['execution_source_sha256'] == execution_sources(accepted['source_sha256'])
            assert audit['keepalive_status'], 'Baseline post-audit did not confirm keepalive'
        else:
            assert 'launcher exit=0;' in driver_log, (name, 'launcher failed')
            assert f'GPU {phase} finished with exit=0;' in gpu_log, (name, 'GPU cleanup failed')
        assert 'GPU keep-alive active' in (run / 'driver.log').read_text(errors='replace'), (name, 'keepalive missing')
        if name in ('stability', 'resume'):
            assert report['profile']['profile'] == accepted['profile'], (name, 'profile changed')
            assert report['profile']['source_sha256'] == accepted['source_sha256'], (name, 'source changed')
        if name == 'baseline':
            assert execution_sources(report['profile']['source_sha256']) == execution_sources(accepted['source_sha256']), 'Baseline execution implementation differs'
        reports[name] = report
    long = reports['long_sequence']
    long_profile = json.loads((Path(evidence['long_sequence']['run_dir']) / 'profile.json').read_text())
    assert long_profile['profile'] == accepted['profile'] and long_profile['source_sha256'] == accepted['source_sha256'], 'Capacity worker source/profile differs'
    assert long['rollout_tp'] == accepted['profile']['rollout_tp']
    assert long['optimizer_primed'] is True and long['world_size'] == 8
    assert long['gate_source_sha256'] == accepted['source_sha256']['tests/test_b200_long_sequence.py']
    maximum = long['updates'][-1]
    assert maximum['response_tokens_per_sequence'] == 8192 and maximum['global_sequences'] == 128
    assert maximum['teacher_cache_gib_per_rank'] > 37
    assert long['nvml_monitor']['min_free_mib'] > 4096, 'Insufficient measured maximum-length memory margin'
    assert reports['stability']['checkpoint_evidence'][-1]['global_step'] == 6
    assert reports['resume']['source']['checkpoint'] == reports['stability']['checkpoint_evidence'][-1]['checkpoint'], 'Resume proof used another checkpoint'
    resumed_steps = sorted(map(int, reports['resume']['steps']))
    assert resumed_steps == [7, 8], 'Acceptance requires resuming the verified step-6 checkpoint'
    # Compare five updates after initialization. Checkpoint/validation timings
    # are reported separately; these update timings include rollout weight sync.
    means = {}
    for name in ('baseline', 'stability'):
        steps = reports[name]['steps']
        means[name] = sum(steps[str(step)]['timing/train_s'] for step in range(2, 7)) / 5
    improvement = 1 - means['stability'] / means['baseline']
    assert improvement >= 0.15, ('Update speed gain below 15%', means)
    return {'baseline_update_seconds': means['baseline'],
            'optimized_update_seconds': means['stability'],
            'update_time_reduction': improvement, 'steps': [2, 3, 4, 5, 6]}


def latest():
    candidates = []
    for out in (RUNTIME / 'runs').glob(f'tip-*/train-output/{EXPERIMENT}'):
        try:
            committed = int((out / 'latest_checkpointed_iteration.txt').read_text().strip())
        except (OSError, ValueError):
            continue
        if committed >= 1739:
            continue
        for checkpoint in out.glob('global_step_*'):
            if re.fullmatch(r'global_step_\d+', checkpoint.name):
                step = int(checkpoint.name.rsplit('_', 1)[1])
                if 0 < step <= committed:
                    candidates.append((step, checkpoint.stat().st_mtime_ns, checkpoint))
    for step, _, checkpoint in sorted(candidates, reverse=True):
        result = subprocess.run([sys.executable, '-B', str(REPO / 'scripts/optimization/verify_resume.py'),
                                 str(checkpoint)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f'[checkpoint] complete optimized production checkpoint: step {step}', file=sys.stderr)
            print(checkpoint)
            return
        print(f'[checkpoint] invalid {checkpoint}: {result.stderr[-400:]}', file=sys.stderr)
    raise SystemExit('No complete optimized production checkpoint found; specify a checkpoint explicitly.')


def tail(path, size=262144):
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - size))
        return stream.read().decode(errors='replace')


def progress(run):
    phase_path = run / 'phase'
    phase = phase_path.read_text().strip() if phase_path.exists() else ''
    paths = [run / f'{phase}-train.log'] if phase else list(run.glob('*-train.log'))
    paths = [path for path in paths if path.exists()]
    live = run / 'ray-live.log'
    if live.exists():
        paths.append(live)
    print(f'phase={phase or "preparing"} run={run.name}')
    best = None
    for path in paths:
        for line in tail(path).splitlines():
            found = re.search(r'(?:^|\s)step:(\d+) - ', line)
            if found:
                step = int(found[1])
                metrics = dict(re.findall(r'([a-zA-Z0-9_/@-]+):([-+0-9.eE]+)', line))
                item = (step, path.stat().st_mtime_ns, path, metrics)
                if best is None or item[:2] > best[:2]:
                    best = item
    if best:
        step, _, path, metrics = best
        names = ['opd/loss', 'opd/grad_norm', 'opd/lr', 'timing/generate_s', 'timing/train_s', 'timing/step_s']
        print(f'completed_step={step} source={path.name} ' +
              ' '.join(f'{name}={metrics[name]}' for name in names if name in metrics))
    else:
        print('No completed optimizer step yet; initialization or generation may be running.')
    kinds = {'smoke': 'smoke-output', 'stability': 'stability-output',
             'stability-resume': 'stability-resume-output', 'resume-smoke': 'resume-smoke-output',
             'train': 'train-output', 'resume': 'train-output'}
    steps = []
    for marker in run.glob(f'{kinds.get(phase, "*output")}/*/latest_checkpointed_iteration.txt'):
        try:
            steps.append(int(marker.read_text().strip()))
        except (OSError, ValueError):
            pass
    if steps:
        print(f'latest_saved_step={max(steps)}')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['latest']:
        latest()
    elif len(args) == 2 and args[0] == 'progress':
        progress(Path(args[1]))
    elif args == ['profile']:
        print(json.dumps(selected_profile(), indent=2))
    elif len(args) == 3 and args[0] == 'capture-profile':
        capture_profile(args[1], args[2])
    elif len(args) == 3 and args[0] == 'verify-acceptance':
        verify_acceptance(args[1], args[2])
    else:
        raise SystemExit('usage: launcher_status.py latest|profile|progress RUN|capture-profile REPO RUN|verify-acceptance REPO STAMP')
