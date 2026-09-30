"""CPU-only post-audit for the baseline's missing INFO completion false alarm.

Run on the paired Taihua GPU Pod while its known eight-worker keepalive is
active, with CUDA_VISIBLE_DEVICES=''. Original driver/GPU exit=1 and all
original logs/profile hashes are preserved. No training or keepalive mutation
occurs. Acceptance must bind this audit separately from the original run.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

import torch

from launcher_status import REPO, ROOT, RUNTIME, execution_sources
from verify_baseline import baseline_sources, verification_changes, verify_baseline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(path):
    return re.sub(r'\x1b\[[0-9;]*m', '', path.read_text(errors='replace'))


def original_failure(run):
    paths = {
        'driver': run / 'driver.log',
        'gpu': run / 'gpu-baseline-stability.log',
        'training': run / 'baseline-stability-train.log',
        'ray_live': run / 'ray-live.log',
        'profile': run / 'baseline-profile.json',
    }
    assert all(path.is_file() for path in paths.values()), paths
    before = {name: digest(path) for name, path in paths.items()}
    texts = {name: plain(path) for name, path in paths.items() if name != 'profile'}
    marker = '=== OPD training completed ==='
    assert marker in texts['training'], 'Training shell did not report successful Python completion'
    assert 'OPD training complete at step 6' not in texts['training'] + texts['ray_live'], 'This audit is restricted to missing INFO completion'
    assert re.findall(r'launcher exit=(\d+);', texts['driver']) == ['1'], 'Original launcher result was not exactly exit=1'
    assert re.findall(r'GPU baseline-stability finished with exit=(\d+);', texts['gpu']) == ['1'], 'Original GPU result was not exactly exit=1'
    assert 'baseline-stability training returned' not in texts['gpu'], 'Training process itself returned nonzero'
    expected_statement = "assert 'OPD training complete at step 6' in log"
    traceback_reports = {}
    for name in ('driver', 'gpu'):
        text = texts[name]
        assert text.count('Traceback (most recent call last):') == 1, (name, 'Unexpected traceback count')
        start = text.index('Traceback (most recent call last):')
        match = re.search(r'Traceback \(most recent call last\):\n(?P<body>.*?)\nAssertionError\s*\n', text, re.S)
        assert match is not None, (name, 'Original failure is not the known AssertionError')
        body = match.group('body')
        assert expected_statement in body, (name, 'Verifier failed on a different assertion')
        frames = re.findall(r'^\s*File "([^"]+)", line \d+, in [^\n]+', body, re.M)
        assert len(frames) == 2 and all(Path(frame) == REPO / 'scripts/optimization/verify_baseline.py' for frame in frames), (name, frames)
        assert marker in text[:start], (name, 'Failure occurred before training completion')
        assert text.count('AssertionError') == 1, (name, 'Another assertion failed')
        traceback_reports[name] = match.group(0)
    for name in ('training', 'ray_live'):
        assert 'Traceback (most recent call last):' not in texts[name], (name, 'Training/worker traceback found')
        assert not re.search(r'OutOfMemoryError|CUDA out of memory|FloatingPointError|illegal memory access|RayTaskError', texts[name], re.I), (name, 'Runtime error found')
    gpu_tail = texts['gpu'][texts['gpu'].index('AssertionError'):]
    assert re.search(r'\[b200-keepalive\] active parent=\(\d+, \d+\), 8 GPU workers', gpu_tail), 'Original cleanup did not confirm restored keepalive'
    growth = re.findall(r'overlay growth: /root=(-?\d+) bytes; /tmp=(-?\d+) bytes', gpu_tail)
    assert growth == [('0', '0')], ('Original storage cleanup was not clean', growth)
    assert 'driver fallback: keep-alive already active' in texts['driver'], 'Original driver did not confirm keepalive'
    return paths, before, traceback_reports


def keepalive_status():
    result = subprocess.run(['bash', str(REPO / 'scripts/b200/keepalive.sh'), 'status'],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, ('Known keepalive is not active', result.returncode, result.stdout, result.stderr)
    assert re.search(r'parent=\(\d+, \d+\) GPU_workers=8', result.stdout), result.stdout
    return result.stdout


def reverify(run):
    assert socket.gethostname() == 'zhaoye-taihua-gpu-0', 'CPU post-audit must run on the paired Taihua GPU Pod'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CUDA_VISIBLE_DEVICES empty is required'
    assert not torch.cuda.is_initialized(), 'The CPU post-audit must not initialize CUDA'
    assert subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(ROOT)], text=True).strip() == 'gpfs'
    run = Path(run).resolve()
    assert run.parent == RUNTIME / 'runs' and run.name.startswith('tip-'), run
    paths, before, tracebacks = original_failure(run)
    initial_keepalive = keepalive_status()
    original_profile = json.loads(paths['profile'].read_text())
    original_sources = original_profile['source_sha256']
    current_sources = baseline_sources(REPO)
    original_execution = execution_sources(original_sources)
    current_execution = execution_sources(current_sources)
    assert original_execution == current_execution, 'Baseline executed code changed; CPU re-audit cannot substitute a new GPU run'
    changed = verification_changes(original_sources, current_sources)
    assert not execution_sources(changed), changed
    # This reads only CPU-mapped rank extra states/data state and validates all
    # six real metrics, three complete checkpoints and 32 final evaluations.
    normal_report = verify_baseline(run)
    assert normal_report['profile'] == original_profile, 'Original recorded profile must be preserved'
    final_keepalive = keepalive_status()
    nvml = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert len(nvml.splitlines()) == 8 and all('B200' in line for line in nvml.splitlines()), nvml
    assert len({int(line.strip()) for line in processes.splitlines() if line.strip()}) == 8, processes
    assert not torch.cuda.is_initialized(), 'CPU post-audit unexpectedly initialized CUDA'
    after = {name: digest(path) for name, path in paths.items()}
    assert before == after, 'Original logs/profile changed during CPU verification'
    report = {
        'passed': True, 'original_launcher_exit': 1, 'original_gpu_wrapper_exit': 1,
        'cpu_verification_exit': 0, 'original_results_preserved': True,
        'execution_source_sha256': original_execution,
        'original_execution_source_sha256': original_execution,
        'current_execution_source_sha256': current_execution,
        'verification_sources_changed': changed,
        'keepalive_status': final_keepalive, 'keepalive_status_before': initial_keepalive,
        'reason': 'Only the post-training observer asserted a py_logger.info completion string that default logging suppressed; the original Python training shell returned successfully and six updates, all eight-rank checkpoint/LR/RNG/data states and 32 final validation outputs independently passed CPU re-verification.',
        'original_failure_tracebacks': tracebacks,
        'original_evidence_sha256': before, 'original_evidence_sha256_after': after,
        'baseline_verification_sha256': digest(run / 'baseline-verification.json'),
        'nvml_gpu_status': nvml, 'nvml_compute_pids': processes,
        'cpu_only': {'CUDA_VISIBLE_DEVICES': '', 'cuda_initialized': False},
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run),
        'scope': 'CPU post-audit of an already completed six-step baseline; original GPU/launcher exit=1 remains recorded; no GPU work, checkpoint writes or keepalive mutations occurred',
    }
    (run / 'baseline-reverification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Baseline CPU post-audit PASS: original launcher=1, GPU wrapper=1 retained; CPU verification=0; original evidence unchanged; known eight-worker keepalive active', flush=True)
    return report


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'usage: CUDA_VISIBLE_DEVICES=\'\' python reverify_baseline.py RUN'
    reverify(sys.argv[1])
