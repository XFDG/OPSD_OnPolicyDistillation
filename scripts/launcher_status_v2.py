"""Read-only checkpoint discovery and progress from original Ray worker logs."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/volume/pt-train/users/zhaoye')
RUNTIME = ROOT/'OPSD-runtime'
EXPERIMENT = 'Qwen3-4B-beijing-tip-full-TIP-rho0.5-reverse_kl-teacher-Qwen3-8B-nothink-lr1e-6-bs8-n16'


def latest():
    candidates = []
    # Only this serial production experiment; never smoke or overlap outputs.
    for out in (RUNTIME/'runs').glob(f'tip-*/train-output/{EXPERIMENT}'):
        marker = out/'latest_checkpointed_iteration.txt'
        try:
            committed = int(marker.read_text().strip())
        except (OSError, ValueError):
            continue
        for p in out.glob('global_step_*'):
            if re.fullmatch(r'global_step_\d+', p.name):
                step = int(p.name.rsplit('_',1)[1])
                if step <= committed:
                    candidates.append((step, p.stat().st_mtime_ns, p))
    for step, _, p in sorted(candidates, reverse=True):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'OPSD_TIP_Aligned/scripts/verify_resume.py'), str(p)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f'[checkpoint] latest complete serial checkpoint: step {step}', file=sys.stderr)
            print(p)
            return
        print(f'[checkpoint] skipping invalid {p}: {result.stderr[-400:]}', file=sys.stderr)
    raise SystemExit('No complete serial production checkpoint found; supply an explicit verified checkpoint.')


def tail(path, size=262144):
    with path.open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0,f.tell()-size))
        return f.read().decode(errors='replace')


def progress(run):
    logs = list(run.glob('*-train.log'))
    sources = list(logs)
    # Ray's redirected actor output advances even when log forwarding lags.
    for path in logs:
        with path.open('rb') as f:
            prefix=f.read(262144).decode(errors='replace')
        pids=set(re.findall(r'OPDTaskRunner pid=(\d+)',prefix))
        for pid in pids:
            for raw in (ROOT/'ray/ray').glob(f'session_*/logs/worker-*-{pid}.out'):
                with raw.open('rb') as f:
                    header=f.read(128)
                if b':actor_name:OPDTaskRunner' in header:
                    sources.append(raw)
    best = None
    for path in sources:
        for line in tail(path).splitlines():
            found = re.search(r'(?:^|\s)step:(\d+) - ', line)
            if found:
                step=int(found[1])
                metrics=dict(re.findall(r'([a-zA-Z0-9_/@-]+):([-+0-9.eE]+)',line))
                item=(step,path.stat().st_mtime_ns,path,metrics)
                if best is None or item[:2] > best[:2]:
                    best=item
    if best:
        step,_,path,m=best
        fields=['opd/loss','opd/grad_norm','opd/lr','timing/generate_s','timing/train_s','timing/step_s']
        print(f'completed_step={step} source={path.name} '+ ' '.join(f'{k}={m[k]}' for k in fields if k in m))
    else:
        print('No completed optimizer step yet (initialization/generation may be running).')
    checkpoints=[]
    for marker in run.glob('*output/*/latest_checkpointed_iteration.txt'):
        try:
            checkpoints.append(int(marker.read_text().strip()))
        except (OSError,ValueError):
            pass
    if checkpoints:
        print(f'latest_saved_step={max(checkpoints)}')


if __name__ == '__main__':
    if sys.argv[1:] == ['latest']:
        latest()
    elif len(sys.argv) == 3 and sys.argv[1]=='progress':
        progress(Path(sys.argv[2]))
    else:
        raise SystemExit('usage: launcher_status_v2.py latest | progress RUN_ROOT')
