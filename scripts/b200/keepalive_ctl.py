"""Control the identified Taihua keepalive parent and its GPU children only."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT = Path('/volume/pt-test/users/zhaoye')
BASE = ROOT/'gpu-workspace/keep_alive'
STATE = ROOT/'OPSD-B200-runtime/keepalive'
MAIN = (BASE/'main.py').resolve()

def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return int(fields[19])
    except FileNotFoundError: return None

def parent():
    matches = []
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal(): continue
        try:
            argv = (p/'cmdline').read_bytes().split(b'\0')
            if len(argv) > 1 and Path(os.fsdecode(argv[0])).name.startswith('python') and Path(os.fsdecode(argv[1])).resolve() == MAIN:
                pid = int(p.name)
                tick = identity(pid)
                if tick is not None: matches.append((pid,tick))
        except (FileNotFoundError, ProcessLookupError): pass
    assert len(matches) <= 1, f'Multiple keepalive parents: {matches}'
    return matches[0] if matches else None

def children(pid):
    found = {pid}
    pending = [pid]
    while pending:
        cur = pending.pop()
        try:
            for task in Path(f'/proc/{cur}/task').iterdir():
                for c in (task/'children').read_text().split():
                    c = int(c)
                    if c not in found: found.add(c); pending.append(c)
        except FileNotFoundError: pass
    return found

def gpu():
    out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {int(x.strip()) for x in out.splitlines() if x.strip()}

def checked():
    p = parent()
    allowed = children(p[0]) if p else set()
    active = gpu()
    assert not active-allowed, f'Unrelated GPU processes; refusing action: {active-allowed}'
    return p, active, allowed

def main(action):
    assert subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(ROOT)],text=True).strip() == 'gpfs'
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p, active, allowed = checked()
        if action == 'status':
            print(f'[b200-keepalive] parent={p} GPU_workers={len(active)}',flush=True)
            return 0 if p and len(active)==8 else 3
        if action == 'stop':
            if p:
                assert parent() == p
                fd = os.pidfd_open(p[0])
                try:
                    assert identity(p[0]) == p[1]
                    signal.pidfd_send_signal(fd,signal.SIGTERM)
                finally: os.close(fd)
                for _ in range(120):
                    assert not gpu()-allowed, 'Unrelated GPU process appeared during stop'
                    if identity(p[0]) is None and not gpu(): break
                    time.sleep(0.5)
                else: raise RuntimeError('Keepalive did not stop; no GPU work may start')
            marker = BASE/'keep-alive.pid'
            if p and marker.exists() and marker.read_text().strip() == str(p[0]): marker.unlink()
            print('[b200-keepalive] stopped, GPUs idle',flush=True)
            return 0
        assert action=='start'
        if not p:
            env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','VIRTUAL_ENV','PYTHON_EXEC'} and not k.startswith(('NCCL_','TORCH_NCCL_'))}
            env.update(KEEP_ALIVE_PYTHON='/opt/venv/bin/python',KEEP_ALIVE_DASHBOARD='0',PYTHONUNBUFFERED='1')
            subprocess.run(['bash',str(BASE/'run.sh'),'start'],env=env,check=True)
        for _ in range(120):
            p, active, _ = checked()
            if p and len(active)==8:
                print(f'[b200-keepalive] active parent={p}, 8 GPU workers',flush=True)
                return 0
            time.sleep(1)
        raise RuntimeError('Keepalive readiness timeout')

if __name__=='__main__':
    try: sys.exit(main(sys.argv[1]))
    except Exception as e:
        print(f'[b200-keepalive] ERROR: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
