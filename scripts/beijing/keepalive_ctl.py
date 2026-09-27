#!/usr/bin/env python3
"""Control only the existing Beijing H200 keep-alive process.

The JSON status file is an audit record, never an authority for signaling.
Every action checks the live /proc script identity, PID start ticks, and NVML
compute PIDs. This program does not import torch or use broad process killing.
Run it on the Beijing GPU node; the paired CPU node cannot inspect GPU PIDs.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Iterator
from uuid import uuid4


SHARED_ROOT = Path("/volume/pt-train/users/zhaoye")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN = SHARED_ROOT / "gpu-workspace/keep_alive/main.py"
DEFAULT_LAUNCHER = SHARED_ROOT / "gpu-workspace/keep_alive/run.sh"
# The existing Beijing process uses this interpreter. Its symlink target is on
# the GPU-only personal mount, so the controller must execute on the GPU node.
DEFAULT_PYTHON = SHARED_ROOT / "runs/ubmoe-ubiq-6c9deea-beijing-gpu-20260920/test-venv/bin/python"
DEFAULT_STATE_DIR = PROJECT_ROOT / "artifacts/keepalive"


class SafetyError(RuntimeError):
    """An ambiguous or unsafe lifecycle operation was refused."""


class AmbiguousLaunchers(SafetyError):
    """The original launcher may still be running a short-lived shell child."""


@dataclass(frozen=True)
class Identity:
    pid: int
    start_ticks: int


@dataclass(frozen=True)
class Config:
    main: Path = DEFAULT_MAIN
    launcher: Path = DEFAULT_LAUNCHER
    python: Path = DEFAULT_PYTHON
    state_dir: Path = DEFAULT_STATE_DIR
    shared_root: Path = SHARED_ROOT
    proc_root: Path = Path("/proc")
    timeout: float = 45.0
    poll: float = 0.5


def log(message: str) -> None:
    print(f"[beijing-keepalive] {message}", flush=True)


def read_identity(pid: int, proc_root: Path = Path("/proc")) -> Identity | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        raise SafetyError(f"cannot inspect PID {pid}") from exc
    # /proc/<pid>/stat field 2 can contain spaces and closing parentheses.
    fields = raw[raw.rfind(")") + 2 :].split()
    if len(fields) < 20:
        raise SafetyError(f"malformed /proc/{pid}/stat")
    if fields[0] in {"Z", "X"}:
        return None
    return Identity(pid, int(fields[19]))  # field 22: process start time


def script_argument(argv: list[str], *, python: bool) -> str | None:
    """Find an interpreter's script operand, never an arbitrary argv match."""
    if not argv:
        return None
    executable = Path(argv[0]).name
    pattern = r"(?:python|pypy)(?:\d+(?:\.\d+)*)?" if python else r"(?:ba|da|k|z)?sh"
    if not re.fullmatch(pattern, executable):
        return None
    safe_options = (
        {"-u", "-B", "-E", "-I", "-s", "-S", "-O", "-OO", "-q"}
        if python else {"-e", "-u", "-eu", "-ue"}
    )
    for argument in argv[1:]:
        if argument == "--" or argument in safe_options:
            continue
        if argument.startswith("-"):
            return None  # especially -c and -m
        return argument
    return None


def find_script(config: Config, target: Path, *, python: bool) -> list[Identity]:
    matches: list[Identity] = []
    try:
        entries = list(config.proc_root.iterdir())
    except OSError as exc:
        raise SafetyError(f"cannot enumerate {config.proc_root}: {exc}") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        before = read_identity(pid, config.proc_root)
        if before is None:
            continue
        try:
            argv = [
                part.decode(errors="surrogateescape")
                for part in (entry / "cmdline").read_bytes().split(b"\0") if part
            ]
            operand = script_argument(argv, python=python)
            if operand is None:
                continue
            candidate = Path(operand)
            if not candidate.is_absolute():
                candidate = (entry / "cwd").resolve(strict=True) / candidate
            if candidate.resolve(strict=True) != target.resolve(strict=True):
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise SafetyError(f"cannot inspect argv/cwd of PID {pid}") from exc
        after = read_identity(pid, config.proc_root)
        if before == after:
            matches.append(before)
    return matches


def snapshot(config: Config) -> tuple[list[Identity], list[Identity]]:
    mains = find_script(config, config.main, python=True)
    launchers = find_script(config, config.launcher, python=False)
    if len(mains) > 1:
        raise SafetyError(
            f"ambiguous keep-alive processes: main={mains}, launchers={launchers}; no action taken"
        )
    if len(launchers) > 1 or (mains and launchers):
        raise AmbiguousLaunchers(
            f"ambiguous keep-alive processes: main={mains}, launchers={launchers}; no action taken"
        )
    return mains, launchers


def gpu_pids() -> set[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f"cannot reliably query GPU compute processes: {exc}") from exc
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not value.isdecimal() for value in values):
        raise SafetyError("unexpected nvidia-smi PID output; refusing lifecycle action")
    return {int(value) for value in values}


def check_gpu(allowed: set[int]) -> set[int]:
    active = gpu_pids()
    foreign = active - allowed
    if foreign:
        raise SafetyError(f"foreign GPU compute PIDs {sorted(foreign)}; no action taken")
    return active


def validate_storage(config: Config) -> None:
    if not config.state_dir.is_absolute() or not config.python.is_absolute():
        raise SafetyError("OPSD_KEEPALIVE_STATE_DIR and OPSD_KEEPALIVE_PYTHON must be absolute paths")
    root = config.shared_root.resolve(strict=True)
    state_dir = config.state_dir.resolve()
    if not state_dir.is_relative_to(root):
        raise SafetyError(f"state/log directory must be below the Beijing shared root: {state_dir}")
    try:
        result = subprocess.run(
            ["findmnt", "-n", "-o", "FSTYPE", "-T", str(root)],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f"cannot verify Beijing persistent mount: {exc}") from exc
    if result.stdout.strip() != "gpfs":
        raise SafetyError(f"Beijing shared root is not mounted as GPFS: {result.stdout.strip()!r}")


@contextlib.contextmanager
def controller_lock(config: Config, deadline: float) -> Iterator[None]:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.state_dir / f"controller-{socket.gethostname()}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SafetyError("timed out acquiring keep-alive controller lock")
                time.sleep(config.poll)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def write_status(config: Config, action: str, outcome: str, identity: Identity | None = None) -> None:
    """Persist an audit snapshot; no lifecycle decision ever reads this file."""
    record = {
        "action": action,
        "outcome": outcome,
        "host": socket.gethostname(),
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "pid": identity.pid if identity else None,
        "start_ticks": identity.start_ticks if identity else None,
        "authority": "live /proc and nvidia-smi only; this file is not a pidfile",
    }
    target = config.state_dir / f"status-{socket.gethostname()}.json"
    temp = config.state_dir / f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp"
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def matches_identity(config: Config, identity: Identity) -> bool:
    return identity in find_script(config, config.main, python=True)


def send_term(config: Config, identity: Identity) -> None:
    if not matches_identity(config, identity):
        raise SafetyError(f"PID identity changed before stop: {identity}")
    pid_fd: int | None = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            pid_fd = os.pidfd_open(identity.pid)
        if not matches_identity(config, identity):
            raise SafetyError(f"PID identity changed before signal: {identity}")
        check_gpu({identity.pid})
        if pid_fd is not None:
            signal.pidfd_send_signal(pid_fd, signal.SIGTERM)
        else:
            os.kill(identity.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # The caller still verifies that it exited and no GPU PID remains.
    finally:
        if pid_fd is not None:
            os.close(pid_fd)


def clean_launch_environment(source: dict[str, str], python: Path) -> dict[str, str]:
    """Keep the original launcher independent of training virtualenvs/ranks."""
    removed = {
        "PYTHONPATH", "PYTHONHOME", "PYTHON_EXEC", "PYTHONUSERBASE",
        "PYTHONSTARTUP", "PYTHONINSPECT", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT",
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER",
        "CUDA_VISIBLE_DEVICES", "KEEP_ALIVE_GPUS", "KEEP_ALIVE_NUM_GPUS",
        "KEEP_ALIVE_GPU_COUNT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT",
    }
    prefixes = ("TORCHELASTIC_", "TORCH_NCCL_", "NCCL_", "ACCELERATE_", "DEEPSPEED_")
    result = {k: v for k, v in source.items() if k not in removed and not k.startswith(prefixes)}
    result.update(
        KEEP_ALIVE_PYTHON=str(python),
        KEEP_ALIVE_DASHBOARD="0",
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
    )
    return result


def start(config: Config, deadline: float) -> Identity:
    mains, pending = snapshot(config)
    check_gpu({item.pid for item in mains})
    if pending:
        raise SafetyError(f"original launcher already pending: {pending}; refusing duplicate start")
    target = mains[0] if mains else None
    child: subprocess.Popen[bytes] | None = None
    if target is None:
        if not config.main.is_file() or not config.launcher.is_file():
            raise SafetyError("known main.py or run.sh is missing")
        if not config.python.is_file() or not os.access(config.python, os.X_OK):
            raise SafetyError(f"explicit KEEP_ALIVE_PYTHON is not executable on this node: {config.python}")
        check_gpu(set())
        log_path = config.state_dir / f"keepalive-{socket.gethostname()}.log"
        fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab", buffering=0) as output:
            child = subprocess.Popen(
                ["nohup", "bash", str(config.launcher)],
                cwd=str(config.launcher.parent),
                env=clean_launch_environment(dict(os.environ), config.python),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        log(f"starting original run.sh pid={child.pid}; GPFS log={log_path}")
    while True:
        active = gpu_pids()
        allowed = {child.pid} if child is not None else ({target.pid} if target is not None else set())
        foreign = active - allowed
        if foreign:
            raise SafetyError(f"foreign GPU compute PIDs {sorted(foreign)} during start")
        try:
            mains, pending = snapshot(config)
        except AmbiguousLaunchers:
            # run.sh uses command substitution while selecting its Python.
            # That short-lived shell child has the same script argv. Retry only
            # during this controller-owned launch; never declare readiness
            # while the process snapshot remains ambiguous.
            if child is None or time.monotonic() >= deadline:
                raise
            if child.poll() is not None:
                raise SafetyError(f"original run.sh exited with code {child.returncode}; inspect GPFS log")
            time.sleep(config.poll)
            continue
        if child is not None and child.poll() is not None:
            raise SafetyError(f"original run.sh exited with code {child.returncode}; inspect GPFS log")
        if target is not None and mains and mains[0] != target:
            raise SafetyError("keep-alive identity changed while verifying start")
        if mains:
            target = mains[0]
            if child is not None and target.pid != child.pid:
                raise SafetyError("unexpected main.py appeared during start; refusing to adopt it")
            if target.pid in active and matches_identity(config, target):
                log(f"running pid={target.pid} start_ticks={target.start_ticks}; GPU compute confirmed")
                return target
        if time.monotonic() >= deadline:
            raise SafetyError("start timed out; no second instance or automatic kill attempted")
        time.sleep(config.poll)


def stop(config: Config, deadline: float) -> None:
    mains, pending = snapshot(config)
    check_gpu({item.pid for item in mains})
    if pending:
        raise SafetyError(f"original launcher is pending: {pending}; refusing ambiguous stop")
    if not mains:
        log("stopped; no known main.py or GPU compute processes")
        return
    target = mains[0]
    send_term(config, target)
    log(f"TERM sent only to pid={target.pid} start_ticks={target.start_ticks}")
    while True:
        mains, pending = snapshot(config)
        if pending or any(item != target for item in mains):
            raise SafetyError("replacement/ambiguous keep-alive appeared during stop; no further signal sent")
        active = check_gpu({target.pid})
        if not mains and not active:
            log("stopped; exact main.py exited and GPU compute processes cleared")
            return
        if time.monotonic() >= deadline:
            raise SafetyError("stop timed out; SIGKILL was not sent; do not start GPU work")
        time.sleep(config.poll)


def status(config: Config) -> tuple[int, Identity | None]:
    mains, pending = snapshot(config)
    active = check_gpu({item.pid for item in mains})
    if pending:
        log(f"starting original run.sh pid={pending[0].pid}; not GPU-ready")
        return 3, None
    if not mains:
        log("stopped; no known main.py or GPU compute processes")
        return 3, None
    target = mains[0]
    if not matches_identity(config, target):
        raise SafetyError("main.py identity changed during status")
    ready = target.pid in active
    log(f"running pid={target.pid} start_ticks={target.start_ticks} gpu_active={int(ready)}")
    return (0 if ready else 3), target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "start", "stop"))
    args = parser.parse_args()
    try:
        config = Config(
            python=Path(os.environ.get("OPSD_KEEPALIVE_PYTHON", str(DEFAULT_PYTHON))),
            state_dir=Path(os.environ.get("OPSD_KEEPALIVE_STATE_DIR", str(DEFAULT_STATE_DIR))),
            timeout=float(os.environ.get("OPSD_KEEPALIVE_TIMEOUT", "45")),
        )
        if not 0 < config.timeout <= 120:
            raise SafetyError("OPSD_KEEPALIVE_TIMEOUT must be in (0, 120] seconds")
        validate_storage(config)
        deadline = time.monotonic() + config.timeout
        with controller_lock(config, deadline):
            try:
                if args.action == "status":
                    result, identity = status(config)
                elif args.action == "start":
                    identity = start(config, deadline)
                    result = 0
                else:
                    stop(config, deadline)
                    identity = None
                    result = 0
                write_status(config, args.action, "ok" if result == 0 else "inactive", identity)
                return result
            except (SafetyError, OSError) as exc:
                write_status(config, args.action, f"error: {exc}")
                raise
    except (SafetyError, OSError, ValueError) as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
