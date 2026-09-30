#!/usr/bin/env bash
# Accepted Taihua 8xB200 TIP optimization: bounded stability gates and manual full training.
# Run on the Taihua GPU Pod. Default only prepares; full training requires explicit all/train.
set -Eeuo pipefail

# Physical rollout parallelism is selectable; the acceptance stamp binds it.
export OPD_ROLLOUT_TP=${OPD_ROLLOUT_TP:-2}
case "$OPD_ROLLOUT_TP" in 1|2) ;; *) printf '[opsd] ERROR: OPD_ROLLOUT_TP must be 1 or 2\n' >&2; exit 2 ;; esac
export OPD_TEACHER_CACHE_DEVICE=cuda OPD_RESIDENT_TEACHER=True
export OPD_ACTOR_PARAM_OFFLOAD=False OPD_ACTOR_OPTIM_OFFLOAD=False OPD_REF_PARAM_OFFLOAD=False
export OPD_PROFILE=False

# Beijing is a remote console; actual work and files live on Taihua GPFS.
if [[ "$(hostname)" == yzhao04-0 ]]; then
    printf -v remote_command '%q ' env "OPD_ROLLOUT_TP=$OPD_ROLLOUT_TP" bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/run_b200_optimized_full.sh "$@"
    exec ssh -i /root/.ssh/id_ed25519_gpu_from_mac -o IdentitiesOnly=yes \
        -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 \
        -p 32064 root@116.178.239.130 "$remote_command"
fi

ROOT=/volume/pt-test/users/zhaoye
REPO=$ROOT/OPSD_B200_Optimize
ASSETS=$ROOT/OPSD-B200-assets
RUNTIME=$ROOT/OPSD-B200-opt-runtime
VENV=$ROOT/envs/opsd-py312-cu128
PY=$VENV/bin/python
VERL_SRC=$ROOT/OPSD-runtime/src/verl-0ddd289
KEEP=$REPO/scripts/b200/keepalive.sh
TRAIN=$REPO/scripts/opd/train_opd.sh
SELF=$REPO/run_b200_optimized_full.sh
SSH_KEY=/root/.ssh/id_ed25519_gpu_from_mac
GPU_TARGET=root@116.178.239.130
GPU_PORT=32064
MODEL_STUDENT=$ASSETS/models/Qwen3-4B
MODEL_TEACHER=$ASSETS/models/Qwen3-8B
VERL_COMMIT=0ddd28933f2d06fbf06d2d4b2cec7da547d596fc
QWEN4_REV=1cfa9a7208912126459214e8b04321603b3df60c
QWEN8_REV=b968826d9c46dd6066d109eabc6255188de91218
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8
          -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p "$GPU_PORT")

die() { printf '[opsd] ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '[opsd] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
check_gpfs() {
    local fs
    [[ -d "$ROOT" ]] || die "shared root missing: $ROOT"
    fs=$(findmnt -n -o FSTYPE -T "$ROOT") || die "cannot inspect shared mount"
    [[ "$fs" == gpfs ]] || die "$ROOT must be GPFS, observed $fs"
    [[ -d "$REPO" ]] || die "repo missing: $REPO"
}
setup_paths() {
    local run_root=$1 cuda_lib
    mkdir -p "$RUNTIME"/{tmp,cache,logs,locks,runs,wheels} "$ROOT/tmp" "$run_root"
    mkdir -p "$RUNTIME/cache"/{xdg,hf,torch,torch-extensions,torchinductor,triton,cuda,flashinfer,sglang,pycache,pip,ray,wandb,matplotlib,numba}
    # multiprocess/datasets also create AF_UNIX sockets under TMPDIR.
    export HYDRA_FULL_ERROR=1
    export TMPDIR="$ROOT/tmp" TMP="$ROOT/tmp" TEMP="$ROOT/tmp"
    export XDG_CACHE_HOME="$RUNTIME/cache/xdg"
    export HF_HOME="$RUNTIME/cache/hf" HF_HUB_CACHE="$RUNTIME/cache/hf/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$RUNTIME/cache/hf/datasets"
    export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
    export TORCH_HOME="$RUNTIME/cache/torch"
    export TORCH_EXTENSIONS_DIR="$RUNTIME/cache/torch-extensions"
    export TORCHINDUCTOR_CACHE_DIR="$RUNTIME/cache/torchinductor"
    export TRITON_CACHE_DIR="$RUNTIME/cache/triton"
    export CUDA_CACHE_PATH="$RUNTIME/cache/cuda"
    export FLASHINFER_CACHE_DIR="$RUNTIME/cache/flashinfer"
    export SGLANG_CACHE_DIR="$RUNTIME/cache/sglang"
    # Ray adds a session and socket suffix. Keep the GPFS prefix short enough
    # for Linux's 107-byte AF_UNIX path limit.
    mkdir -p "$ROOT/bray"
    export RAY_TMPDIR="$ROOT/bray"
    mkdir -p "$RAY_TMPDIR"
    export PIP_CACHE_DIR="$RUNTIME/cache/pip"
    export PYTHONPYCACHEPREFIX="$RUNTIME/cache/pycache"
    export MPLCONFIGDIR="$RUNTIME/cache/matplotlib"
    export NUMBA_CACHE_DIR="$RUNTIME/cache/numba"
    export WANDB_DIR="$RUNTIME/cache/wandb" WANDB_CACHE_DIR="$RUNTIME/cache/wandb"
    export RAY_DEDUP_LOGS=0
    export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    export PATH="$VENV/bin:$PATH"
    # SGLang spawns a fresh Python process, which must find CUDA wheel libraries
    # before Python has imported torch and loaded them into the process.
    for cuda_lib in "$VENV"/lib/python3.12/site-packages/nvidia/*/lib \
                    "$VENV"/lib/python3.12/site-packages/torch/lib; do
        [[ -d "$cuda_lib" ]] || continue
        export LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
}
ssh_gpu() {
    if [[ "$(hostname)" == zhaoye-taihua-gpu-0 ]]; then
        bash -c "$1" 9>&-
    else
        ssh "${SSH_OPTS[@]}" "$GPU_TARGET" "$1" 9>&-
    fi
}
keep_status() {
    local rc=0
    ssh_gpu "bash '$KEEP' status" || rc=$?
    return "$rc"
}
ensure_keepalive() {
    local rc=0
    keep_status || rc=$?
    case "$rc" in
        0) note "GPU keep-alive active" ;;
        3) note "GPU keep-alive absent; starting"; ssh_gpu "bash '$KEEP' start" || return $? ;;
        *) note "keep-alive controller refused status (exit $rc)"; return "$rc" ;;
    esac
}
watchdog_pid=
start_watchdog() {
    local parent_pid=$$
    watchdog_stop_file="$RUN_ROOT/watchdog.stop"
    (
        exec 9>&-
        local ticks=0
        while sleep 5; do
            [[ ! -e "$watchdog_stop_file" ]] || exit 0
            kill -0 "$parent_pid" 2>/dev/null || exit 0
            ticks=$((ticks + 1))
            if ((ticks % 6 == 0)); then
                if ! ensure_keepalive; then
                    printf 'keep-alive watchdog failed at %s\n' "$(date -u +%FT%TZ)" > "$RUN_ROOT/keepalive-watchdog.failed"
                    exit 1
                fi
            fi
        done
    ) &
    watchdog_pid=$!
}
stop_watchdog() {
    if [[ -n "$watchdog_pid" ]]; then
        : > "$watchdog_stop_file"
        wait "$watchdog_pid" 2>/dev/null || true
        watchdog_pid=
    fi
    [[ ! -f "$RUN_ROOT/keepalive-watchdog.failed" ]] || die "GPU keep-alive watchdog failed; see log"
}

verify_env() {
    "$PY" -B - "$VERL_SRC" "$REPO" <<'PY'
import importlib
import importlib.metadata as md
import json, pathlib, sys
want = {"torch":"2.9.1+cu128", "sglang":"0.5.6", "transformers":"4.57.1",
        "flash-attn":"2.8.3", "flashinfer-jit-cache":"0.5.3+cu128",
        "flashinfer-python":"0.5.3", "cachetools":"5.5.2",
        "nvidia-cutlass-dsl":"4.7.1", "quack-kernels":"0.6.5", "flash-attn-4":"0.0.0+e9cf2c1"}
for name, version in want.items():
    found = md.version(name)
    assert found == version, f"{name}: {found} != {version}"
url = json.loads(md.distribution("verl").read_text("direct_url.json"))["url"]
assert url == pathlib.Path(sys.argv[1]).as_uri(), f"wrong editable verl source: {url}"
import torch
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI, "Torch CXX11 ABI mismatch"
sys.path.insert(0, str(pathlib.Path(sys.argv[2]) / "src"))
for module in ("verl.experimental.agent_loop", "verl.workers.rollout.sglang_rollout",
               "opd.opd_trainer", "opd.opd_worker"):
    importlib.import_module(module)
print("venv versions, verl source, CUDA ABI: PASS")
PY
}
prepare_env() {
    note "[1/4] verifying migrated Python environment"
    [[ -x "$PY" ]] || die "migrated venv is missing"
    [[ "$(git -C "$VERL_SRC" rev-parse HEAD)" == "$VERL_COMMIT" ]] || die "wrong Verl revision"
    [[ -z "$(git -C "$VERL_SRC" status --porcelain)" ]] || die "pinned Verl source has local changes"
    export PYTHONPATH="$REPO/src"
    "$PY" -B "$REPO/scripts/b200/apply_fa4.py"
    verify_env
    "$PY" -m pip check
}

prepare_data() {
    local data=$ASSETS/data
    note "[2/4] verifying pinned migrated assets and preparing smoke data"
    "$PY" -B "$REPO/scripts/b200/verify_assets.py" "$ASSETS" "$REPO/scripts/b200/assets.sha256.json"
    "$PY" -B - "$data" "$RUN_ROOT" <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import sys
data, run = map(Path, sys.argv[1:])
expected = {"train":13918, "val_dapo":3480, "val_aime24":30, "val_aime25":30, "val_math500":500}
tables = {}
for name, rows in expected.items():
    path = data / "grpo_processed" / f"{name}.parquet"
    table = pq.read_table(path)
    assert table.num_rows == rows, (name, table.num_rows, rows)
    assert {"data_source", "prompt", "ability", "reward_model", "extra_info"} <= set(table.column_names)
    tables[name] = table
smoke = run / "smoke-data"
smoke.mkdir(parents=True, exist_ok=True)
pq.write_table(tables["train"].slice(0, 16), smoke / "train.parquet")
pq.write_table(tables["val_math500"].slice(0, 2), smoke / "val.parquet")
assert pq.read_metadata(smoke / "train.parquet").num_rows == 16
assert pq.read_metadata(smoke / "val.parquet").num_rows == 2
print("processed data and smoke 16/2 rows: PASS")
PY
}
model_gate() {
    "$PY" -B - "$1" "$2" <<'PY'
from pathlib import Path
import json, sys
root, revision = Path(sys.argv[1]), sys.argv[2]
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
assert shards and len(shards) >= 3, shards
needed = {"config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.json", "merges.txt",
          "model.safetensors.index.json"} | shards
for name in needed:
    path = root / name
    assert path.is_file() and path.stat().st_size > 0, f"missing or empty: {path}"
    meta = root / ".cache/huggingface/download" / (name + ".metadata")
    assert meta.is_file() and meta.read_text().splitlines()[0] == revision, f"wrong model revision: {name}"
from safetensors import safe_open
for shard in shards:
    with safe_open(root / shard, framework="pt", device="cpu") as sf:
        assert len(sf.keys()) > 0, f"empty shard: {shard}"
print(f"{root.name}: {len(shards)} shards at pinned {revision}: PASS")
PY
}
prepare_model() {
    note "[3/4] checking pinned model $1"
    model_gate "$3" "$2" || die "migrated model incomplete; repeat migration"
}

prepare_assets() {
    ensure_keepalive
    start_watchdog
    prepare_env
    prepare_data
    prepare_model Qwen/Qwen3-4B "$QWEN4_REV" "$MODEL_STUDENT"
    prepare_model Qwen/Qwen3-8B "$QWEN8_REV" "$MODEL_TEACHER"
    stop_watchdog
    ensure_keepalive
    note "asset preparation and pinned assets PASS"
}

# GPU-only helpers: all work and all logs stay on GPFS.
gpu_compute_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        awk 'NF {gsub(/[[:space:]]/, ""); if ($0 !~ /^[0-9]+$/) exit 2; print}' | sort -u
}
wait_gpu_idle() {
    local i pids
    for ((i=0; i<120; i++)); do
        pids=$(gpu_compute_pids) || return 1
        [[ -z "$pids" ]] && return 0
        sleep 1
    done
    note "GPU compute PIDs remain: $pids"
    return 1
}
du_bytes() { du -sb "$1" | awk '{print $1}'; }
gpu_cleanup() {
    local rc=$? pgid now_ticks after_root after_tmp root_growth tmp_growth
    trap - EXIT INT TERM HUP
    set +e
    if [[ -n "${raw_follow_pid:-}" ]]; then kill "$raw_follow_pid" 2>/dev/null; fi
    if [[ -n "${heartbeat_pid:-}" ]]; then kill "$heartbeat_pid" 2>/dev/null; fi
    if [[ -n "${work_pid:-}" ]] && kill -0 "$work_pid" 2>/dev/null; then
        pgid=$(ps -o pgid= -p "$work_pid" | tr -d '[:space:]')
        now_ticks=$(awk '{print $22}' "/proc/$work_pid/stat" 2>/dev/null)
        if [[ "$pgid" == "$work_pid" && "$now_ticks" == "$work_ticks" ]]; then
            note "stopping only this launch's process group $work_pid"
            kill -TERM -- "-$work_pid" 2>/dev/null
            wait "$work_pid" 2>/dev/null
        else
            note "process identity changed; refusing group signal"
            rc=5
        fi
    fi
    if [[ "${pause_requested:-0}" == 1 ]]; then
        if bash "$KEEP" status >/dev/null 2>&1; then
            note "GPU keep-alive already restored"
        elif wait_gpu_idle; then
            bash "$KEEP" start || rc=5
        else
            note "GPU remains occupied; keep-alive restoration safely refused"
            rc=5
        fi
    fi
    after_root=$(du_bytes /root) || rc=5
    after_tmp=$(du_bytes /tmp) || rc=5
    root_growth=$((after_root - before_root))
    tmp_growth=$((after_tmp - before_tmp))
    note "overlay growth: /root=$root_growth bytes; /tmp=$tmp_growth bytes"
    if ((root_growth > 67108864 || tmp_growth > 67108864)); then
        note "unexpected overlay growth above 64 MiB; inspect before expanding workload"
        rc=5
    fi
    note "GPU ${gpu_kind:-unknown} finished with exit=$rc; log=$RUN_ROOT/gpu-${gpu_kind:-unknown}.log"
    exit "$rc"
}
gpu_run() {
    local kind=$1 names gpu_count status=0 train_log
    [[ "$(hostname)" == zhaoye-taihua-gpu-0 ]] || die "GPU submode must run on zhaoye-taihua-gpu-0"
    check_gpfs
    RUN_ROOT=${OPSD_RUN_ROOT:?OPSD_RUN_ROOT required}
    [[ "$RUN_ROOT" == "$RUNTIME"/runs/* && -d "$RUN_ROOT" ]] || die "unsafe run root: $RUN_ROOT"
    setup_paths "$RUN_ROOT"
    exec > >(tee -a "$RUN_ROOT/gpu-$kind.log") 2>&1
    exec 8>"$ROOT/OPSD-B200-runtime/locks/gpu-full.lock"
    flock -n 8 || die "another OPSD GPU task owns the node lock"
    gpu_kind=$kind
    # all performs smoke and production in one run directory. Preserve smoke
    # evidence while ensuring progress never reports its steps as production.
    if [[ -f "$RUN_ROOT/ray-live.log" ]]; then
        mv -- "$RUN_ROOT/ray-live.log" "$RUN_ROOT/ray-live-before-$kind.log"
    fi
    printf '%s\n' "$kind" > "$RUN_ROOT/phase"
    work_pid= work_ticks= heartbeat_pid= pause_requested=0
    before_root=$(du_bytes /root)
    before_tmp=$(du_bytes /tmp)
    trap gpu_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    names=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    gpu_count=$(printf '%s\n' "$names" | wc -l)
    [[ "$gpu_count" == 8 ]] && ! printf '%s\n' "$names" | grep -qv 'B200' ||
        die "this profile requires exactly eight visible B200 GPUs"
    bash "$KEEP" status || die "known keep-alive must be active before GPU work"
    pause_requested=1
    bash "$KEEP" stop
    wait_gpu_idle || die "GPU is not idle after keep-alive stop"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export MODEL_PATH="$MODEL_STUDENT" TEACHER_MODEL_PATH="$MODEL_TEACHER"
    export MODEL_NAME=Qwen3-4B DATA_DIR="$ASSETS/data" SKIP_DATA_PREP=1
    export PYTHONPATH="$REPO/src" PYTHON_EXEC="$PY"
    unset RAY_ADDRESS RAY_NAMESPACE
    # Paper settings; batch=8 is interpreted as prompts (8 x 16 = 128 rollouts).
    # One pass keeps response count comparable to the previous 15-epoch n=1 run.
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GPUS_PER_NODE=8 OPSD_EXPECTED_GPUS=8
    export AGENT_NUM_WORKERS=8 ENABLE_THINKING=False ROLLOUT_N=16
    export TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 PPO_MICRO_BATCH_SIZE_PER_GPU=1
    export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=8192 OPD_MAX_LENGTH=16384
    export OPD_CHUNK_SIZE=512 OPD_LOSS_TYPE=reverse_kl TIP_ENABLED=True TIP_KEEP_RATIO=0.5
    export TIP_ENTROPY_CLIP_QUANTILE=0.98 LEARNING_RATE=1e-6 LR_SCHEDULER_TYPE=cosine
    export TEMPERATURE=1.0 TOP_P=1.0 TOP_K=-1 VAL_TEMPERATURE=1.0 VAL_TOP_P=1.0 VAL_TOP_K=-1
    export SGLANG_ATTENTION_BACKEND=fa4
    export TP_SIZE="$OPD_ROLLOUT_TP" GPU_MEMORY_UTIL=0.6 VAL_N=16 VAL_BEFORE_TRAIN=False
    export OPD_TEACHER_CACHE_DEVICE=cuda OPD_RESIDENT_TEACHER=True OPD_PROFILE=False
    export OPD_ACTOR_PARAM_OFFLOAD=False OPD_ACTOR_OPTIM_OFFLOAD=False OPD_REF_PARAM_OFFLOAD=False
    if [[ "$kind" == train || "$kind" == resume ]]; then
        "$PY" -B "$REPO/scripts/optimization/launcher_status.py" verify-acceptance "$REPO" "$RUNTIME/accepted-profile.json" ||
            die "production optimization profile is not accepted for these source/settings"
    fi
    export TOTAL_EPOCHS=1 SAVE_FREQ=50 TEST_FREQ=50
    unset OPD_REWARD_BETA TRAIN_FILE_OVERRIDE VAL_FILES_OVERRIDE TOTAL_TRAINING_STEPS
    export RESUME_MODE=disable RESUME_FROM_PATH=null STOP_AFTER_STEPS=0
    if [[ "$kind" == smoke ]]; then
        export TRAIN_FILE_OVERRIDE="$RUN_ROOT/smoke-data/train.parquet"
        export VAL_FILES_OVERRIDE="['$RUN_ROOT/smoke-data/val.parquet']"
        export TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1 TEST_FREQ=2
        export OUTPUT_ROOT="$RUN_ROOT/smoke-output" RUN_ID=b200-opt-smoke
    else
        export OUTPUT_ROOT="$RUN_ROOT/train-output" RUN_ID=b200-opt-full
    fi
    if [[ "$kind" == resume || "$kind" == resume-smoke ]]; then
        export RESUME_MODE=resume_path RESUME_FROM_PATH="${OPSD_RESUME_FROM:?checkpoint required}"
        if [[ "$kind" == resume-smoke ]]; then
            export STOP_AFTER_STEPS=2 SAVE_FREQ=1 TEST_FREQ=0
            export VAL_FILES_OVERRIDE="['$RUN_ROOT/smoke-data/val.parquet']"
            export OUTPUT_ROOT="$RUN_ROOT/resume-smoke-output" RUN_ID=b200-opt-resume-smoke
        fi
    fi
    if [[ "$kind" == stability ]]; then
        export STOP_AFTER_STEPS=6 SAVE_FREQ=2 TEST_FREQ=0
        export VAL_FILES_OVERRIDE="['$RUN_ROOT/smoke-data/val.parquet']"
        export OUTPUT_ROOT="$RUN_ROOT/stability-output" RUN_ID=b200-opt-stability
    fi
    if [[ "$kind" == stability-resume ]]; then
        export RESUME_MODE=resume_path RESUME_FROM_PATH="${OPSD_RESUME_FROM:?stability checkpoint required}"
        export STOP_AFTER_STEPS=2 SAVE_FREQ=1 TEST_FREQ=0
        export VAL_FILES_OVERRIDE="['$RUN_ROOT/smoke-data/val.parquet']"
        export OUTPUT_ROOT="$RUN_ROOT/stability-resume-output" RUN_ID=b200-opt-stability-resume
    fi
    # Re-check profile continuity before launching any resumed GPU worker.
    if [[ "$kind" == resume || "$kind" == resume-smoke ]]; then
        "$PY" -B "$REPO/scripts/optimization/verify_resume.py" "$OPSD_RESUME_FROM" "$RUN_ROOT"
    elif [[ "$kind" == stability-resume ]]; then
        "$PY" -B "$REPO/scripts/optimization/verify_resume.py" "$OPSD_RESUME_FROM" "$RUN_ROOT" --stability
    fi
    "$PY" -B - <<'PY'
import ctypes, os
ctypes.CDLL("libcudart.so.12")
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
expected = int(os.environ["OPSD_EXPECTED_GPUS"])
assert torch.cuda.device_count() == expected
import flash_attn_2_cuda
import flashinfer
import verl
print(f"GPU Python gate: torch={torch.__version__}, CUDA={torch.version.cuda}, visible={torch.cuda.device_count()}, verl={verl.__file__}")
PY
    "$PY" -B "$REPO/scripts/b200/kernel_gate.py"
    "$PY" -B "$REPO/scripts/b200/fa4_gate.py" "$RUN_ROOT/fa4-kernel-verification.json"
    "$PY" -B "$REPO/tests/test_b200_cache_equivalence.py" "$RUN_ROOT/cache-equivalence.json"
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 "$REPO/tests/test_b200_cache_distributed.py" "$RUN_ROOT/cache-distributed.json"
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 "$REPO/tests/test_b200_cache_distributed.py" "$RUN_ROOT/cache-zero-rank.json" --zero-rank
    if [[ "$kind" == preflight ]]; then
        note "B200 eight-GPU kernel preflight PASS"
        return
    fi
    cd "$RUN_ROOT"
    train_log="$RUN_ROOT/$kind-train.log"
    note "[4/4] starting $kind; training log=$train_log"
    if [[ "$kind" == long-gate ]]; then
        [[ "${OPSD_LONG_CONFIG:-}" == "$RUNTIME"/runs/*/.hydra/config.yaml && -f "$OPSD_LONG_CONFIG" ]] ||
            die "long-gate requires an existing resolved candidate config under the experiment runtime"
        setsid bash -c 'set -o pipefail; "$1" -B "$2" --config "$3" --report "$4" --prime-optimizer --rollout-tp "$5" 2>&1 | tee -a "$6"' \
            bash "$PY" "$REPO/tests/test_b200_long_sequence.py" "$OPSD_LONG_CONFIG" \
            "$RUN_ROOT/long-sequence-verification.json" "$OPD_ROLLOUT_TP" "$train_log" &
    else
        setsid bash -c 'set -o pipefail; bash "$1" 2>&1 | tee -a "$2"' bash "$TRAIN" "$train_log" &
    fi
    work_pid=$!
    "$PY" -u "$REPO/scripts/b200/follow_ray.py" "$RAY_TMPDIR" "$work_pid" "$RUN_ROOT" &
    raw_follow_pid=$!
    work_ticks=$(awk '{print $22}' "/proc/$work_pid/stat")
    (
        exec 8>&- 9>&-
        while kill -0 "$work_pid" 2>/dev/null; do
            note "GPU $kind running; latest training progress:"
            "$PY" -B "$REPO/scripts/optimization/launcher_status.py" progress "$RUN_ROOT" || true
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
            sleep 10
        done
    ) &
    heartbeat_pid=$!
    wait "$work_pid" || status=$?
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=
    # The follower drains the actor logs once more after the training process exits.
    wait "${raw_follow_pid:-}" 2>/dev/null || true
    work_pid=
    ((status == 0)) || die "$kind training returned $status"
    if [[ "$kind" == smoke ]]; then
        grep -Eq 'OPD training complete at step 2|training/global_step:2' "$train_log" "$RUN_ROOT/ray-live.log" ||
            die "smoke exited zero without evidence of two optimizer steps"
        "$PY" -B "$REPO/scripts/b200/verify_tip_smoke.py" "$RUN_ROOT"
    fi
    if [[ "$kind" == resume-smoke ]]; then
        "$PY" -B "$REPO/scripts/b200/verify_resume_smoke.py" "$RUN_ROOT" "$OPSD_RESUME_FROM"
    fi
    if [[ "$kind" == stability ]]; then
        "$PY" -B "$REPO/scripts/optimization/verify_stability.py" "$RUN_ROOT"
    elif [[ "$kind" == stability-resume ]]; then
        "$PY" -B "$REPO/scripts/optimization/verify_stability_resume.py" "$RUN_ROOT" "$OPSD_RESUME_FROM"
    fi
    note "$kind training PASS; EXIT trap will restore keep-alive"
}

driver_exit() {
    local rc=$? status_rc=0
    trap - EXIT INT TERM HUP
    set +e
    if [[ -n "${watchdog_pid:-}" ]]; then
        : > "$watchdog_stop_file"
        wait "$watchdog_pid" 2>/dev/null
    fi
    # Fallback if the GPU workload disconnects or exits before its trap.
    # The exact controller refuses start when any training/foreign GPU PID lives.
    if [[ "${gpu_attempted:-0}" == 1 ]]; then
        ssh_gpu "bash '$KEEP' status" || status_rc=$?
        if ((status_rc == 3)); then
            note "driver fallback: GPU protection absent; attempting safe start"
            ssh_gpu "bash '$KEEP' start" || rc=5
        elif ((status_rc == 0)); then
            note "driver fallback: keep-alive already active"
        else
            note "driver fallback: status=$status_rc; training/foreign GPU process or connection failure; no forced restart"
            rc=5
        fi
    fi
    note "launcher exit=$rc; log=${RUN_ROOT:-unknown}/driver.log"
    exit "$rc"
}
driver_main() {
    local mode=$1 stamp driver_host resume_q config_q
    driver_host=$(hostname)
    [[ "$driver_host" == zhaoye-taihua-cpu-0 || "$driver_host" == zhaoye-taihua-gpu-0 ]] ||
        die "manual entrypoint must run on Taihua zhaoye-taihua-cpu-0 or zhaoye-taihua-gpu-0"
    check_gpfs
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    RUN_ROOT="$RUNTIME/runs/tip-$stamp-$$"
    setup_paths "$RUN_ROOT"
    exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1
    exec 9>"$ROOT/OPSD-B200-runtime/locks/full.lock"
    flock -n 9 || die "another OPSD full/smoke/prepare owns the lock"
    ln -sfn "$RUN_ROOT" "$RUNTIME/latest-run"
    gpu_attempted=0 watchdog_pid=
    trap driver_exit EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    note "mode=$mode; run root=$RUN_ROOT; host=$(hostname); rollout TP=$OPD_ROLLOUT_TP"
    if [[ "$driver_host" == zhaoye-taihua-cpu-0 ]]; then
        [[ -r "$SSH_KEY" ]] || die "GPU SSH key missing"
    fi
    [[ -f "$KEEP" && -f "$TRAIN" ]] || die "required keep-alive/training scripts missing"
    "$PY" -B "$REPO/scripts/optimization/launcher_status.py" capture-profile "$REPO" "$RUN_ROOT"
    ensure_keepalive
    if [[ "$mode" == train || "$mode" == all || "$mode" == resume ]]; then
        "$PY" -B "$REPO/scripts/optimization/launcher_status.py" verify-acceptance "$REPO" "$RUNTIME/accepted-profile.json" ||
            die "production optimization profile is not accepted for these source/settings"
    fi
    if [[ "$mode" == preflight ]]; then
        prepare_env
        gpu_attempted=1
        ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_preflight"
        return
    fi
    prepare_assets
    if [[ "$mode" == long-gate ]]; then
        OPSD_LONG_CONFIG=${2:?long-gate requires a resolved candidate config.yaml}
        [[ "$OPSD_LONG_CONFIG" == "$RUNTIME"/runs/*/.hydra/config.yaml && -f "$OPSD_LONG_CONFIG" ]] ||
            die "long-gate config must exist under the experiment runtime"
        printf -v config_q '%q' "$OPSD_LONG_CONFIG"
        gpu_attempted=1
        ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_LONG_CONFIG=$config_q OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_longgate"
        ensure_keepalive
    fi
    if [[ "$mode" == resume || "$mode" == resume-smoke || "$mode" == stability-resume ]]; then
        OPSD_RESUME_FROM=${2:-${OPSD_RESUME_FROM:-}}
        if [[ "$mode" == stability-resume ]]; then
            [[ -n "$OPSD_RESUME_FROM" ]] || die "stability-resume requires an explicit stability checkpoint"
            "$PY" -B "$REPO/scripts/optimization/verify_resume.py" "$OPSD_RESUME_FROM" "$RUN_ROOT" --stability
        else
            if [[ -z "$OPSD_RESUME_FROM" ]]; then
                OPSD_RESUME_FROM=$("$PY" -B "$REPO/scripts/optimization/launcher_status.py" latest) || die "no valid checkpoint found"
            fi
            "$PY" -B "$REPO/scripts/optimization/verify_resume.py" "$OPSD_RESUME_FROM" "$RUN_ROOT"
        fi
        printf -v resume_q '%q' "$OPSD_RESUME_FROM"
        gpu_attempted=1
        case "$mode" in
            stability-resume) ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_RESUME_FROM=$resume_q OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_stability_resume" ;;
            resume-smoke) ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_RESUME_FROM=$resume_q OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_resume_smoke" ;;
            resume) ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_RESUME_FROM=$resume_q OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_resume" ;;
        esac
        ensure_keepalive
    fi
    if [[ "$mode" == smoke || "$mode" == all ]]; then
        gpu_attempted=1
        ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_smoke"
        ensure_keepalive
    fi
    if [[ "$mode" == stability ]]; then
        gpu_attempted=1
        ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_stability"
        ensure_keepalive
    fi
    if [[ "$mode" == train || "$mode" == all ]]; then
        gpu_attempted=1
        ssh_gpu "OPD_ROLLOUT_TP='$OPD_ROLLOUT_TP' OPSD_RUN_ROOT='$RUN_ROOT' bash '$SELF' __gpu_train"
        ensure_keepalive
    fi
    note "mode=$mode completed; outputs=$RUN_ROOT"
}
if (($# > 2)); then die "usage: bash $SELF [prepare|preflight|smoke|long-gate CONFIG|stability|stability-resume CHECKPOINT|all|train|resume|resume-smoke|status|follow]"; fi
case "${1:-prepare}" in
    status|follow)
        check_gpfs
        [[ -L "$RUNTIME/latest-run" ]] || die "no optimized B200 run yet"
        RUN_ROOT=$(readlink -f "$RUNTIME/latest-run")
        if [[ "$1" == follow ]]; then exec tail -n 80 -F "$RUN_ROOT/driver.log"; fi
        setup_paths "$RUN_ROOT"
        "$PY" -B "$REPO/scripts/optimization/launcher_status.py" progress "$RUN_ROOT"
        exit 0 ;;
    prepare|preflight|smoke|stability|train|all)
        (($# <= 1)) || die "checkpoint argument requires a resume mode"
        driver_main "${1:-prepare}" ;;
    long-gate|stability-resume|resume|resume-smoke) driver_main "$1" "${2:-}" ;;
    __gpu_preflight) gpu_run preflight ;;
    __gpu_smoke) gpu_run smoke ;;
    __gpu_stability) gpu_run stability ;;
    __gpu_stability_resume) gpu_run stability-resume ;;
    __gpu_longgate) gpu_run long-gate ;;
    __gpu_train) gpu_run train ;;
    __gpu_resume) gpu_run resume ;;
    __gpu_resume_smoke) gpu_run resume-smoke ;;
    *) die "usage: bash $SELF [prepare|preflight|smoke|long-gate CONFIG|stability|stability-resume CHECKPOINT|all|train|resume|resume-smoke|status|follow]" ;;
esac
