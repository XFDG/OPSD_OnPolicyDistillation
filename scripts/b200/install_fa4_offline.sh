#!/usr/bin/env bash
# Recreate the FA4 layer on the copied B200 venv using already synchronized artifacts.
# Run only while no OPSD training/smoke owns the launcher lock.
set -Eeuo pipefail
ROOT=/volume/pt-test/users/zhaoye
RUNTIME=$ROOT/OPSD-B200-runtime
REPO=$ROOT/OPSD_B200
PY=$ROOT/envs/opsd-py312-cu128/bin/python
SOURCE=$RUNTIME/src/flash-attention-official
WHEELS=$RUNTIME/migration/fa4-wheels
[[ $(hostname) == zhaoye-taihua-gpu-0 ]]
[[ $(findmnt -n -o FSTYPE -T "$ROOT") == gpfs ]]
mkdir -p "$RUNTIME/locks" "$ROOT/tmp" "$RUNTIME/cache/pip" "$RUNTIME/cache/xdg"
exec 9>"$RUNTIME/locks/full.lock"
flock -n 9
bash "$REPO/scripts/b200/keepalive.sh" start
export TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp
export PIP_CACHE_DIR=$RUNTIME/cache/pip XDG_CACHE_HOME=$RUNTIME/cache/xdg
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
[[ $(git -C "$SOURCE" rev-parse HEAD) == e9cf2c1651d2303191eb40a739a3c135fda00999 ]]
[[ -z $(git -C "$SOURCE" status --porcelain --untracked-files=no) ]]
"$PY" -m pip install --no-index --no-deps --find-links "$WHEELS" \
  nvidia-cutlass-dsl==4.7.1 nvidia-cutlass-dsl-libs-base==4.7.1 \
  nvidia-cutlass-dsl-libs-core==4.7.1 nvidia-cutlass-dsl-libs-cu12==4.7.1 \
  nvidia-cuda-nvdisasm==13.4.92 quack-kernels==0.6.5 \
  torch-c-dlpack-ext==0.1.5 setuptools-scm==8.3.1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4=0.0.0+e9cf2c1
"$PY" -m pip wheel --no-index --no-deps --no-build-isolation \
  "$SOURCE/flash_attn/cute" -w "$WHEELS"
"$PY" -m pip install --no-index --no-deps --find-links "$WHEELS" flash-attn-4==0.0.0+e9cf2c1
"$PY" -B "$REPO/scripts/b200/apply_fa4.py"
"$PY" -m pip check
printf 'Official FA4 layer installed. Run run_b200_full.sh preflight and smoke before training.\n'
