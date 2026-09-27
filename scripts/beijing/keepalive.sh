#!/usr/bin/env bash
# Run on the Beijing GPU node: bash keepalive.sh status|start|stop.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Never pass a parent training lock or Python/distributed overlay to the
# controller. The controller also strips those variables from run.sh.
exec 9>&-
exec env -u PYTHONPATH -u PYTHONHOME -u PYTHON_EXEC -u PYTHONUSERBASE \
    PYTHONNOUSERSITE=1 "${OPSD_KEEPALIVE_CTL_PYTHON:-python3}" \
    "${script_dir}/keepalive_ctl.py" "$@"
