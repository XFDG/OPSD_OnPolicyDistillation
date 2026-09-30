#!/usr/bin/env bash
# Manual B200 optimization entrypoint. Default mode is preparation only.
set -Eeuo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_b200_optimized_full.sh" "$@"
