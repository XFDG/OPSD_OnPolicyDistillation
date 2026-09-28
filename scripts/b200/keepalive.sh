#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/python3 -B "$(dirname "$0")/keepalive_ctl.py" "$@"
