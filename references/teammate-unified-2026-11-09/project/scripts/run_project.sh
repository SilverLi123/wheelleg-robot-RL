#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/source/agent_tasks:$PROJECT_ROOT/source/agent_world:$PROJECT_ROOT/source/agent_rl:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ $# -eq 0 ]]; then
    echo 'Usage: WHEELBIPE_PYTHON=/path/to/python WHEELBIPE_RUN_ROOT=/persistent/runs scripts/run_project.sh scripts/FILE.py [args...]' >&2
    exit 2
fi
RUNTIME="${WHEELBIPE_PYTHON:-python3}"
RUN_ROOT="${WHEELBIPE_RUN_ROOT:-$PROJECT_ROOT/runs}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd -- "$RUN_ROOT" && pwd)"
export XDG_CACHE_HOME="${WHEELBIPE_CACHE_ROOT:-$RUN_ROOT/cache/sim51-lab230}"
export WHEELBIPE_DIAGNOSTICS_DIR="${WHEELBIPE_DIAGNOSTICS_DIR:-$RUN_ROOT/diagnostics}"
mkdir -p "$XDG_CACHE_HOME" "$WHEELBIPE_DIAGNOSTICS_DIR"
ENTRY="$1"
shift
if [[ "$ENTRY" != /* ]]; then ENTRY="$PROJECT_ROOT/$ENTRY"; fi
if [[ ! -f "$ENTRY" ]]; then echo "Missing script: $ENTRY" >&2; exit 2; fi
cd -- "$RUN_ROOT"
exec "$RUNTIME" "$ENTRY" "$@"
