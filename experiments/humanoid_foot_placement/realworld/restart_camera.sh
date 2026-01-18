#!/usr/bin/env bash
set -euo pipefail

USER="booster"
HOST="192.168.10.102"
# path to the remote script relative to the remote user's home, or absolute path
REMOTE_SCRIPT="Workspace/Puze/realsense/restart_camera.sh"

# allocate a tty (-t) so interactive prompts (ssh password, sudo) work;
# use bash -lc on the remote side so `source` runs in a bash login-like shell.
ssh -t "${USER}@${HOST}" "bash -lc 'if [ -f \"${REMOTE_SCRIPT}\" ]; then source \"${REMOTE_SCRIPT}\"; else echo \"Remote script not found: ${REMOTE_SCRIPT}\" >&2; exit 2; fi'"