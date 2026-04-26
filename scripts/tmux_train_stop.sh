#!/usr/bin/env bash
set -euo pipefail

SESSION="smartrain-train"

usage() {
  cat <<'EOF'
Usage:
  tmux_train_stop.sh [--session NAME]

Sends Ctrl+C to the active pane of the selected tmux session.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      shift
      [[ $# -gt 0 ]] || { echo "Error: missing value for --session" >&2; usage; exit 2; }
      SESSION="$1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed. Install it first (e.g. sudo apt-get install -y tmux)." >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Error: tmux session '$SESSION' does not exist." >&2
  exit 1
fi

tmux send-keys -t "$SESSION" C-c
echo "Sent Ctrl+C to tmux session '$SESSION'."
echo "If needed, attach and verify: tmux attach -t $SESSION"
echo "To remove idle session: tmux kill-session -t $SESSION"
