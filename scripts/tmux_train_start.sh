#!/usr/bin/env bash
set -euo pipefail

SESSION="smartrain-train"

usage() {
  cat <<'EOF'
Usage:
  tmux_train_start.sh [--session NAME] -- <command> [args...]

Examples:
  ./scripts/tmux_train_start.sh --session smartrain-train -- smartrain train --data my_dataset --model yolo11n.pt --device 0
  ./scripts/tmux_train_start.sh -- bash -lc 'smartrain train --data my_dataset 2>&1 | tee -a runs/train.log'
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
    --)
      shift
      break
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Error: command is required after --" >&2
  usage
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed. Install it first (e.g. sudo apt-get install -y tmux)." >&2
  exit 1
fi

cmd_quoted=()
for arg in "$@"; do
  printf -v q "%q" "$arg"
  cmd_quoted+=("$q")
done
cmd_string="${cmd_quoted[*]}"

pwd_quoted=""
printf -v pwd_quoted "%q" "$PWD"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  window_name="train-$(date +%H%M%S)"
  tmux new-window -t "$SESSION" -n "$window_name" "cd $pwd_quoted && $cmd_string; exec bash"
  target="$SESSION:$window_name"
else
  tmux new-session -d -s "$SESSION" -n train "cd $pwd_quoted && $cmd_string; exec bash"
  target="$SESSION:train"
fi

echo "Started command in tmux target: $target"
echo "Attach: tmux attach -t $SESSION"
echo "Detach without stopping: Ctrl+B then D"
echo "Stop from attached session: Ctrl+C"
