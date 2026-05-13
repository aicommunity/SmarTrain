import os
import subprocess
import time
import argparse
from pathlib import Path

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.workspace_paths import (
    resolve_workspace_root,
    workspace_queue_path,
    workspace_queue_status_path,
)

# Keep default queue/status files under package root (`smartrain/`) after module move.
BASE_DIR = str(Path(__file__).resolve().parents[2])
QUEUE_TXT = os.path.join(BASE_DIR, "training_queue.txt")
STATUS_FILE = os.path.join(BASE_DIR, "tmp/status.txt")


def resolve_queue_status_paths(queue_file_cli, workspace_cli, status_file_cli):
    """
    Queue and status file.
    Precedence: explicit --queue-file; otherwise, if resolve workspace is successful - queue.txt in the workspace root;
    otherwise QUEUE_TXT next to the script.
    Status: explicit --status-file; otherwise tmp/status.txt next to the queue file (for workspace - workspace/tmp/status.txt).
    """
    if status_file_cli is not None and status_file_cli.strip():
        status_path = os.path.abspath(os.path.expanduser(status_file_cli.strip()))
    else:
        status_path = None

    if queue_file_cli is not None and queue_file_cli.strip():
        queue_path = os.path.abspath(os.path.expanduser(queue_file_cli.strip()))
        if status_path is None:
            status_path = os.path.join(os.path.dirname(queue_path), "tmp", "status.txt")
        return queue_path, status_path

    try:
        root = resolve_workspace_root(workspace_cli)
    except ValueError:
        queue_path = QUEUE_TXT
        if status_path is None:
            status_path = STATUS_FILE
        return queue_path, status_path

    queue_path = workspace_queue_path(root)
    if status_path is None:
        status_path = workspace_queue_status_path(root)
    return queue_path, status_path


def get_queue_tasks(queue_path=None):
    """Queue lines without \\n, without empty and without comments."""
    path = queue_path or QUEUE_TXT
    lines = read_txt(path)
    out = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def main_window(status_file: str) -> None:
    subprocess.Popen([
        "gnome-terminal", "--",
        "bash", "-c",
        f"watch -n 1 cat {status_file}; exec bash"
    ])


def update_status(index, status, tasks):
    """Update task status by index (order as in queue)."""
    if index < 0 or index >= len(tasks):
        return
    statuses = load_statuses()
    statuses[tasks[index]] = status
    save_statuses(tasks, statuses)


def start_new_process(cmd, cwd=None):
    work_dir = cwd if cwd is not None else os.getcwd()
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=work_dir,
    )
    return process.wait()


def read_txt(txt_file):
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.readlines()
    except Exception as e:
        print(f"[ERROR] Failed to open txt file: {e}")
        content = []
    return content


def process_line(line):
    try:
        s = line.strip()
        if not s or s.startswith("#"):
            return None
        arguments = s.split()
        first = arguments[0]
        if first == "smartrain" or first.endswith("/smartrain"):
            return s
        if first in ("python3", "python"):
            return s
        if not first.startswith("python3"):
            arguments.insert(0, "python3")
        if len(arguments) > 1 and not arguments[1].endswith(".py"):
            arguments[1] += ".py"
        return " ".join(arguments)
    except Exception as e:
        print(f"[ERROR] Error processing command: {e}")
        return None


def load_statuses():
    if not os.path.exists(STATUS_FILE):
        return {}
    statuses = {}
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " | " not in line:
                continue
            task, st = line.split(" | ", 1)
            statuses[task.strip()] = st.strip()
    return statuses


def save_statuses(tasks, statuses, status_file=None):
    """Writes status.txt in queue line order."""
    path = status_file or STATUS_FILE
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            st = statuses.get(t, "Waiting to be completed")
            if isinstance(st, str):
                st = st.strip()
            f.write(f"{t} | {st}\n")


def run_queue(no_terminal=False, cwd=None, queue_path=None, status_file=None):
    """
    Sequentially executes tasks from the queue.
    no_terminal: Don't open gnome-terminal.
    cwd: working directory for subprocess (current directory by default).
    """
    qpath = queue_path or QUEUE_TXT
    st_file = status_file or STATUS_FILE
    work_cwd = cwd if cwd is not None else os.getcwd()

    st_dir = os.path.dirname(st_file)
    if st_dir:
        os.makedirs(st_dir, exist_ok=True)

    if not no_terminal:
        main_window(st_file)

    def _load():
        if not os.path.exists(st_file):
            return {}
        statuses = {}
        with open(st_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or " | " not in line:
                    continue
                task, st = line.split(" | ", 1)
                statuses[task.strip()] = st.strip()
        return statuses

    statuses = _load()

    try:
        while True:
            tasks = get_queue_tasks(qpath)
            statuses = {t: statuses.get(t, "Waiting to be completed") for t in tasks}
            save_statuses(tasks, statuses, status_file=st_file)

            next_task = None
            for t in tasks:
                if statuses.get(t) == "Waiting to be completed":
                    next_task = t
                    break

            if next_task is None:
                time.sleep(5)
                continue

            statuses[next_task] = "Running"
            save_statuses(tasks, statuses, status_file=st_file)

            cmd = process_line(next_task)
            if cmd is None:
                statuses[next_task] = "Error"
                save_statuses(tasks, statuses, status_file=st_file)
                continue

            result = start_new_process(cmd, cwd=work_cwd)
            statuses[next_task] = "Done" if result == 0 else "Error"
            save_statuses(tasks, statuses, status_file=st_file)
    finally:
        if os.path.exists(st_file):
            os.remove(st_file)


def build_queue_run_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Training queue: sequentially running commands from training_queue.txt"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open gnome-terminal (status only in tmp/status.txt)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory to run commands (current directory by default)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root: queue workspace/queue.txt (aka SMART_TRAIN_WORKSPACE)",
    )
    parser.add_argument(
        "--queue-file",
        type=str,
        default=None,
        help="Explicit path to queue file (overrides --workspace)",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Explicit path to status.txt of the artist",
    )
    return parser


def main(argv=None):
    if argv is None:
        import sys
        argv = sys.argv[1:]
    args = build_queue_run_arg_parser().parse_args(argv)
    qpath, stpath = resolve_queue_status_paths(
        args.queue_file, args.workspace, args.status_file
    )
    run_queue(
        no_terminal=args.no_gui,
        cwd=args.cwd,
        queue_path=qpath,
        status_file=stpath,
    )


if __name__ == "__main__":
    main()
