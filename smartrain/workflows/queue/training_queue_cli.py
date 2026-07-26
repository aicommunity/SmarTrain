#!/usr/bin/env python3
"""
CLI for managing the training queue file (training_queue.txt).
"""
import argparse
import os
import sys
import tempfile
import shutil

import smartrain.workflows.queue.training_queue as tq
from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.file_lock import locked_file


def _queue_and_status(args):
    return tq.resolve_queue_status_paths(
        getattr(args, "queue_file", None),
        getattr(args, "workspace", None),
        getattr(args, "status_file", None),
    )


def _queue_path(args):
    q, _ = _queue_and_status(args)
    return q


def _with_file_lock(queue_path, fn):
    try:
        with locked_file(queue_path):
            return fn()
    except OSError as exc:
        print(f"[ERROR] Failed to lock queue file: {exc}", file=sys.stderr)
        sys.exit(1)


def _rebuild_queue_file(queue_path, new_tasks):
    """Does not reliably store comments and empty lines between task blocks; store the comment prefix before the first task."""
    raw_lines = tq.read_txt(queue_path)
    prefix = []
    i = 0
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            prefix.append(line)
            i += 1
        else:
            break
    suffix_lines = raw_lines[i + len(tq.get_queue_tasks(queue_path)) :]
    d = os.path.dirname(os.path.abspath(queue_path)) or "."
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=d) as tmp:
        tmp.writelines(prefix)
        for t in new_tasks:
            tmp.write(t + "\n")
        tmp.writelines(suffix_lines)
        tmp_path = tmp.name
    shutil.move(tmp_path, queue_path)


def cmd_list(args):
    path, st_file = _queue_and_status(args)
    if not os.path.exists(path):
        print("(queue file missing)")
        return
    tasks = tq.get_queue_tasks(path)
    statuses = {}
    if os.path.exists(st_file):
        with open(st_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if " | " in line:
                    task, st = line.split(" | ", 1)
                    statuses[task.strip()] = st.strip()
    for i, task in enumerate(tasks, start=1):
        st = statuses.get(task, "—")
        print(f"{i:3d}  [{st}]  {task}")


def cmd_add(args):
    path = _queue_path(args)
    line = " ".join(args.command).strip() if isinstance(args.command, list) else args.command.strip()
    if not line or line.startswith("#"):
        print("[ERROR] Empty line or comment.", file=sys.stderr)
        sys.exit(1)

    def do_add():
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(f"[OK] Added: {line}")

    _with_file_lock(path, do_add)


def cmd_remove(args):
    path = _queue_path(args)

    def do_remove():
        if not os.path.exists(path):
            print("[ERROR] Queue file not found.", file=sys.stderr)
            sys.exit(1)
        tasks = tq.get_queue_tasks(path)
        if args.index is not None:
            idx = args.index - 1
            if idx < 0 or idx >= len(tasks):
                print(f"[ERROR] There is no line number {args.index}.", file=sys.stderr)
                sys.exit(1)
            new_tasks = tasks[:idx] + tasks[idx + 1 :]
            removed = tasks[idx]
        else:
            sub = args.substring
            matching = [t for t in tasks if sub in t]
            if not matching:
                print(f"[ERROR] No line contains: {sub!r}", file=sys.stderr)
                sys.exit(1)
            if len(matching) > 1 and not args.all:
                print(f"[ERROR] Lines matched: {len(matching)}. Please clarify or use --all.", file=sys.stderr)
                sys.exit(1)
            new_tasks = [t for t in tasks if sub not in t]
            removed = sub

        raw_lines = tq.read_txt(path)
        prefix = []
        i = 0
        for line in raw_lines:
            s = line.strip()
            if not s or s.startswith("#"):
                prefix.append(line)
                i += 1
            else:
                break
        old_task_count = len(tasks)
        tail_start = i + old_task_count
        suffix_lines = raw_lines[tail_start:] if tail_start <= len(raw_lines) else []

        d = os.path.dirname(os.path.abspath(path)) or "."
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=d) as tmp:
            tmp.writelines(prefix)
            for t in new_tasks:
                tmp.write(t + "\n")
            tmp.writelines(suffix_lines)
            tmp_path = tmp.name
        shutil.move(tmp_path, path)
        if args.index is not None:
            print(f"[OK] Removed: {removed}")
        else:
            print(f"[OK] Removed lines with substring {args.substring!r}: {len(matching)}")

    _with_file_lock(path, do_remove)


def cmd_clear(args):
    path = _queue_path(args)

    def do_clear():
        if os.path.exists(path):
            raw = tq.read_txt(path)
            prefix = []
            for line in raw:
                s = line.strip()
                if not s or s.startswith("#"):
                    prefix.append(line)
                else:
                    break
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(prefix)
        print("[OK] The task queue has been cleared (the comments at the beginning of the file have been saved).")

    _with_file_lock(path, do_clear)


def cmd_run(args):
    qpath, stpath = _queue_and_status(args)
    tq.run_queue(
        no_terminal=args.no_gui,
        cwd=args.cwd,
        queue_path=qpath,
        status_file=stpath,
        max_retries=int(getattr(args, "max_retries", 0) or 0),
        retry_backoff_sec=float(getattr(args, "retry_backoff_sec", 30.0) or 0.0),
        retry_exit_codes=tq._parse_retry_exit_codes(getattr(args, "retry_exit_codes", "1")),
    )


def build_queue_cli_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root: queue queue.txt, statuses tmp/status.txt (aka SMART_TRAIN_WORKSPACE)",
    )
    common.add_argument(
        "--queue-file",
        type=str,
        default=None,
        help="Explicit path to queue file (overrides --workspace)",
    )
    common.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Explicit path to status.txt of the artist",
    )

    parser = CliArgumentParser(description="Learning Queue Management")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", parents=[common], help="Show tasks and statuses")
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", parents=[common], help="Add command to end of queue")
    p_add.add_argument("command", nargs=argparse.REMAINDER, help="Command string (as in training_queue.txt)")
    p_add.set_defaults(func=cmd_add)

    p_rem = sub.add_parser("remove", parents=[common], help="Delete task")
    g = p_rem.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, metavar="N", help="Line number (as in list, starting with 1)")
    g.add_argument("--substring", type=str, help="Remove lines containing substring")
    p_rem.add_argument("--all", action="store_true", help="With --substring: remove all matches")
    p_rem.set_defaults(func=cmd_remove)

    p_clear = sub.add_parser("clear", parents=[common], help="Delete all tasks")
    p_clear.set_defaults(func=cmd_clear)

    p_run = sub.add_parser("run", parents=[common], help="Start queue executor")
    p_run.add_argument("--no-gui", action="store_true", help="No gnome-terminal")
    p_run.add_argument("--cwd", type=str, default=None, help="Working directory for subprocess")
    p_run.add_argument("--max-retries", type=int, default=0, help="Extra retries (default 0).")
    p_run.add_argument("--retry-backoff-sec", type=float, default=30.0, help="Base backoff seconds.")
    p_run.add_argument("--retry-exit-codes", type=str, default="1", help="CSV of retryable exit codes.")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_queue_cli_arg_parser()
    args = parser.parse_args(argv)
    if args.cmd == "add" and not args.command:
        parser.error("add: specify the command after add")
    args.func(args)


if __name__ == "__main__":
    main()
