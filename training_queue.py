import os
import subprocess
import time
import argparse

from workspace_paths import (
    resolve_workspace_root,
    workspace_queue_path,
    workspace_queue_status_path,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEUE_TXT = os.path.join(BASE_DIR, "training_queue.txt")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
STATUS_FILE = os.path.join(BASE_DIR, "tmp/status.txt")

os.makedirs(TMP_DIR, exist_ok=True)


def resolve_queue_status_paths(queue_file_cli, workspace_cli, status_file_cli):
    """
    Очередь и файл статусов.
    Приоритет: явный --queue-file; иначе при успешном resolve workspace — queue.txt в корне workspace;
    иначе QUEUE_TXT рядом со скриптом.
    Статус: явный --status-file; иначе tmp/status.txt рядом с файлом очереди (для workspace — workspace/tmp/status.txt).
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
    """Строки очереди без \\n, без пустых и без комментариев."""
    path = queue_path or QUEUE_TXT
    lines = read_txt(path)
    out = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def main_window():
    subprocess.Popen([
        "gnome-terminal", "--",
        "bash", "-c",
        f"watch -n 1 cat {STATUS_FILE}; exec bash"
    ])


def update_status(index, status, tasks):
    """Обновить статус задачи по индексу (порядок как в очереди)."""
    if index < 0 or index >= len(tasks):
        return
    statuses = load_statuses()
    statuses[tasks[index]] = status
    save_statuses(tasks, statuses)


def start_new_process(cmd, cwd=None):
    work_dir = cwd if cwd is not None else BASE_DIR
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
        print(f"[ERROR] Не удалось открыть txt-файл: {e}")
        content = []
    return content


def process_line(line):
    try:
        arguments = line.strip().split()

        if not arguments or arguments[0].startswith("#"):
            return None

        if not arguments[0].startswith("python3"):
            arguments.insert(0, "python3")

        if not arguments[1].endswith(".py"):
            arguments[1] += ".py"

        return " ".join(arguments)
    except Exception as e:
        print(f"[ERROR] Ошибка при обработке команды: {e}")
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
    """Пишет status.txt в порядке строк очереди."""
    path = status_file or STATUS_FILE
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            st = statuses.get(t, "Ждет выполнения")
            if isinstance(st, str):
                st = st.strip()
            f.write(f"{t} | {st}\n")


def run_queue(no_terminal=False, cwd=None, queue_path=None, status_file=None):
    """
    Последовательно выполняет задачи из очереди.
    no_terminal: не открывать gnome-terminal.
    cwd: рабочая директория для subprocess (по умолчанию BASE_DIR).
    """
    qpath = queue_path or QUEUE_TXT
    st_file = status_file or STATUS_FILE
    work_cwd = cwd if cwd is not None else BASE_DIR

    st_dir = os.path.dirname(st_file)
    if st_dir:
        os.makedirs(st_dir, exist_ok=True)

    if not no_terminal:
        main_window()

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
            statuses = {t: statuses.get(t, "Ждет выполнения") for t in tasks}
            save_statuses(tasks, statuses, status_file=st_file)

            next_task = None
            for t in tasks:
                if statuses.get(t) == "Ждет выполнения":
                    next_task = t
                    break

            if next_task is None:
                time.sleep(5)
                continue

            statuses[next_task] = "Выполняется"
            save_statuses(tasks, statuses, status_file=st_file)

            cmd = process_line(next_task)
            if cmd is None:
                statuses[next_task] = "Ошибка"
                save_statuses(tasks, statuses, status_file=st_file)
                continue

            result = start_new_process(cmd, cwd=work_cwd)
            statuses[next_task] = "Выполнено" if result == 0 else "Ошибка"
            save_statuses(tasks, statuses, status_file=st_file)
    finally:
        if os.path.exists(st_file):
            os.remove(st_file)


def main():
    parser = argparse.ArgumentParser(description="Очередь обучения: последовательный запуск команд из training_queue.txt")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Не открывать gnome-terminal (статус только в tmp/status.txt)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Рабочая директория для запуска команд (по умолчанию каталог скрипта)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Корень workspace: очередь workspace/queue.txt (иначе SMART_TRAIN_WORKSPACE)",
    )
    parser.add_argument(
        "--queue-file",
        type=str,
        default=None,
        help="Явный путь к файлу очереди (перекрывает --workspace)",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Явный путь к status.txt исполнителя",
    )
    args = parser.parse_args()
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
