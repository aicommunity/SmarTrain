import sys
import os
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEUE_TXT = os.path.join(BASE_DIR, "training_queue.txt")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
STATUS_FILE = os.path.join(BASE_DIR, "tmp/status.txt")

os.makedirs(TMP_DIR, exist_ok=True)


def main_window():
    subprocess.Popen([
        "gnome-terminal", "--",
        "bash", "-c",
        f"watch -n 1 cat {STATUS_FILE}; exec bash"
    ])


def update_status(index, status):
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    base_line = lines[index].split(" | ")[0]
    lines[index] = f"{base_line} | {status}\n"

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


def start_new_process(cmd):
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=BASE_DIR
    )

    result = process.wait()
    return result


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

        if len(arguments) < 1:
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
            if " | " in line:
                task, st = line.split(" | ")
                statuses[task] = st
    return statuses


def save_statuses(statuses):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        for task, status in statuses.items():
            f.write(f"{task} | {status}\n")     


def main():
    main_window()
    statuses = load_statuses()

    try:
        while True:
            tasks = read_txt(QUEUE_TXT)
            
            for t in tasks:
                if t not in statuses:
                    statuses[t] = "Ждет выполнения"
            save_statuses(statuses)

            next_task = None
            for t in tasks:
                if statuses.get(t) == "Ждет выполнения":
                    next_task = t
                    break

            if next_task is None:
                time.sleep(5)
                continue

            statuses[next_task] = "Выполняется"
            save_statuses(statuses)
            
            cmd = process_line(next_task)
            result = start_new_process(cmd)

            if result == 0:
                statuses[next_task] = "Выполнено"
            else:
                statuses[next_task] = "Ошибка"
            
            save_statuses(statuses)
    finally:
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)


if __name__ == "__main__":
    main()
