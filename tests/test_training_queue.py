"""Tests for training queue command parsing and subprocess execution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from smartrain.workflows.queue import training_queue as tq


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", None),
        ("# comment", None),
        ("smartrain train --data ds -y", ["smartrain", "train", "--data", "ds", "-y"]),
        ("/usr/bin/smartrain scan", ["/usr/bin/smartrain", "scan"]),
        ("python3 script.py --flag", ["python3", "script.py", "--flag"]),
        ("train.py --epochs 1", ["python3", "train.py", "--epochs", "1"]),
    ],
)
def test_process_line_parses_to_argv(line: str, expected: list[str] | None) -> None:
    assert tq.process_line(line) == expected


def test_start_new_process_uses_shell_false() -> None:
    with patch("smartrain.workflows.queue.training_queue.subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        code = tq.start_new_process(["echo", "ok"])
    assert code == 0
    popen.assert_called_once()
    kwargs = popen.call_args.kwargs
    assert kwargs["shell"] is False
    assert popen.call_args.args[0] == ["echo", "ok"]


def test_get_queue_tasks_skips_comments_and_blanks(tmp_path) -> None:
    queue_file = tmp_path / "queue.txt"
    queue_file.write_text("# header\n\nsmartrain scan\n\n", encoding="utf-8")
    assert tq.get_queue_tasks(str(queue_file)) == ["smartrain scan"]
