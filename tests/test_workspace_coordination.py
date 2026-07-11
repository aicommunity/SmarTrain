"""Tests for multi-user workspace coordination."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from smartrain.core.runtime.workspace_coordination import (
    LOCK_SCAN,
    ScanLockBusy,
    WorkspaceSession,
    classify_command,
    list_active_peers,
    lock_path_for,
    reconcile_local_holders,
)
from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.datasets.dataset_cli_common import update_datasets_sidecar
from smartrain.services.datasets.dataset_scan_preflight import maybe_run_auto_scan


@pytest.fixture
def layout(tmp_path: Path) -> WorkspaceLayout:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "datasets").mkdir()
    (root / "raw_data").mkdir()
    os.environ["SMART_TRAIN_WORKSPACE"] = str(root)
    return WorkspaceLayout(str(root))


def test_reconcile_removes_stale_local_lock(layout: WorkspaceLayout) -> None:
    lock = lock_path_for(layout, LOCK_SCAN)
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": "stale",
        "hostname": __import__("socket").gethostname(),
        "pid": 999999999,
        "command": "smartrain scan",
        "acquired_at": "2000-01-01T00:00:00Z",
        "last_heartbeat": "2000-01-01T00:00:00Z",
    }
    lock.write_text(json.dumps(payload), encoding="utf-8")
    removed = reconcile_local_holders(layout)
    assert removed >= 1
    assert not lock.is_file()


def test_scan_lock_exclusive(layout: WorkspaceLayout) -> None:
    with WorkspaceSession(layout, ["scan"], register_peer=False, warn_peers=False) as s1:
        s1.acquire_lock(LOCK_SCAN, blocking=False, timeout_sec=0)
        with WorkspaceSession(layout, ["scan"], register_peer=False, warn_peers=False) as s2:
            with pytest.raises(ScanLockBusy):
                s2.acquire_lock(LOCK_SCAN, blocking=False, timeout_sec=0)


def test_workspace_session_releases_lock_on_exit(layout: WorkspaceLayout) -> None:
    with WorkspaceSession(layout, ["train"], register_peer=False, warn_peers=False) as session:
        session.acquire_lock(LOCK_SCAN, blocking=False, timeout_sec=0)
        assert lock_path_for(layout, LOCK_SCAN).is_file()
    assert not lock_path_for(layout, LOCK_SCAN).is_file()


def test_classify_explicit_scan_policy() -> None:
    policy = classify_command([], prog="scan")
    assert policy.explicit_scan is True
    assert policy.catalog_lock is False


def test_classify_merge_catalog_lock() -> None:
    policy = classify_command(["--data", "ds"], prog="merge")
    assert policy.catalog_lock is True
    assert policy.explicit_scan is False


def test_maybe_run_auto_scan_skips_when_scan_lock_busy(layout: WorkspaceLayout, monkeypatch) -> None:
    lock = lock_path_for(layout, LOCK_SCAN)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps(
            {
                "hostname": "other-host",
                "pid": 1,
                "command": "smartrain scan",
                "last_heartbeat": "2099-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    called = {"n": 0}

    def _fake_scan(argv):
        called["n"] += 1

    monkeypatch.setattr(
        "smartrain.services.datasets.dataset_scan_preflight.scan_main",
        _fake_scan,
    )
    t0 = time.monotonic()
    maybe_run_auto_scan(
        ["--workspace", layout.root],
        ensure_scan=True,
        auto_scan_disabled=False,
    )
    assert time.monotonic() - t0 < 2.0
    assert called["n"] == 0


def test_update_datasets_sidecar_concurrent(layout: WorkspaceLayout) -> None:
    target = Path(layout.datasets) / "ds_a"
    target.mkdir()
    errors: list[str] = []

    def worker(idx: int) -> None:
        try:
            d = Path(layout.datasets) / f"ds_{idx}"
            d.mkdir(exist_ok=True)
            update_datasets_sidecar(
                layout=layout,
                output_key=f"ds_{idx}",
                class_map={"a": 0},
                target_dir=str(d),
                output_hash=f"hash{idx}",
            )
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    catalog = json.loads(Path(layout.work_datasets_info_path()).read_text(encoding="utf-8"))
    assert len(catalog) == 4


def test_list_active_peers(layout: WorkspaceLayout) -> None:
    with WorkspaceSession(layout, ["train --data ds"], register_peer=True, warn_peers=False):
        peers = list_active_peers(layout)
        assert len(peers) == 1
        assert peers[0]["command"].startswith("smartrain")


def test_signal_handler_releases_lock(layout: WorkspaceLayout) -> None:
  if sys.platform == "win32":
      pytest.skip("SIGINT handler test is Unix-specific")
  proc = subprocess.Popen(
      [
          sys.executable,
          "-c",
          (
              "import os, signal, time; "
              "os.environ['SMART_TRAIN_WORKSPACE']=os.environ['WS']; "
              "from smartrain.core.runtime.workspace_paths import WorkspaceLayout; "
              "from smartrain.core.runtime.workspace_coordination import WorkspaceSession, LOCK_SCAN; "
              "layout=WorkspaceLayout(os.environ['WS']); "
              "s=WorkspaceSession(layout, ['hold']); "
              "s.__enter__(); "
              "s.acquire_lock(LOCK_SCAN, blocking=False, timeout_sec=0); "
              "time.sleep(30)"
          ),
      ],
      env={**os.environ, "WS": layout.root},
  )
  time.sleep(0.5)
  proc.send_signal(signal.SIGINT)
  proc.wait(timeout=5)
  assert proc.returncode != 0
  assert not lock_path_for(layout, LOCK_SCAN).is_file()
