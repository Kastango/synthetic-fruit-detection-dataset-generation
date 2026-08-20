from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from .common import atomic_write_json

HEARTBEAT_SECONDS = 300


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"stages": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("stages", {})
    return state


def _eta(estimated_seconds: float | None, elapsed: float) -> str:
    if estimated_seconds is None:
        return "calculando; ainda não há histórico desta etapa"
    if elapsed >= estimated_seconds:
        return "acima do tempo histórico"
    return f"~{_duration(estimated_seconds - elapsed)}"


class StageHeartbeat:
    def __init__(
        self,
        title: str,
        command: list[str],
        state_path: Path,
        interval_seconds: float,
    ) -> None:
        self.title = title
        self.command = command
        self.state_path = state_path
        self.interval_seconds = interval_seconds
        self.state = _load_state(state_path)
        previous = self.state["stages"].get(title, {})
        self.estimated_seconds = previous.get("estimated_duration_seconds")
        self.started = 0.0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def _save_elapsed(self, elapsed: float) -> None:
        self.state["current"]["elapsed_seconds"] = round(elapsed, 2)
        atomic_write_json(self.state_path, self.state)

    def _report(self) -> None:
        while not self.stop.wait(self.interval_seconds):
            elapsed = time.monotonic() - self.started
            self._save_elapsed(elapsed)
            print(
                f"[em andamento] {self.title} | decorrido: {_duration(elapsed)} | "
                f"ETA aproximado: {_eta(self.estimated_seconds, elapsed)}",
                flush=True,
            )

    def __enter__(self) -> Self:
        self.started = time.monotonic()
        self.state["current"] = {
            "stage": self.title,
            "command": self.command,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": 0.0,
        }
        atomic_write_json(self.state_path, self.state)
        print(
            f"[início] {self.title} | "
            f"ETA aproximado: {_eta(self.estimated_seconds, 0)}",
            flush=True,
        )
        self.thread = threading.Thread(target=self._report, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        elapsed = time.monotonic() - self.started
        if exc_type is None:
            if self.estimated_seconds is None or elapsed >= 60:
                self.estimated_seconds = elapsed
            self.state["stages"][self.title] = {
                "last_duration_seconds": round(elapsed, 2),
                "estimated_duration_seconds": round(float(self.estimated_seconds), 2),
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            self.state.pop("last_failure", None)
            status = "concluída"
        else:
            self.state["last_failure"] = {
                "stage": self.title,
                "elapsed_seconds": round(elapsed, 2),
                "failed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            status = "falhou"
        self.state.pop("current", None)
        atomic_write_json(self.state_path, self.state)
        print(
            f"[{status}] {self.title} | duração: {_duration(elapsed)}",
            flush=True,
        )


def run_stage(
    title: str,
    command: list[str],
    *,
    cwd: Path,
    state_path: Path,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> None:
    with StageHeartbeat(title, command, state_path, heartbeat_seconds):
        subprocess.run(command, cwd=cwd, check=True)
