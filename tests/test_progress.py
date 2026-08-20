from __future__ import annotations

import json
import sys

from fruit_pipeline.progress import run_stage


def test_run_stage_reports_heartbeat_and_records_duration(tmp_path, capsys) -> None:
    state_path = tmp_path / "pipeline_state.json"
    run_stage(
        "Etapa de teste",
        [sys.executable, "-c", "import time; time.sleep(0.08)"],
        cwd=tmp_path,
        state_path=state_path,
        heartbeat_seconds=0.02,
    )

    output = capsys.readouterr().out
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "[início] Etapa de teste" in output
    assert "[em andamento] Etapa de teste" in output
    assert "[concluída] Etapa de teste" in output
    assert "ETA aproximado" in output
    assert "current" not in state
    assert state["stages"]["Etapa de teste"]["last_duration_seconds"] > 0


def test_run_stage_uses_previous_duration_for_eta(tmp_path, capsys) -> None:
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(
        json.dumps(
            {"stages": {"Etapa conhecida": {"estimated_duration_seconds": 120.0}}}
        ),
        encoding="utf-8",
    )

    run_stage(
        "Etapa conhecida",
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        state_path=state_path,
        heartbeat_seconds=1,
    )

    assert "ETA aproximado: ~00:02:00" in capsys.readouterr().out
