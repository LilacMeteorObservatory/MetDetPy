import json
from types import SimpleNamespace
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

from MetLib import get_detector
from MetLib.collector import MeteorCollector, MeteorSeries
from MetLib.metstruct import MDTarget, MainDetectCfg
from MetLib.metvisu import TimeSeriesChartHandle, TimeSeriesChartVisu
from MetLib.packaging.common import post_process
from MetLib.packaging import pyinstaller
from MetLib.packaging.pyinstaller import create_merged_spec


def _legacy_target_dict() -> dict:
    return {
        "start_frame": 0,
        "start_time": "00:00:00.000",
        "end_time": "00:00:00.100",
        "last_activate_frame": 3,
        "last_activate_time": "00:00:00.100",
        "duration": 3,
        "speed": 1.0,
        "dist": 10.0,
        "fix_dist": 1.0,
        "fix_speed": 1.0,
        "fix_motion_duration": 0.1,
        "fix_duration": 0.1,
        "num_pts": 2,
        "category": "METEOR",
        "pt1": [0, 0],
        "pt2": [10, 10],
        "drct_loss": 0.0,
        "score": 0.8,
        "real_dist": 10.0,
    }


def test_legacy_target_without_drct_cv_uses_none():
    target = MDTarget.from_dict(_legacy_target_dict())
    assert target.drct_cv is None


def test_experimental_detector_entry_points_are_not_public():
    with pytest.raises(Exception, match="No class named BrightnessDetector"):
        get_detector("BrightnessDetector")

    config_data = json.loads(Path("config/m3det_normal.json").read_text(
        encoding="utf-8"))
    config_data["aux_detectors"] = [{"name": "BrightnessDetector"}]
    cfg = MainDetectCfg.from_dict(config_data)
    assert not hasattr(cfg, "aux_detectors")


def _make_series(start_frame: int) -> MeteorSeries:
    return MeteorSeries(start_frame=start_frame,
                        cur_frame=start_frame + 1,
                        init_pts=np.array([[0, 0], [10, 10], [5, 5]]),
                        max_acceptable_dist=2048,
                        max_acti_frame=30,
                        cate_prob=np.array([1.0]),
                        fps=30.0,
                        runtime_size=[960, 540])


def test_score_cache_follows_series_lifetime_and_count():
    collector = object.__new__(MeteorCollector)
    calls: list[int] = []

    def compute(series: MeteorSeries) -> float:
        calls.append(series.start_frame)
        return float(series.start_frame)

    collector._compute_score = compute
    first = _make_series(10)
    second = _make_series(20)

    assert collector.prob_meteor(first) == 10.0
    assert collector.prob_meteor(first) == 10.0
    assert collector.prob_meteor(second) == 20.0
    assert calls == [10, 20]

    first.count += 1
    assert collector.prob_meteor(first) == 10.0
    assert calls == [10, 20, 10]


def test_time_series_chart_caps_history_and_renders():
    handle = TimeSeriesChartHandle(name="score",
                                   corner="left-top",
                                   chart_w=200,
                                   chart_h=100,
                                   max_points=3)
    for value in (1.0, 2.0, 3.0, 4.0):
        handle.push(value)
    assert handle._buffer == [2.0, 3.0, 4.0]

    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    visu = TimeSeriesChartVisu(name="score",
                               current_value=4.0,
                               corner="left-top",
                               chart_w=200,
                               chart_h=100)
    rendered = handle.render(image, visu)
    assert np.any(rendered != 255)


def test_onefile_post_process_copies_resources_and_zips_all_artifacts(
        tmp_path: Path):
    source_root = tmp_path / "source"
    compile_path = tmp_path / "dist"
    compile_path.mkdir()
    for folder in ("config", "weights", "global"):
        resource_dir = source_root / folder
        resource_dir.mkdir(parents=True)
        (resource_dir / f"{folder}.txt").write_text(folder, encoding="utf-8")
    for executable in ("MetDetPy.exe", "ClipToolkit.exe", "MetDetPhoto.exe"):
        (compile_path / executable).write_bytes(b"executable")

    post_process(str(compile_path),
                 onefile_mode=True,
                 apply_zip=True,
                 source_root=str(source_root))

    assert (compile_path / "config" / "config.txt").is_file()
    assert (compile_path / "weights" / "weights.txt").is_file()
    assert (compile_path / "global" / "global.txt").is_file()

    zip_path = next(compile_path.glob("*.zip"))
    with zipfile.ZipFile(zip_path) as package:
        names = set(package.namelist())
    assert "MetDetPy.exe" in names
    assert "ClipToolkit.exe" in names
    assert "MetDetPhoto.exe" in names
    assert "config/config.txt" in names
    assert "weights/weights.txt" in names
    assert "global/global.txt" in names
    assert zip_path.name not in names


def test_generated_pyinstaller_spec_escapes_windows_paths(tmp_path: Path):
    work_path = tmp_path / "folder-with-backslash"
    work_path.mkdir()
    spec_path = Path(
        create_merged_spec(str(work_path), onefile=True, console=True))
    spec_source = spec_path.read_text(encoding="utf-8")

    assert repr(str(work_path)) in spec_source
    assert "MERGE(" not in spec_source
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(spec_source, str(spec_path), "exec")


def test_pyinstaller_build_uses_project_output_paths(tmp_path: Path,
                                                    monkeypatch):
    launcher = tmp_path / "make_package.py"
    launcher.write_text("", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def fake_run_cmd(command):
        captured["command"] = command
        return 0, 0.0

    monkeypatch.setattr(pyinstaller.sys, "argv", [str(launcher)])
    monkeypatch.setattr(pyinstaller, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(pyinstaller, "post_process", lambda *args, **kwargs: None)

    args = SimpleNamespace(onefile=True,
                           windowed=False,
                           icon=None,
                           apply_upx=False,
                           apply_zip=False)
    pyinstaller.build(args)

    command = captured["command"]
    assert command[command.index("--distpath") + 1] == str(tmp_path / "dist")
    assert command[command.index("--workpath") + 1] == str(tmp_path / "build")
