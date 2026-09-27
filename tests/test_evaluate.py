from types import SimpleNamespace

import pytest

import evaluate
from MetLib.metstruct import SingleMDRecord


def target(start=0, end=10, score=0.9):
    item = SimpleNamespace(start_frame=start, last_activate_frame=end,
                           start_time=f"{start:06d}", end_time=f"{end:06d}",
                           pt1=[1, 2], pt2=[8, 9], category="METEOR",
                           score=score)
    item.to_dict = lambda: dict(pt1=item.pt1, pt2=item.pt2)
    return item


def report(items, kind="prediction"):
    return SimpleNamespace(
        type=kind, anno_size=[10, 10],
        results=[SingleMDRecord("0", "1", [10, 10], items)])


def compare(monkeypatch, baseline, predictions):
    captured = {}
    monkeypatch.setattr("pprint.pprint", lambda stats: captured.update(stats))
    monkeypatch.setattr(evaluate, "print_confusion_matrix",
                        lambda matrix, labels: captured.update(matrix=matrix.copy()))
    evaluate.compare(SimpleNamespace(size=[10, 10]), baseline, predictions)
    return captured


@pytest.mark.parametrize("base_count,new_count", [(0, 0), (0, 2), (1, 0)])
def test_compare_empty_results(monkeypatch, base_count, new_count):
    stats = compare(monkeypatch, report([target()] * base_count),
                    report([target()] * new_count))
    assert stats["matched_num"] == 0
    assert stats["fn_num"] == base_count
    assert stats["lost_num"] == base_count
    assert stats["added_num"] == new_count
    assert stats["category_changed_num"] == 0
    assert stats["matrix"][:, -1].sum() == new_count
    assert stats["cross_ratio(A n B / A u B)"] == (
        1.0 if base_count == new_count == 0 else 0.0)


def test_compare_counts_all_tail_predictions(monkeypatch):
    stats = compare(monkeypatch, report([target()]),
                    report([target(), target(20, 30), target(40, 50)]))
    assert stats["matched_num"] == 1
    assert stats["matrix"][evaluate.NAME2ID["METEOR"], -1] == 2
    assert stats["new_predict_num"] == 3
    assert stats["added_num"] == 2
    assert stats["lost_num"] == 0


def test_normalization_returns_independent_targets():
    original = target()
    source = report([original])
    video = SimpleNamespace(size=[20, 30])
    first = evaluate.get_regularized_results(source, video)
    second = evaluate.get_regularized_results(source, video)
    assert first[0].pt1 == second[0].pt1 == [2, 6]
    assert first[0].pt2 == second[0].pt2 == [16, 27]
    assert original.pt1 == [1, 2]
    assert original.pt2 == [8, 9]
    first[0].category = "DROPPED"
    assert original.category == "METEOR"


def test_low_score_candidates_remain_in_total(monkeypatch):
    stats = compare(monkeypatch, report([], "annotation"),
                    report([target(score=0.2), target(score=0.8)]))
    assert stats["new_predict_num"] == 2
    assert stats["low_score_predict_num"] == 1
    assert stats["matrix"][:, -1].sum() == 1


def test_classification_errors_count_as_fp_and_fn():
    wrong = target()
    wrong.category = "RED_SPRITE"
    result = evaluate.calculate_detection_metrics([target()], [wrong])
    metrics = result["metrics"]
    assert metrics["matched_num"] == 1
    assert metrics["per_class"]["METEOR"]["fn"] == 1
    assert metrics["per_class"]["RED_SPRITE"]["fp"] == 1
    assert metrics["micro"]["tp"] == 0
    assert metrics["micro"]["f1"] == 0


def test_metrics_and_pr_curve_include_low_score_valid_candidates():
    result = evaluate.calculate_detection_metrics(
        [target(), target(20, 30)],
        [target(score=0.9), target(20, 30, score=0.2), target(40, 50, score=0.8)])
    metrics = result["metrics"]["micro"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 1)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert result["pr_curve"][0]["evaluated_predict_num"] == 0
    assert result["pr_curve"][-1]["micro"]["recall"] == 1.0
    assert result["pr_curve"][-1]["micro"]["precision"] == pytest.approx(2 / 3)
    assert result["candidate_num"] == 3


def test_metrics_match_high_score_first_and_only_once():
    result = evaluate.calculate_detection_metrics(
        [target()], [target(score=0.6), target(score=0.9)])
    assert result["metrics"]["micro"]["tp"] == 1
    assert result["metrics"]["micro"]["fp"] == 1
    assert result["metrics"]["micro"]["f1"] == pytest.approx(2 / 3)


def test_metrics_exclude_dropped_and_low_score_annotations():
    dropped = target(score=0.8)
    dropped.category = "DROPPED"
    result = evaluate.calculate_detection_metrics(
        [target(score=0.2)], [dropped])
    assert result["reference_num"] == 0
    assert result["candidate_num"] == result["dropped_predict_num"] == 1
    assert result["metrics"]["evaluated_predict_num"] == 0
    assert "DROPPED" not in result["metrics"]["per_class"]


@pytest.mark.parametrize("baseline,predictions", [([], []), ([target()], []), ([], [target()])])
def test_metrics_empty_inputs_are_json_serializable(baseline, predictions):
    import json
    result = evaluate.calculate_detection_metrics(baseline, predictions, gt_mode=False)
    json.dumps(result, allow_nan=False)
    assert result["mode"] == "baseline_consistency"
    assert result["metrics"]["micro"]["f1"] == 0.0


def test_monitor_uses_cpu_time_and_records_memory(monkeypatch):
    cpu_times = iter([SimpleNamespace(user=1, system=2),
                      SimpleNamespace(user=2, system=3)])
    rss = iter([10 * 1024 ** 2, 12 * 1024 ** 2])
    process = SimpleNamespace(cpu_times=lambda: next(cpu_times),
                              memory_info=lambda: SimpleNamespace(rss=next(rss)))
    monkeypatch.setattr(evaluate.psutil, "Process", lambda: process)
    monkeypatch.setattr(evaluate.os, "cpu_count", lambda: 4)
    clock = iter([100, 104])
    monkeypatch.setattr(evaluate.time, "perf_counter", lambda: next(clock))
    stats, result = evaluate.monitor_performance(lambda: 42, [], {}, interval=60)
    assert result == 42
    assert stats["tot_time"] == 4
    assert stats["cpu_time"] == 2
    assert stats["avg_cpu_usage"] == 12.5
    assert stats["peak_mem_usage"] == 12
    assert stats["mem_growth"] == 2
    assert stats["mem_sample_count"] == 2


def test_monitor_propagates_errors_and_stops_sampler():
    def fail():
        raise RuntimeError("test error")
    with pytest.raises(RuntimeError, match="test error"):
        evaluate.monitor_performance(fail, [], {}, interval=60)


def test_manifest_resolves_paths_and_selects_cases(tmp_path):
    import json
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps(dict(cases=[
        dict(id="night", json="night.json", cfg="config.json"),
        dict(id="noise", json="noise.json")])))
    cases = evaluate.load_cases(manifest, ["night"])
    assert len(cases) == 1
    assert cases[0]["json"] == str(tmp_path / "night.json")
    assert cases[0]["cfg"] == str(tmp_path / "config.json")
    with pytest.raises(ValueError, match="unknown case"):
        evaluate.load_cases(manifest, ["missing"])
    manifest.write_text(json.dumps(dict(cases=[dict(id="x", json="a"),
                                             dict(id="x", json="b")])))
    with pytest.raises(ValueError, match="unique"):
        evaluate.load_cases(manifest)


def test_batch_isolates_passes_and_aggregates(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        destination = Path(command[command.index('--run-summary') + 1])
        destination.write_text(json.dumps(dict(performance=dict(
            tot_time=len(calls), cpu_time=1, avg_cpu_usage=50, peak_mem_usage=12),
            metrics=None)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    args = SimpleNamespace(manifest=None, report=str(tmp_path / "input.json"),
                           case=[], passes=2, output_dir=str(tmp_path / "out"),
                           cfg="config/m3det_normal.json", metric=True, debug=False)
    evaluate.run_batch(args)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert all('--metrics-path' in command for command in calls)
    assert all('--report' in command for command in calls)
    assert summary["cases"][0]["performance_summary"]["tot_time"]["median"] == 1.5


def test_batch_records_failure_and_continues(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(evaluate.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=2))
    args = SimpleNamespace(manifest=None, report="input.json", case=[], passes=2,
                           output_dir=str(tmp_path), cfg="config/m3det_normal.json",
                           metric=False, debug=False)
    with pytest.raises(SystemExit):
        evaluate.run_batch(args)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["cases"][0]["runs"]) == 2
    assert all(run["status"] == "failed" for run in summary["cases"][0]["runs"])


def test_comparison_summary_class_changes_and_unmatched_are_distinct(monkeypatch):
    baseline = report([target(), target(20, 30)])
    predictions = report([target(), target(40, 50)])
    baseline.results[0].target[0].category = "RED_SPRITE"
    # 隔离差异 MDRF 导出；本测试关注匹配和摘要。
    monkeypatch.setattr(SingleMDRecord, "from_target", lambda *args: None)
    stats = compare(monkeypatch, baseline, predictions)
    assert stats["matched_num"] == 1
    assert stats["added_num"] == stats["lost_num"] == 1
    assert stats["category_changed_num"] == 1
    assert stats["category_transitions"] == [dict(
        from_category="RED_SPRITE", to_category="METEOR", count=1)]


def test_compare_exports_summary_without_changing_return_type(monkeypatch):
    monkeypatch.setattr(evaluate, "print_confusion_matrix", lambda *args: None)
    summary = {}
    result = evaluate.compare(SimpleNamespace(size=[10, 10]), report([]),
                              report([target()]), summary_out=summary)
    assert result.results == []
    assert summary == dict(added_num=1, lost_num=0, category_changed_num=0,
                           category_transitions=[])


@pytest.mark.parametrize("arguments", [[], ['--report', 'a', '--manifest', 'b'],
                                      ['a.json'], ['--report', 'a', '--batch'],
                                      ['--report', 'a', '--case', 'night']])
def test_cli_rejects_ambiguous_or_legacy_inputs(monkeypatch, arguments):
    monkeypatch.setattr(evaluate.sys, 'argv', ['evaluate.py', *arguments])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [
    ['--manifest', 'cases.json', '--case', 'night'],
    ['--report', 'report.json', '--passes', '2']])
def test_cli_dispatches_manifest_and_repeat_runs(monkeypatch, arguments):
    captured = []
    monkeypatch.setattr(evaluate.sys, 'argv', ['evaluate.py', *arguments])
    monkeypatch.setattr(evaluate, 'run_batch', captured.append)
    evaluate.main()
    assert len(captured) == 1
    assert captured[0].manifest == ('cases.json' if '--manifest' in arguments else None)
