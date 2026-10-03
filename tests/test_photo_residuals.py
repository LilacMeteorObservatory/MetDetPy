import json

import pytest

from tools.compare_photo import build_report, compare_records, main
from tools.compare_photo import analyze_perception as analyze


def test_residuals_relate_pixel_shift_to_size_class_and_score():
    small = ([0, 0, 10, 10], 'METEOR', '0.60')
    large = ([0, 0, 100, 100], 'BUGS', '0.80')
    report = compare_records(['small', 'large'], {'small': [small], 'large': [large]},
                             {'small': [([1, 0, 11, 10], 'METEOR', '0.59')],
                              'large': [([1, 0, 101, 100], 'BUGS', '0.85')]})
    residuals = report['residuals']
    assert residuals['all_pairs']['signed_score_delta']['mean'] == pytest.approx(0.02)
    assert residuals['all_pairs']['abs_score_delta']['p50'] == pytest.approx(0.03)
    small_stats = residuals['by_baseline_area_px2']['below_1024']
    large_stats = residuals['by_baseline_area_px2']['at_least_9216']
    assert small_stats['max_edge_delta_px']['mean'] == large_stats['max_edge_delta_px']['mean'] == 1
    assert small_stats['iou_loss']['mean'] > large_stats['iou_loss']['mean']
    assert small_stats['normalized_center_shift']['mean'] == pytest.approx(0.1)
    assert large_stats['normalized_center_shift']['mean'] == pytest.approx(0.01)
    assert residuals['by_baseline_score']['0.6-0.7']['pairs'] == 1
    assert residuals['by_baseline_class']['BUGS']['signed_score_delta']['mean'] == 0.05
    assert residuals['correlations']['iou_loss_vs_log10_baseline_area']['pearson_r'] == pytest.approx(-1)
    for histogram in residuals['histograms'].values():
        assert sum(histogram['counts']) == histogram['count'] == 2
    json.dumps(report, allow_nan=False)


def test_residual_population_includes_unchanged_pairs_and_separates_unmatched():
    a = ([0, 0, 10, 10], 'METEOR', '0.90')
    b = ([40, 40, 50, 50], 'BUGS', '0.40')
    report = compare_records(['a', 'b', 'c'], {'a': [a], 'b': [b]},
                             {'a': [a], 'c': [b]})
    residuals = report['residuals']
    assert residuals['all_pairs']['pairs'] == 1
    assert residuals['changed_pairs']['pairs'] == 0
    assert residuals['changed_pairs']['iou']['p95'] is None
    assert residuals['unmatched']['baseline']['by_class'] == {'BUGS': 1}
    assert residuals['unmatched']['candidate']['score']['mean'] == 0.4
    assert residuals['correlations']['abs_score_delta_vs_baseline_score']['pearson_r'] is None


def test_combined_report_preserves_perceptual_matching_and_thresholds():
    a = {'a': [([0, 0, 10, 10], 'METEOR', '0.50'),
               ([20, 20, 30, 30], 'BUGS', '0.60')]}
    b = {'a': [([0, 0, 10, 10], 'METEOR', '0.49'),
               ([20, 20, 30, 30], 'BUGS', '0.65')]}
    combined = build_report(['a'], a, b)
    perception = dict(combined['perception'])
    assert perception.pop('thresholds') == dict(score_min='0.5', strict_iou_gt=0.9,
                                               strict_score_delta_lt='0.05')
    assert perception == analyze(['a'], a, b)
    assert perception['summary']['consistent_pairs'] == 0
    assert combined['summary']['matched_boxes'] == 2


def test_unified_entrypoint_writes_combined_report(tmp_path):
    image = tmp_path/'a.png'
    image.write_bytes(b'not decoded by report tools')
    manifest = tmp_path/'images.txt'
    manifest.write_text('a.png\n', encoding='utf-8')
    source = tmp_path/'source.json'
    source.write_text(json.dumps(dict(type='image-prediction', results=[
        dict(img_filename=str(image), boxes=[[0, 0, 10, 10]],
             preds=['METEOR'], prob=['0.60'])])), encoding='utf-8')
    output = tmp_path/'combined.json'
    main([str(manifest), str(source), str(source), '--output', str(output)])
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['perception']['summary']['consistent_pairs'] == 1
    assert report['residuals']['all_pairs']['pairs'] == 1
    assert report['summary']['exact_saved_result_rate'] == 1
    assert len(report['plots']) == 4
    from pathlib import Path
    for paths in report['plots'].values():
        assert set(paths) == {'png'}
        assert Path(paths['png']).read_bytes().startswith(b'\x89PNG')


def test_empty_and_zero_area_statistics_are_json_safe():
    empty = build_report([], {}, {})
    assert empty['perception']['summary']['different_image_rate'] is None
    zero = ([1, 1, 1, 1], 'METEOR', '1.00')
    report = build_report(['a'], {'a': [zero]}, {'a': [zero]})
    assert report['residuals']['all_pairs']['max_normalized_edge_delta']['count'] == 0
    assert report['residuals']['by_baseline_score']['0.9-1.0']['pairs'] == 1
    json.dumps(report, allow_nan=False)
