from decimal import Decimal

from tools.compare_photo import analyze_perception as analyze, consistent_matching


def test_strict_iou_and_decimal_score_difference():
    a = [([0, 0, 10, 10], 'METEOR', '0.60')]
    assert not consistent_matching(a, [([0, 0, 9, 10], 'METEOR', '0.60')], 0.9, Decimal('0.05'))
    assert not consistent_matching(a, [([0, 0, 10, 10], 'METEOR', '0.65')], 0.9, Decimal('0.05'))
    assert not consistent_matching(a, [([0, 0, 10, 10], 'BUGS', '0.60')], 0.9, Decimal('0.05'))
    assert consistent_matching(a, [([0, 0, 10, 10], 'METEOR', '0.64')], 0.9, Decimal('0.05')) == [(0, 0)]


def test_independent_filter_keeps_point_five():
    a = {'a': [([0, 0, 10, 10], 'METEOR', '0.50')]}
    b = {'a': [([0, 0, 10, 10], 'METEOR', '0.49')]}
    report = analyze(['a', 'empty'], a, b)
    assert report['summary']['baseline_kept'] == 1
    assert report['summary']['candidate_kept'] == 0
    assert report['summary']['baseline_inconsistent'] == 1
    assert report['summary']['both_empty_images'] == 1
    assert report['diagnostics']['candidate_below_threshold_pairs'] == 1


def test_matching_uses_augmenting_paths_instead_of_greedy_pairing():
    a = [([0, 0, 10, 10], 'METEOR', '0.80'), ([0, 0, 9, 10], 'METEOR', '0.80')]
    b = [([0, 0, 10, 10], 'METEOR', '0.80'), ([1, 0, 10, 10], 'METEOR', '0.80')]
    assert consistent_matching(a, b, 0.8, Decimal('0.05')) == [(0, 1), (1, 0)]
