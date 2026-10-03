import json

import cv2
import numpy as np
import pytest

from tools.compare_photo import build_report, compare_records
from tools.export_photo_diff import export_report, selected_groups
from tools.compare_photo import analyze_perception as analyze


def test_group_selection_includes_equal_counts_with_unmatched_boxes():
    assert selected_groups(dict(baseline=[1], candidate=[2],
                                baseline_unmatched=[0], candidate_unmatched=[0])) == ['unmatched']
    assert selected_groups(dict(baseline=[], candidate=[1],
                                baseline_unmatched=[], candidate_unmatched=[0])) == ['count_changed', 'unmatched']


def test_export_overlap_same_basename_and_prediction_delta(tmp_path):
    sources = []
    for directory in ('one', 'two'):
        folder = tmp_path/directory
        folder.mkdir()
        source = folder/'image.png'
        ok, encoded = cv2.imencode('.png', np.zeros((100, 120, 3), np.uint8))
        assert ok
        source.write_bytes(encoded.tobytes())
        sources.append(str(source))
    a = ([10, 10, 30, 30], 'METEOR', '0.90')
    b = ([11, 10, 30, 30], 'METEOR', '0.89')
    c = ([60, 60, 80, 80], 'BUGS', '0.80')
    report = compare_records(sources, {sources[0]: [a], sources[1]: [a]},
                             {sources[0]: [b, c], sources[1]: [c]})
    path = tmp_path/'comparison.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    output = tmp_path/'exports'
    index = export_report(path, output, 'FP32', 'FP16')
    assert index['counts'] == {'count_changed': 1, 'unmatched': 2}
    assert index['unique_images'] == 2
    assert len({row['original'] for row in index['groups']['unmatched']}) == 2
    for group in index['groups'].values():
        for row in group:
            assert (output/row['original']).read_bytes() == sources_bytes(row['image'])
            image = cv2.imdecode(np.fromfile(output/row['diff'], np.uint8), cv2.IMREAD_COLOR)
            assert image.shape == (190, 120, 3)
    row = index['groups']['count_changed'][0]
    detail = json.loads((output/row['predictions']).read_text(encoding='utf-8'))
    assert detail['prediction_delta']['count_delta'] == 1
    assert detail['prediction_delta']['matched'][0]['delta_xyxy'] == [1, 0, 0, 0]
    assert detail['prediction_delta']['candidate_only'][0]['prediction'][0] == c[0]
    assert json.loads((output/'index.json').read_text(encoding='utf-8')) == index
    with pytest.raises(ValueError, match='must be empty'):
        export_report(path, output)


def sources_bytes(filename):
    from pathlib import Path
    return Path(filename).read_bytes()


@pytest.mark.parametrize('combined', [False, True])
def test_deployment_export_uses_tolerance_and_keeps_filtered_context(tmp_path, combined):
    source = tmp_path/'image.png'
    ok, encoded = cv2.imencode('.png', np.zeros((100, 120, 3), np.uint8))
    assert ok
    source.write_bytes(encoded.tobytes())
    name = str(source)
    baseline = [([10, 10, 40, 40], 'METEOR', '0.80'),
                ([70, 65, 90, 85], 'BUGS', '0.60')]
    candidate = [([11, 10, 40, 40], 'METEOR', '0.79'),
                 ([70, 65, 90, 85], 'BUGS', '0.49')]
    report = (build_report if combined else analyze)([name], {name: baseline}, {name: candidate})
    path = tmp_path/'deployment.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    output = tmp_path/'exports'
    index = export_report(path, output, 'FP32', 'FP16', deployment=True)
    assert index['counts'] == {'affected': 1}
    row = index['groups']['affected'][0]
    assert row['original'] is None
    assert not (output/'affected'/'original').exists()
    detail = json.loads((output/row['predictions']).read_text(encoding='utf-8'))
    assert detail['baseline_unmatched'] == [1]
    assert detail['candidate_unmatched'] == []
    assert detail['candidate_filtered'][0][2] == '0.49'
    # Render directly to avoid JPEG compression artifacts: the consistent
    # shifted box must be faint gray in the single deployment diff panel.
    from tools.export_photo_diff import deployment_record, render_diff
    view = render_diff(np.zeros((100, 120, 3), np.uint8),
                       deployment_record((report['perception'] if combined else report)['differences'][0]),
                       'FP32', 'FP16', True)
    assert view.shape == (190, 120, 3)
    assert view[90+40, 10].tolist() == [100, 100, 100]
    np.testing.assert_allclose(view[90+85, 70], [0, 165, 255], atol=3)
