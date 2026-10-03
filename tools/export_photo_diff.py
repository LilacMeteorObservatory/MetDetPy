"""Export source images, prediction JSON and annotated views from compare_photo.

Groups overlap: count_changed contains unequal detection counts; unmatched
contains images with unmatched boxes on either side at the report's IoU cutoff.
No inference or matching is repeated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from MetLib.fileio import (SUPPORT_COMMON_FORMAT, is_ext_within,
                           load_8bit_image, load_raw_with_preprocess)
from tools.compare_photo import box_iou

BASELINE_COLOR = (0, 165, 255)  # orange, BGR
CANDIDATE_COLOR = (255, 200, 0)  # cyan, BGR


def deployment_record(record):
    """Adapt strict-tolerance matches without recomputing their assignment."""
    matches = []
    for i, j in record['consistent_pairs']:
        a, b = record['baseline'][i], record['candidate'][j]
        matches.append(dict(baseline_index=i, candidate_index=j, iou=box_iou(a[0], b[0]),
                            coordinate_delta=max(abs(x-y) for x, y in zip(a[0], b[0])),
                            rounded_score_delta=abs(float(a[2])-float(b[2])), class_changed=False))
    return dict(record, matches=matches, baseline_unmatched=record['baseline_inconsistent'],
                candidate_unmatched=record['candidate_inconsistent'])


def selected_groups(record):
    groups = []
    if len(record["baseline"]) != len(record["candidate"]):
        groups.append("count_changed")
    if record["baseline_unmatched"] or record["candidate_unmatched"]:
        groups.append("unmatched")
    return groups


def prediction_delta(record):
    changes = []
    for match in record["matches"]:
        i, j = match["baseline_index"], match["candidate_index"]
        a, b = record["baseline"][i], record["candidate"][j]
        changes.append(dict(**match, baseline=a, candidate=b,
                            delta_xyxy=[y-x for x, y in zip(a[0], b[0])],
                            saved_score_delta=float(b[2])-float(a[2])))
    return dict(count_delta=len(record["candidate"])-len(record["baseline"]),
                matched=changes,
                baseline_only=[dict(index=i, prediction=record["baseline"][i])
                               for i in record["baseline_unmatched"]],
                candidate_only=[dict(index=i, prediction=record["candidate"][i])
                                for i in record["candidate_unmatched"]])


def draw_prediction(image, prediction, label, color, unmatched=False, dashed=False):
    box, category, score = prediction
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    # Clip only the display coordinates; JSON retains the model's coordinates.
    x1, x2 = [min(w-1, max(0, x)) for x in (x1, x2)]
    y1, y2 = [min(h-1, max(0, y)) for y in (y1, y2)]
    thickness = 4 if unmatched else 2
    if dashed:
        for start in range(x1, x2+1, 14):
            for y in (y1, y2):
                cv2.line(image, (start, y), (min(start+7, x2), y), color, thickness)
        for start in range(y1, y2+1, 14):
            for x in (x1, x2):
                cv2.line(image, (x, start), (x, min(start+7, y2)), color, thickness)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text = f"{label}{'*' if unmatched else ''} {category} {score}"
    # Separate the two sides' labels on overlapping boxes in the single panel.
    text_y = min(h-6, y2+18) if label.startswith('B') else max(18, y1-6)
    position = (min(x1, max(0, w-300)), text_y)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def render_diff(image, record, baseline_label, candidate_label, deployment=False):
    """Single overlay: shared predictions are faint; differences use side colors."""
    panel = image.copy()
    changed_b, changed_c = set(record["baseline_unmatched"]), set(record["candidate_unmatched"])
    for match in record["matches"]:
        i, j = match["baseline_index"], match["candidate_index"]
        a, b = record["baseline"][i], record["candidate"][j]
        if not deployment and (a[0] != b[0] or a[1] != b[1]):
            changed_b.add(i)
            changed_c.add(j)
    # Draw context first so highlighted differences remain visible on top.
    h, w = image.shape[:2]
    shared_boxes = set()
    for side, changed in [('baseline', changed_b), ('candidate', changed_c)]:
        for i, prediction in enumerate(record[side]):
            if i in changed:
                continue
            box = tuple(prediction[0])
            if box in shared_boxes:
                continue
            shared_boxes.add(box)
            x1, y1, x2, y2 = box
            cv2.rectangle(panel, (min(w-1, max(0, x1)), min(h-1, max(0, y1))),
                          (min(w-1, max(0, x2)), min(h-1, max(0, y2))), (100, 100, 100), 1)
    if deployment:
        # Filtered predictions near affected boxes are context, not retained detections.
        for side, opposite, prefix in [('baseline', 'candidate', 'Bf'),
                                       ('candidate', 'baseline', 'Cf')]:
            affected = [record[opposite][i] for i in record[opposite+'_unmatched']]
            for i, p in enumerate(record[side+'_filtered']):
                if any(p[1] == q[1] and box_iou(p[0], q[0]) > 0.5 for q in affected):
                    draw_prediction(panel, p, f'{prefix}{i} FILTERED', (100, 100, 100), dashed=True)
    for side, indices, prefix, color, dashed in [
            ("baseline", changed_b, "B", BASELINE_COLOR, True),
            ("candidate", changed_c, "C", CANDIDATE_COLOR, False)]:
        for i in sorted(indices):
            draw_prediction(panel, record[side][i], f"{prefix}{i}", color,
                            i in record[side+"_unmatched"], dashed)
    header_height = 90
    canvas = np.full((h+header_height, w, 3), 30, np.uint8)
    canvas[header_height:] = panel
    def header_text(text, y, scale):
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        scale *= min(1, max(1, w-20)/max(1, text_width))
        cv2.putText(canvas[:header_height], text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (240, 240, 240), 1)
    header_text(f'DIFF | B: {baseline_label} | C: {candidate_label}', 28, 0.6)
    header_text('B dashed orange; C solid cyan; shared gray; * unmatched', 55, 0.5)
    status = 'Affected' if deployment else 'Unmatched'
    header_text(f"{status} B={len(record['baseline_unmatched'])} C={len(record['candidate_unmatched'])}", 78, 0.5)
    return canvas


def export_report(report_path, output, baseline_label="baseline", candidate_label="candidate", deployment=False):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    analysis = report.get('perception', report) if deployment else report
    output = Path(output)
    # Avoid stale images from an earlier report being mistaken for this run.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    groups = {group: [] for group in (('affected',) if deployment else ("count_changed", "unmatched"))}
    folders = ('diff', 'predictions') if deployment else ('original', 'diff', 'predictions')
    for group in groups:
        for folder in folders:
            (output/group/folder).mkdir(parents=True, exist_ok=True)
    selected = ([deployment_record(record) for record in analysis['differences']] if deployment else
                [record for record in analysis["differences"] if selected_groups(record)])
    # Verify all sources before writing any image artifacts.
    for record in selected:
        if not Path(record["image"]).is_file():
            raise FileNotFoundError(record["image"])
    for record in selected:
        source = Path(record["image"])
        tag = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
        stem = source.stem[:60] + "_" + tag
        image = (load_8bit_image(str(source)) if is_ext_within(str(source), SUPPORT_COMMON_FORMAT)
                 else load_raw_with_preprocess(str(source), output_bps=8))
        view = render_diff(image, record, baseline_label, candidate_label, deployment)
        ok, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"Cannot encode diff image: {source}")
        detail = dict(record, prediction_delta=prediction_delta(record))
        for group in (['affected'] if deployment else selected_groups(record)):
            target = output/group
            original = target/"original"/(stem+source.suffix)
            diff = target/"diff"/(stem+".diff.jpg")
            predictions = target/"predictions"/(stem+".json")
            if not deployment:
                shutil.copy2(source, original)
            diff.write_bytes(encoded.tobytes())
            predictions.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
            groups[group].append(dict(image=str(source), baseline_count=len(record["baseline"]),
                                      candidate_count=len(record["candidate"]),
                                      baseline_unmatched=record["baseline_unmatched"],
                                      candidate_unmatched=record["candidate_unmatched"],
                                      original=None if deployment else str(original.relative_to(output)),
                                      diff=str(diff.relative_to(output)),
                                      predictions=str(predictions.relative_to(output))))
    index = dict(report=str(Path(report_path).resolve()), inputs=report.get("inputs"),
                 baseline_label=baseline_label, candidate_label=candidate_label,
                 deployment=deployment,
                 unique_images=len(selected),
                 counts={group: len(entries) for group, entries in groups.items()},
                 groups=groups)
    (output/"index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="comparison.json generated by compare_photo.py")
    parser.add_argument("--output", required=True, help="Empty export directory")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument('--deployment', action='store_true',
                        help='Export perception differences from a combined or legacy tolerance report')
    args = parser.parse_args(argv)
    index = export_report(args.report, args.output, args.baseline_label, args.candidate_label, args.deployment)
    print(json.dumps(dict(unique_images=index["unique_images"], **index["counts"]), indent=2))


if __name__ == "__main__":
    main()
