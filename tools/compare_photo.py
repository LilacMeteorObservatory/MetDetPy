"""Compare two complete photo MDRF runs on the same image manifest.

Omitted images mean zero detections. Report exact equality, matched prediction
residuals and end-to-end consistency under independently applied score gates.
"""
import argparse
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MetLib.image_manifest import load_image_manifest


def image_key(filename):
    return os.path.normcase(str(Path(filename).resolve()))


def load_records(filename, images):
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    if data.get("type") != "image-prediction":
        raise ValueError(f"Not an image-prediction MDRF: {filename}")
    records = {}
    for record in data["results"]:
        key = image_key(record["img_filename"])
        if key not in images or key in records:
            raise ValueError(f"Unknown or duplicate result image: {key}")
        boxes, labels, scores = record["boxes"], record["preds"], record["prob"]
        if not (len(boxes) == len(labels) == len(scores)):
            raise ValueError(f"Unaligned detections: {key}")
        if any(len(box) != 4 or any(type(v) is not int for v in box) for box in boxes):
            raise ValueError(f"Expected integer xyxy boxes: {key}")
        records[key] = list(zip(boxes, labels, scores))
    return records


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    union = area(a) + area(b) - intersection
    return intersection / union if union else float(a == b)


def distribution(values):
    """Descriptive population statistics; undefined values are excluded."""
    values = [v for v in values if v is not None]
    if not values:
        return dict(count=0, mean=None, std=None, min=None, p50=None,
                    p90=None, p95=None, p99=None, max=None)
    data = np.asarray(values, dtype=float)
    percentiles = np.percentile(data, [50, 90, 95, 99])
    return dict(count=len(values), mean=float(data.mean()), std=float(data.std()),
                min=float(data.min()), max=float(data.max()),
                **{key: float(value) for key, value in
                   zip(('p50', 'p90', 'p95', 'p99'), percentiles)})


def prediction_residual(a, b, iou):
    box, other = a[0], b[0]
    width, height = max(0, box[2]-box[0]), max(0, box[3]-box[1])
    area = width*height
    candidate_area = max(0, other[2]-other[0])*max(0, other[3]-other[1])
    edges = [y-x for x, y in zip(box, other)]
    score_delta = float(Decimal(b[2])-Decimal(a[2]))
    normalized = ([edges[0]/width, edges[1]/height,
                   edges[2]/width, edges[3]/height] if width and height else None)
    center_shift = (float(np.hypot((edges[0]+edges[2])/(2*width),
                                   (edges[1]+edges[3])/(2*height)))
                    if width and height else None)
    return dict(baseline_score=float(a[2]), candidate_score=float(b[2]),
                signed_score_delta=score_delta, abs_score_delta=abs(score_delta),
                iou=iou, iou_loss=1-iou, edge_delta_px=edges,
                max_edge_delta_px=max(map(abs, edges)),
                max_normalized_edge_delta=max(map(abs, normalized)) if normalized else None,
                normalized_center_shift=center_shift,
                relative_area_delta=(candidate_area-area)/area if area else None,
                baseline_width_px=width, baseline_height_px=height,
                baseline_area_px2=area, candidate_area_px2=candidate_area,
                baseline_class=a[1], candidate_class=b[1],
                changed=a[0] != b[0] or a[1] != b[1] or Decimal(a[2]) != Decimal(b[2]))


RESIDUAL_METRICS = ('signed_score_delta', 'abs_score_delta', 'iou', 'iou_loss',
                    'max_edge_delta_px', 'max_normalized_edge_delta',
                    'normalized_center_shift', 'relative_area_delta')


def residual_summary(rows):
    return dict(pairs=len(rows), changed_pairs=sum(row['changed'] for row in rows),
                **{metric: distribution([row[metric] for row in rows])
                   for metric in RESIDUAL_METRICS})


def correlation(rows, x, y, log_x=False):
    pairs = [(row[x], row[y]) for row in rows
             if row[x] is not None and row[y] is not None and (not log_x or row[x] > 0)]
    result = dict(pairs=len(pairs), pearson_r=None)
    if len(pairs) >= 2:
        left, right = np.asarray(pairs, dtype=float).T
        if log_x:
            left = np.log10(left)
        if np.ptp(left) > 0 and np.ptp(right) > 0:
            result['pearson_r'] = float(np.corrcoef(left, right)[0, 1])
    return result


def residual_histogram(rows, metric):
    values = [row[metric] for row in rows if row[metric] is not None]
    if not values:
        return dict(count=0, edges=[], counts=[])
    if metric in ('signed_score_delta', 'abs_score_delta'):
        # One bin per saved hundredth; zero stays a separate, centered bin.
        edges = np.arange(np.floor(min(values)*100)-0.5,
                          np.ceil(max(values)*100)+1.5)/100
    else:
        edges = np.linspace(0, max(0.01, max(values)), 26)
    counts, edges = np.histogram(values, bins=edges)
    return dict(count=len(values), edges=edges.tolist(), counts=counts.tolist())


def summarize_residuals(rows, unmatched):
    changed = [row for row in rows if row['changed']]
    by_class = {}
    by_score = {f'{i/10:.1f}-{(i+1)/10:.1f}': [] for i in range(10)}
    by_area = {'below_1024': [], '1024_to_9216': [], 'at_least_9216': []}
    transitions = Counter()
    for row in rows:
        by_class.setdefault(row['baseline_class'], []).append(row)
        index = min(9, max(0, int(Decimal(str(row['baseline_score']))*10)))
        by_score[f'{index/10:.1f}-{(index+1)/10:.1f}'].append(row)
        area = row['baseline_area_px2']
        by_area['below_1024' if area < 1024 else
                '1024_to_9216' if area < 9216 else 'at_least_9216'].append(row)
        if row['baseline_class'] != row['candidate_class']:
            transitions[(row['baseline_class'], row['candidate_class'])] += 1
    unmatched_stats = {}
    for side, predictions in unmatched.items():
        unmatched_stats[side] = dict(
            count=len(predictions), by_class=dict(Counter(p[1] for p in predictions)),
            score=distribution([float(p[2]) for p in predictions]),
            area_px2=distribution([max(0, p[0][2]-p[0][0])*max(0, p[0][3]-p[0][1])
                                  for p in predictions]))
    return dict(
        semantics=dict(score='Saved rounded scores, candidate minus baseline',
                       population='Pairs selected by diagnostic greedy IoU matching; unmatched reported separately',
                       grouping='Baseline class, baseline score and baseline box area',
                       score_bins='Left inclusive, right exclusive; final bin includes 1.0',
                       area_bins='Native image pixels squared; descriptive bins, not universal object sizes',
                       normalization='x edges divided by baseline width, y edges by baseline height',
                       statistics='Population std and linear-interpolated percentiles; correlations are descriptive'),
        all_pairs=residual_summary(rows), changed_pairs=residual_summary(changed),
        histograms={metric: residual_histogram(rows, metric) for metric in
                    ('signed_score_delta', 'abs_score_delta', 'iou_loss', 'normalized_center_shift')},
        same_class_pairs=residual_summary([r for r in rows if r['baseline_class'] == r['candidate_class']]),
        by_baseline_class={key: residual_summary(group) for key, group in sorted(by_class.items())},
        by_baseline_score={key: residual_summary(group) for key, group in by_score.items()},
        by_baseline_area_px2={key: residual_summary(group) for key, group in by_area.items()},
        correlations=dict(
            iou_loss_vs_log10_baseline_area=correlation(rows, 'baseline_area_px2', 'iou_loss', True),
            abs_score_delta_vs_baseline_score=correlation(rows, 'baseline_score', 'abs_score_delta')),
        class_transitions=[dict(baseline_class=a, candidate_class=b, pairs=count)
                           for (a, b), count in sorted(transitions.items())],
        unmatched=unmatched_stats)


def compare_records(images, baseline, candidate, iou_threshold=0.5):
    """Exact equality is order independent; IoU diagnostics use greedy matching.

    Match exact boxes first, then remaining pairs by descending IoU, ignoring
    labels so category changes are reported instead of hidden as unmatched.
    Neither model is ground truth. No pass/fail tolerance is implied.
    """
    summary = dict(images=len(images), exact_box_class_images=0,
                   exact_saved_result_images=0, both_empty_images=0,
                   count_changed_images=0, baseline_boxes=0, candidate_boxes=0,
                   matched_boxes=0, class_changed_pairs=0, baseline_unmatched=0,
                   candidate_unmatched=0, max_coordinate_delta=0,
                   max_rounded_score_delta=0.0)
    differences = []
    residuals = []
    unmatched = dict(baseline=[], candidate=[])
    for filename in images:
        a, b = baseline.get(filename, []), candidate.get(filename, [])
        signature = lambda rows: Counter((tuple(box), label) for box, label, _ in rows)
        saved = lambda rows: Counter((tuple(box), label, score) for box, label, score in rows)
        exact = signature(a) == signature(b)
        exact_saved = saved(a) == saved(b)
        summary["exact_box_class_images"] += exact
        summary["exact_saved_result_images"] += exact_saved
        summary["both_empty_images"] += not a and not b
        summary["count_changed_images"] += len(a) != len(b)
        summary["baseline_boxes"] += len(a)
        summary["candidate_boxes"] += len(b)
        pairs = sorted(((box_iou(x[0], y[0]), x[0] == y[0], x[1] == y[1], i, j)
                        for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
        used_a, used_b, matches = set(), set(), []
        for iou, _, _, i, j in pairs:
            if iou < iou_threshold or i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            delta = max(abs(x-y) for x, y in zip(a[i][0], b[j][0]))
            residual = prediction_residual(a[i], b[j], iou)
            residuals.append(residual)
            score_delta = residual['abs_score_delta']
            summary["matched_boxes"] += 1
            summary["class_changed_pairs"] += a[i][1] != b[j][1]
            summary["max_coordinate_delta"] = max(summary["max_coordinate_delta"], delta)
            summary["max_rounded_score_delta"] = max(summary["max_rounded_score_delta"], score_delta)
            matches.append(dict(baseline_index=i, candidate_index=j, iou=iou,
                                coordinate_delta=delta, rounded_score_delta=score_delta,
                                class_changed=a[i][1] != b[j][1], residual=residual))
        summary["baseline_unmatched"] += len(a)-len(used_a)
        summary["candidate_unmatched"] += len(b)-len(used_b)
        unmatched['baseline'].extend(p for i, p in enumerate(a) if i not in used_a)
        unmatched['candidate'].extend(p for j, p in enumerate(b) if j not in used_b)
        if not exact_saved:
            differences.append(dict(image=filename, box_class_equal=exact,
                                    baseline=a, candidate=b, matches=matches,
                                    baseline_unmatched=sorted(set(range(len(a)))-used_a),
                                    candidate_unmatched=sorted(set(range(len(b)))-used_b)))
    summary["exact_box_class_rate"] = summary["exact_box_class_images"] / len(images) if images else None
    summary["exact_saved_result_rate"] = summary["exact_saved_result_images"] / len(images) if images else None
    nonempty = len(images) - summary["both_empty_images"]
    summary["nonempty_images"] = nonempty
    summary["nonempty_exact_box_class_rate"] = (
        (summary["exact_box_class_images"] - summary["both_empty_images"]) / nonempty
        if nonempty else None)
    return dict(summary=summary, differences=differences,
                residuals=summarize_residuals(residuals, unmatched))


def consistent_matching(baseline, candidate, iou_limit, score_delta):
    edges = {}
    for i, a in enumerate(baseline):
        edges[i] = sorted(
            [j for j, b in enumerate(candidate)
             if a[1] == b[1] and box_iou(a[0], b[0]) > iou_limit
             and abs(Decimal(a[2])-Decimal(b[2])) < score_delta],
            key=lambda j: (-box_iou(a[0], candidate[j][0]), j))
    right = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in right or augment(right[j], visited):
                right[j] = i
                return True
        return False

    for i in range(len(baseline)):
        augment(i, set())
    return sorted((i, j) for j, i in right.items())


def analyze_perception(images, baseline, candidate, score_min=Decimal('0.5'),
            iou_limit=0.9, score_delta=Decimal('0.05')):
    summary = Counter(dict.fromkeys(('baseline_kept', 'candidate_kept', 'baseline_filtered',
                                    'candidate_filtered', 'consistent_pairs', 'baseline_inconsistent',
                                    'candidate_inconsistent', 'fully_consistent_images',
                                    'count_changed_images', 'both_empty_images'), 0))
    summary['images'] = len(images)
    by_class = {}
    differences = []
    for filename in images:
        raw_a, raw_b = baseline.get(filename, []), candidate.get(filename, [])
        a = [p for p in raw_a if Decimal(p[2]) >= score_min]
        b = [p for p in raw_b if Decimal(p[2]) >= score_min]
        pairs = consistent_matching(a, b, iou_limit, score_delta)
        used_a, used_b = {i for i, _ in pairs}, {j for _, j in pairs}
        for key, value in dict(baseline_kept=len(a), candidate_kept=len(b),
                               baseline_filtered=len(raw_a)-len(a), candidate_filtered=len(raw_b)-len(b),
                               consistent_pairs=len(pairs), baseline_inconsistent=len(a)-len(pairs),
                               candidate_inconsistent=len(b)-len(pairs),
                               fully_consistent_images=len(a)==len(b)==len(pairs),
                               count_changed_images=len(a)!=len(b),
                               both_empty_images=not a and not b).items():
            summary[key] += value
        for side, rows in [('baseline', a), ('candidate', b)]:
            for p in rows:
                by_class.setdefault(p[1], Counter())[side+'_kept'] += 1
        for i, _ in pairs:
            by_class[a[i][1]]['consistent_pairs'] += 1
        if len(a) != len(pairs) or len(b) != len(pairs):
            differences.append(dict(image=filename, baseline=a, candidate=b,
                                    consistent_pairs=pairs,
                                    baseline_inconsistent=sorted(set(range(len(a)))-used_a),
                                    candidate_inconsistent=sorted(set(range(len(b)))-used_b),
                                    baseline_filtered=[p for p in raw_a if Decimal(p[2]) < score_min],
                                    candidate_filtered=[p for p in raw_b if Decimal(p[2]) < score_min]))
    summary = dict(summary)
    # Explain residuals after valid matching; these secondary diagnostic
    # matches do not change the requested consistency counts.
    diagnostics = Counter(score_difference_pairs=0, localization_pairs_iou_gt_05=0,
                          baseline_remaining=0, candidate_remaining=0,
                          candidate_below_threshold_pairs=0, baseline_below_threshold_pairs=0)
    for record in differences:
        a = [record['baseline'][i] for i in record['baseline_inconsistent']]
        b = [record['candidate'][i] for i in record['candidate_inconsistent']]
        for name, threshold in [('score_difference_pairs', iou_limit),
                                ('localization_pairs_iou_gt_05', min(0.5, iou_limit))]:
            pairs = consistent_matching(a, b, threshold, Decimal('2'))
            diagnostics[name] += len(pairs)
            used_a, used_b = {i for i, _ in pairs}, {j for _, j in pairs}
            a = [p for i, p in enumerate(a) if i not in used_a]
            b = [p for j, p in enumerate(b) if j not in used_b]
        diagnostics['baseline_remaining'] += len(a)
        diagnostics['candidate_remaining'] += len(b)
        diagnostics['candidate_below_threshold_pairs'] += len(
            consistent_matching(a, record['candidate_filtered'], iou_limit, Decimal('2')))
        diagnostics['baseline_below_threshold_pairs'] += len(
            consistent_matching(record['baseline_filtered'], b, iou_limit, Decimal('2')))
    summary['different_images'] = len(differences)
    summary['different_image_rate'] = len(differences)/len(images) if images else None
    summary['baseline_consistency_rate'] = summary['consistent_pairs']/summary['baseline_kept'] if summary['baseline_kept'] else None
    summary['candidate_consistency_rate'] = summary['consistent_pairs']/summary['candidate_kept'] if summary['candidate_kept'] else None
    for counts in by_class.values():
        for side in ('baseline', 'candidate'):
            counts[side+'_inconsistent'] = counts[side+'_kept']-counts['consistent_pairs']
    return dict(summary=summary, by_class=by_class, diagnostics=dict(diagnostics), differences=differences)


def build_report(images, baseline, candidate, iou_threshold=0.5,
                 score_min=Decimal('0.5'), perceptual_iou=0.9, score_delta=Decimal('0.05')):
    report = compare_records(images, baseline, candidate, iou_threshold)
    report['perception'] = analyze_perception(images, baseline, candidate, score_min, perceptual_iou, score_delta)
    report['perception']['thresholds'] = dict(score_min=str(score_min),
                                             strict_iou_gt=perceptual_iou,
                                             strict_score_delta_lt=str(score_delta))
    return report


def export_statistical_plots(report, output):
    """Render report aggregates, without reloading images or repeating inference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    residuals = report['residuals']
    colors = ('#2878a8', '#e79436')

    def save(figure, name, title, note):
        figure.suptitle(title, fontsize=15)
        figure.text(0.02, 0.02, note, fontsize=9, color='#555555')
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
        path = output / f'{name}.png'
        figure.savefig(path, dpi=160, facecolor='white')
        paths[name] = dict(png=str(path))
        plt.close(figure)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, histogram) in zip(axes.flat, residuals['histograms'].items()):
        edges, counts = histogram['edges'], histogram['counts']
        if counts:
            ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
                   color=colors[0], edgecolor='white', linewidth=0.4)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No matched samples', ha='center', transform=ax.transAxes)
        ax.set_title(f"{metric.replace('_', ' ')} (n={histogram['count']})")
        ax.set_xlabel('Candidate - baseline' if metric == 'signed_score_delta' else 'Residual')
        ax.set_ylabel('Pair count (log scale)')
        ax.grid(axis='y', alpha=0.2)
    save(fig, 'residual_distributions', 'Matched prediction residual distributions',
         'Includes unchanged pairs. Scores are saved to 2 decimals; unmatched predictions are excluded.')

    groups = residuals['by_baseline_score']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(groups))
    labels = [f"{name}\nn={stats['pairs']}" for name, stats in groups.items()]
    for ax, metric in zip(axes, ('abs_score_delta', 'iou_loss')):
        for statistic, color in zip(('mean', 'p95'), colors):
            values = [stats[metric][statistic] for stats in groups.values()]
            ax.plot(x, [v if v is not None else np.nan for v in values],
                    'o-', label=statistic.upper(), color=color)
        ax.set_xticks(x, labels, rotation=45, ha='right')
        ax.set_title(metric.replace('_', ' '))
        ax.set_xlabel('Baseline saved score bin')
        ax.set_ylabel('Residual')
        ax.legend()
        ax.grid(alpha=0.2)
    save(fig, 'residuals_by_score', 'Residuals by baseline prediction score',
         'Diagnostic greedy IoU pairs; empty bins are gaps. Final score bin includes 1.0.')

    class_count = len(residuals['by_baseline_class'])
    fig, axes = plt.subplots(2, 2, figsize=(13, max(8, class_count*0.65)))
    for column, key in enumerate(('by_baseline_area_px2', 'by_baseline_class')):
        groups = residuals[key]
        y = np.arange(len(groups))
        for row, metric in enumerate(('iou_loss', 'normalized_center_shift')):
            ax = axes[row, column]
            labels = [f"{name} (n={stats[metric]['count']})" for name, stats in groups.items()]
            for offset, statistic, color in zip((-0.18, 0.18), ('mean', 'p95'), colors):
                values = [stats[metric][statistic] for stats in groups.values()]
                ax.barh(y+offset, [v if v is not None else np.nan for v in values],
                        height=0.35, color=color, label=statistic.upper())
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set_title(metric.replace('_', ' '))
            ax.set_xlabel('Residual')
            ax.legend()
            ax.grid(axis='x', alpha=0.2)
    save(fig, 'residuals_by_size_and_class', 'Localization residuals by baseline size and class',
         'Area bins are native pixels squared. Center shifts are normalized by baseline width/height; zero-size boxes excluded.')

    perception = report['perception']
    summary = perception['summary']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(('Fully consistent', 'Different'),
                 (summary['fully_consistent_images'], summary['different_images']), color=colors)
    axes[0].set_xlabel('Images')
    axes[0].set_title(f"End-to-end image agreement (n={summary['images']})")
    axes[0].set_xlim(0, max(1, summary['images'])*1.15)
    for i, count in enumerate((summary['fully_consistent_images'], summary['different_images'])):
        axes[0].text(count, i, f' {count}', va='center')
    classes = sorted(perception['by_class'])
    y = np.arange(len(classes))
    for offset, side, color in zip((-0.18, 0.18), ('baseline', 'candidate'), colors):
        rates = [100*perception['by_class'][name][side+'_inconsistent']/perception['by_class'][name][side+'_kept']
                 if perception['by_class'][name][side+'_kept'] else np.nan for name in classes]
        axes[1].barh(y+offset, rates, height=0.35, label=side, color=color)
    axes[1].set_yticks(y, classes)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Inconsistent retained predictions (%)')
    axes[1].set_title('Perception differences by class')
    axes[1].legend()
    thresholds = perception['thresholds']
    save(fig, 'perception_overview', 'End-to-end perception differences',
         f"Score >= {thresholds['score_min']}; same class; IoU > {thresholds['strict_iou_gt']}; "
         f"score delta < {thresholds['strict_score_delta_lt']}. Both-empty images count as consistent.")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="The common image list used for both successful runs")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument('--score-min', type=Decimal, default=Decimal('0.5'))
    parser.add_argument('--perceptual-iou', type=float, default=0.9)
    parser.add_argument('--score-delta', type=Decimal, default=Decimal('0.05'))
    parser.add_argument("--output", required=True)
    parser.add_argument('--plots-dir', help='Plot directory; defaults to <output stem>_plots')
    args = parser.parse_args(argv)
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")
    if (not 0 <= args.perceptual_iou < 1 or not args.score_min.is_finite()
            or not 0 <= args.score_min <= 1 or not args.score_delta.is_finite()
            or args.score_delta <= 0):
        parser.error('Invalid perception score/IoU thresholds')
    images = [image_key(p) for p in load_image_manifest(args.manifest)]
    baseline = load_records(args.baseline, set(images))
    candidate = load_records(args.candidate, set(images))
    report = build_report(images, baseline, candidate, args.iou, args.score_min,
                          args.perceptual_iou, args.score_delta)
    report["inputs"] = dict(manifest=str(Path(args.manifest).resolve()),
                            baseline=str(Path(args.baseline).resolve()),
                            candidate=str(Path(args.candidate).resolve()),
                            iou_threshold=args.iou)
    output = Path(args.output)
    plots_dir = Path(args.plots_dir) if args.plots_dir else output.with_name(output.stem+'_plots')
    report['plots'] = export_statistical_plots(report, plots_dir)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(exact=report['summary'], perception=report['perception']['summary']),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
