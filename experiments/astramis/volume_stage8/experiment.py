from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
import csv
import hashlib
import importlib.util
import json
import math

W = Path('/home/jovyan/work')
R = W / 'runs'
OUT = R / 'volume_stage8'
D7 = R / 'diagnostic_stage7'
VOL = R / 'rgbd_volume_stage2'
TAKE = R / 'take_stage3'
BASE = TAKE / 'submission_take_stage3_diagnostic.csv'
BASE_SHA = '2c55a9bf5bab0e9118c8ea7cbee69f0d392e47712f17c288d7bee034a19dafd1'
SCORER = W / 'datasets/score.py'
LABELS = W / 'datasets/dataset/labels/real_train.json'
SPLIT = W / 'baseline_v0/split_v0.json'
OOF_PREDS = R / 'material_final_m_20261121_7f35d05_oof4/oof_predictions.json'
FACTOR = 0.75
MAX_Q95_M = 0.030


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_json(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index(path):
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    result = {(row['frame'], int(row['det'])): row for row in rows}
    assert len(result) == len(rows), path
    return result


def checked(path):
    assert sha(path) == read(D7 / 'verification.json')['inputs'][str(path.relative_to(W))]['sha256'], path
    return path


spec = importlib.util.spec_from_file_location('stage8_official_score', checked(SCORER))
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
MATERIALS = scorer.MATERIALS


def finite_float(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def eligibility(probabilities, height, fallback):
    probs = {material: finite_float(probabilities.get('p_' + material)) for material in MATERIALS}
    q95 = finite_float(height)
    valid_p = all(p is not None and 0 <= p <= 1 for p in probs.values())
    valid_p = valid_p and math.isclose(sum(probs.values()), 1, rel_tol=0, abs_tol=1e-6)
    valid_h = q95 is not None and q95 >= 0 and finite_float(fallback) == 0
    top = sorted(MATERIALS, key=probs.get, reverse=True)[:2] if valid_p else []
    eligible = valid_p and valid_h and 'wood' in top and q95 <= MAX_Q95_M
    return dict(probabilities_valid=valid_p, height_valid=valid_h, q95_mm=None if q95 is None else q95*1000,
                dino_top1=top[0] if top else None, dino_top2=top[1] if top else None,
                eligible=eligible, **{'p_' + material: p for material, p in probs.items()})


def compare(gt, a, b):
    ma, mb = scorer.components(gt, a), scorer.components(gt, b)
    assert all(ma[k] == mb[k] for k in ('D', 'M', 'S'))
    delta = {k: mb[k] - ma[k] for k in ma}
    assert math.isclose(delta['Score'], 100 * scorer.WEIGHTS['V'] * delta['V'], abs_tol=1e-10)
    return {'A': ma, 'B': mb, 'delta': delta}


def aggregate(rows):
    return dict(frames=len(rows), changed_predictions=sum(r['changed'] for r in rows),
                V_A=mean(r['V_A'] for r in rows), V_B=mean(r['V_B'] for r in rows),
                delta_V=mean(r['delta_V'] for r in rows), sum_delta_V=sum(r['delta_V'] for r in rows),
                abs_log_error_A=mean(r['abs_log_error_A'] for r in rows),
                abs_log_error_B=mean(r['abs_log_error_B'] for r in rows),
                delta_abs_log_error=mean(r['abs_log_error_B'] - r['abs_log_error_A'] for r in rows),
                MAE_ml_A=mean(r['abs_error_ml_A'] for r in rows), MAE_ml_B=mean(r['abs_error_ml_B'] for r in rows))


def counts(objects):
    deltas = [o['delta_V'] for o in objects]
    return dict(physical_objects=len(deltas), improved=sum(d > 1e-12 for d in deltas),
                worsened=sum(d < -1e-12 for d in deltas), unchanged=sum(abs(d) <= 1e-12 for d in deltas))


def evaluate(part):
    assert part in ('train', 'val') and sha(BASE) == BASE_SHA
    tag = 'grouped_oof' if part == 'train' else 'independent_val'
    if part == 'val':
        assert read(OUT / 'grouped_oof_metrics.json')['val_gate']
        assert not (OUT / 'independent_val_metrics.json').exists(), 'Independent val was already evaluated'
    pack = read(checked(VOL / f'{part}_frames.json'))
    geo = index(checked(VOL / f'{part}_predictions.csv'))
    upstream = index(checked(TAKE / ('oof_strict_predictions.csv' if part == 'train' else 'val_strict_predictions.csv')))
    source = read(checked(OOF_PREDS))['A'] if part == 'train' else pack['pred']
    keys = {(f, j) for f, objects in pack['pred'].items() for j in range(len(objects))}
    assert keys == set(geo) == set(upstream) and set(source) == set(pack['pred'])
    predictions_a = {}
    predictions_b = {}
    details = {}
    for f, objects in pack['pred'].items():
        predictions_a[f] = []
        predictions_b[f] = []
        assert len(objects) == len(source[f])
        for j, box in enumerate(objects):
            key = f, j
            u, g, ref = upstream[key], geo[key], source[f][j]
            volume = float(u['corrected_ml'])
            assert math.isfinite(volume) and volume > 0 and math.isclose(volume, float(g['corrected_ml']), rel_tol=1e-9)
            assert box['form'] == u['form'] == ref['form']
            assert abs(box['score'] - ref['score']) < 1e-6 and max(abs(x-y) for x,y in zip(box['box'], ref['box'])) < .001
            assert math.isclose(float(u['mass']), float(ref['mass']), rel_tol=1e-9)
            old = dict(box, material=u['predicted_material'], volume=volume, mass=float(u['mass']), take=int(u['B']))
            inputs = eligibility(u, g.get('q95_m'), g.get('base_fallback'))
            if inputs['probabilities_valid']:
                assert inputs['dino_top1'] == old['material']
            new_volume = FACTOR * volume if inputs['eligible'] else volume
            predictions_a[f].append(old)
            predictions_b[f].append(dict(old, volume=new_volume))
            details[key] = dict(frame=f, det=j, **inputs, volume_A_ml=volume, volume_B_ml=new_volume,
                                changed=new_volume != volume, matched=False, object_id='', true_material='')
    # Ground truth enters only after both predictions have been built.
    split = read(checked(SPLIT))
    solution = scorer.labels_to_solution(checked(LABELS))
    official = {f: scorer._parse(cell, False, f) for f, cell in zip(solution.image_id, solution.objects)}
    assert set(pack['truth']) == set(predictions_a) == set(split[part]['image_ids'])
    gt = {f: official[f] for f in predictions_a}
    all_objects = {g['object_id'] for objects in pack['truth'].values() for g in objects}
    assert all_objects == set(split[part]['object_ids']) and len(all_objects) == (32 if part == 'train' else 8)
    assert not set(split['train']['object_ids']).intersection(split['val']['object_ids'])
    folds = read(VOL / 'configuration.json')['held']
    grouped = {o: int(f) for f, ids in folds.items() for o in ids}
    assert set(grouped) == set(split['train']['object_ids']) and len(folds) == 4
    full = compare(gt, predictions_a, predictions_b)
    baseline_stage7 = read(TAKE / 'metrics.json')[part + '_strict']['B']
    assert all(math.isclose(v, baseline_stage7[k], abs_tol=1e-9) for k, v in full['A'].items())
    matched = []
    object_rows = defaultdict(list)
    for f, objects in gt.items():
        assert len(objects) == len(pack['truth'][f]) <= 1
        for ground, match, original in zip(objects, scorer._match(objects, predictions_a[f]), pack['truth'][f], strict=True):
            assert match is not None
            j = next(j for j, p in enumerate(predictions_a[f]) if p is match)
            d = details[f, j]
            oid = original['object_id']
            if part == 'train':
                assert int(geo[f, j]['fold']) == int(upstream[f, j]['group']) == grouped[oid]
            assert ground['volume'] == original['volume'] and ground['material'] == original['material']
            d.update(matched=True, object_id=oid, true_material=ground['material'])
            target = ground['volume']
            errors = [abs(math.log(d[k] / target)) for k in ('volume_A_ml', 'volume_B_ml')]
            va, vb = (max(0., 1 - e / math.log(3)) for e in errors)
            row = dict(d, volume_true_ml=target, V_A=va, V_B=vb, delta_V=vb-va,
                       abs_log_error_A=errors[0], abs_log_error_B=errors[1],
                       abs_error_ml_A=abs(d['volume_A_ml']-target), abs_error_ml_B=abs(d['volume_B_ml']-target))
            matched.append(row)
            object_rows[oid].append(row)
    assert len(matched) == (165 if part == 'train' else 49) and set(object_rows) == all_objects
    assert all(math.isclose(mean(r['V_'+key] for r in matched), full[key]['V'], abs_tol=1e-12) for key in ('A','B'))
    objects = [dict(object_id=o, true_material=rs[0]['true_material'], **aggregate(rs)) for o,rs in sorted(object_rows.items())]
    materials = []
    for mat in sorted({r['true_material'] for r in matched}) + ['non_wood']:
        included = {o['object_id'] for o in objects if (o['true_material'] != 'wood' if mat == 'non_wood' else o['true_material'] == mat)}
        materials.append(dict(true_material=mat, **counts([o for o in objects if o['object_id'] in included]),
                              **aggregate([r for r in matched if r['object_id'] in included])))
    best = max(objects, key=lambda o: o['sum_delta_V'])
    leave_one_out = []
    for obj in objects if part == 'train' else [best]:
        excluded = {r['frame'] for r in object_rows[obj['object_id']]}
        keep = [f for f in gt if f not in excluded]
        result = compare(*({f: data[f] for f in keep} for data in (gt, predictions_a, predictions_b)))
        leave_one_out.append(dict(excluded_object=obj['object_id'], true_material=obj['true_material'],
                                  excluded_frames=len(excluded), excluded_sum_delta_V=obj['sum_delta_V'],
                                  V_A=result['A']['V'], V_B=result['B']['V'], delta_V=result['delta']['V'],
                                  Score_A=result['A']['Score'], Score_B=result['B']['Score'], delta_Score=result['delta']['Score']))
    without_best = next(r for r in leave_one_out if r['excluded_object'] == best['object_id'])
    diagnostics = list(details.values())
    totals = dict(image_ids=len(gt), matched_frames=len(matched), detections=len(diagnostics),
                  changed_predictions=sum(r['changed'] for r in diagnostics),
                  changed_matched=sum(r['changed'] for r in matched),
                  changed_physical_objects=sum(o['changed_predictions'] > 0 for o in objects),
                  changed_unmatched=sum(r['changed'] and not r['matched'] for r in diagnostics),
                  valid_probabilities=sum(r['probabilities_valid'] for r in diagnostics),
                  valid_height=sum(r['height_valid'] for r in diagnostics),
                  both_valid=sum(r['probabilities_valid'] and r['height_valid'] for r in diagnostics),
                  **counts(objects))
    result = dict(split=tag, factor=FACTOR, max_q95_m=MAX_Q95_M, **full, counts=totals,
                  without_best=without_best, by_material=materials)
    if part == 'train':
        result['val_gate'] = full['delta']['V'] > 0 and without_best['delta_V'] > 0 and totals['improved'] > 1
    write_csv(tag + '_detections.csv', diagnostics)
    write_csv(tag + '_matched.csv', matched)
    write_csv(tag + '_objects.csv', objects)
    write_csv(tag + '_materials.csv', materials)
    write_csv(tag + '_leave_one_out.csv', leave_one_out)
    write_json(tag + '_metrics.json', result)
    assert sha(BASE) == BASE_SHA
    return result
