from decimal import Decimal
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.special import softmax
import torch

from experiment import BASE, BASE_SHA, D7, FACTOR, OUT, R, VOL, W, checked, eligibility, read, scorer, sha, write_csv, write_json

MAT = R / 'dinov2_material_stage1'
DATA = W / 'datasets/dataset'
TEST = DATA / 'real/test'
MASKS = VOL / 'test_positive_masks.jsonl.gz'
HEAD = MAT / 'head_train32.npz'
WEIGHT = R / 'material_dinov2_v1_20261122/dinov2_vits14_pretrain.pth'
STAGE2 = VOL / 'submission_dinov2_volume_diagnostic.csv'
SAMPLE = W / 'datasets/sample_submission.csv'
EMBEDDINGS = OUT / 'test_detector_embeddings.npz'
DEST = OUT / 'submission_volume_wood_v1.csv'


def parse(blob):
    return list(csv.reader(io.StringIO(blob.decode('utf-8'), newline=''), strict=True))


def render(rows):
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\n').writerows(rows)
    return stream.getvalue().encode('utf-8')


def verify(rows, original, sample_ids, cam, changes):
    assert rows[0] == original[0] == ['image_id', 'prediction']
    assert [f for f, _ in rows[1:]] == sample_ids == [f for f, _ in original[1:]]
    assert len(rows) == 138 and len(set(sample_ids)) == 137
    observed = set()
    reverse = [rows[0]]
    total = empty = 0
    for (frame, before), (_, after) in zip(original[1:], rows[1:], strict=True):
        a = scorer._parse(before, True, frame)
        b = scorer._parse(after, True, frame)
        assert len(a) == len(b)
        total += len(b)
        if not a:
            assert before == after == 'none'
            reverse.append([frame, after])
            empty += 1
            continue
        restored = []
        for j, (old, new) in enumerate(zip(before.split(';'), after.split(';'), strict=True)):
            t, u = old.split(' '), new.split(' ')
            assert len(t) == len(u) == 10 and t[:7] == u[:7] and t[8:] == u[8:]
            x1, y1, x2, y2, conf = map(float, u[:5])
            volume, mass = float(u[7]), float(u[8])
            assert 0 <= x1 < x2 <= cam['width'] and 0 <= y1 < y2 <= cam['height']
            assert 0 <= conf <= 1 and math.isfinite(volume) and volume > 0 and math.isfinite(mass) and mass >= 0
            assert u[9] in ('0', '1') and b[j]['volume'] == volume
            key = frame, j
            if t[7] != u[7]:
                observed.add(key)
                assert (t[7], u[7]) == changes[key]
                assert Decimal(u[7]) == Decimal(t[7]) * Decimal(str(FACTOR))
                assert Decimal(u[7]) / Decimal(str(FACTOR)) == Decimal(t[7])
                u[7] = changes[key][0]
            restored.append(' '.join(u))
        reverse.append([frame, ';'.join(restored)])
    assert (total, empty) == (144, 19) and observed == set(changes)
    return render(reverse), total, empty


def main():
    oof = read(OUT / 'grouped_oof_metrics.json')
    val = read(OUT / 'independent_val_metrics.json')
    assert oof['val_gate'] and val['delta']['V'] > 0
    raw = BASE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    rows = parse(raw)
    assert render(rows) == raw
    sample_ids = [f for f, _ in parse(SAMPLE.read_bytes())[1:]]
    assert [f for f, _ in rows[1:]] == sample_ids == sorted(p.name for p in TEST.iterdir() if p.is_dir())
    vm = read(VOL / 'manifest.json')['files']
    assert all(sha(p) == vm[p.name] for p in (MASKS, STAGE2, VOL / 'features.py'))
    assert sha(STAGE2) == read(VOL / 'test_integrity.json')['candidate_sha256']
    mm = read(MAT / 'manifest.json')['files']
    assert all(sha(p) == mm[p.name] for p in (HEAD, MAT / 'material_classifier.py', MAT / 'encoder.json'))
    checked(HEAD)
    checked(WEIGHT)
    for (f, candidate), (previous_f, previous) in zip(rows[1:], parse(STAGE2.read_bytes())[1:], strict=True):
        assert f == previous_f and (candidate == 'none') == (previous == 'none')
        if candidate != 'none':
            assert all(t.split(' ')[:9] == u.split(' ')[:9] for t, u in zip(candidate.split(';'), previous.split(';'), strict=True))
    with gzip.open(MASKS, 'rt') as file:
        mask_rows = [json.loads(line) for line in file]
    masks = {r['frame_id']: r for r in mask_rows}
    assert len(masks) == len(mask_rows) and set(masks) == {f for f, line in rows[1:] if line != 'none'}
    cam = read(DATA / 'camera.json')
    sys.path.insert(0, str(VOL))
    sys.path.insert(0, str(MAT))
    from features import features, polygon_mask, read_frame
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    embedding_identity = [BASE_SHA, sha(MASKS), sha(WEIGHT)]
    embedding_cache = {}
    reused_embeddings = EMBEDDINGS.is_file()
    if reused_embeddings:
        with np.load(EMBEDDINGS, allow_pickle=False) as cached:
            assert cached['identity'].tolist() == embedding_identity
            embedding_cache = {(str(f), int(j)): z for f,j,z in zip(cached['frames'], cached['detections'], cached['vectors'], strict=True)}
    else:
        from material_classifier import MaterialClassifier
        model = MaterialClassifier(HEAD)
        assert all(p.device.type == 'cpu' and not p.requires_grad for p in model.encoder.parameters())
        assert not model.encoder.training
    with np.load(HEAD, allow_pickle=False) as head:
        classes, weight, bias = head['classes'], head['weight'], head['bias']
        assert set(classes) == set(scorer.MATERIALS) and len(set(head['fit_objects'])) == 32
    print('Using cached test masks and CatBoost volumes. Frozen DINOv2 train32 on CPU.', flush=True)
    details = []
    changes = {}
    new_rows = [rows[0]]
    for frame_no, (frame, cell) in enumerate(rows[1:], 1):
        if cell == 'none':
            new_rows.append([frame, cell])
            continue
        folder = TEST / frame
        geometry_error = rgb_error = ''
        try:
            depth, plane = read_frame(folder, cam)
        except (OSError, AssertionError, AttributeError, KeyError, TypeError, ValueError, cv2.error) as err:
            depth = plane = None
            geometry_error = type(err).__name__
        rgb = None
        if not reused_embeddings:
            color = folder / 'color.png'
            if not color.is_file():
                color = color.with_suffix('.jpg')
            try:
                with Image.open(color) as image:
                    rgb = image.convert('RGB')
            except OSError as err:
                rgb_error = type(err).__name__
        new_objects = []
        for j, (token, det) in enumerate(zip(cell.split(';'), masks[frame]['detections'], strict=True)):
            t = token.split(' ')
            assert len(t) == 10 and det['detection_index'] == j
            assert t[:6] == [*(f'{x:.1f}' for x in det['box']), f'{det["score"]:.3f}', det['form']], (frame, j)
            geo = {}
            if depth is not None:
                geo, _ = features(depth, plane, polygon_mask(depth, det), det['box'], det['form'], cam)
            key = frame, j
            z = embedding_cache.get(key)
            if rgb is not None:
                z = model.embed(rgb, det['box'])
                embedding_cache[key] = z
            probabilities = {} if z is None else dict(zip(('p_' + c for c in classes), softmax(weight @ z + bias), strict=True))
            rule = eligibility(probabilities, geo.get('q95_m'), geo.get('base_fallback'))
            old = t[7]
            if rule['eligible']:
                t[7] = format(Decimal(old) * Decimal(str(FACTOR)), 'f')
                changes[key] = old, t[7]
            details.append(dict(frame=frame, det=j, **rule, geometry_error=geometry_error, rgb_error=rgb_error,
                                volume_A_ml=old, volume_B_ml=t[7], changed=old != t[7]))
            new_objects.append(' '.join(t))
        new_rows.append([frame, ';'.join(new_objects)])
        if frame_no % 20 == 0:
            print(f'{frame_no}/137 frames, {len(details)} detections, {len(changes)} changes', flush=True)
    if not reused_embeddings and embedding_cache:
        keys = list(embedding_cache)
        np.savez_compressed(EMBEDDINGS, identity=embedding_identity, frames=[f for f,j in keys],
                            detections=[j for f,j in keys], vectors=np.stack([embedding_cache[k] for k in keys]))
    blob = render(new_rows)
    restored, total, empty = verify(parse(blob), rows, sample_ids, cam, changes)
    assert restored == raw == BASE.read_bytes()
    assert len(details) == total and len(changes) == sum(d['changed'] for d in details)
    assert all(d['eligible'] == d['changed'] for d in details)
    assert not DEST.exists() or DEST.read_bytes() == blob
    candidate_tmp = OUT / (DEST.name + '.tmp')
    candidate_tmp.write_bytes(blob)
    assert candidate_tmp.read_bytes() == blob
    candidate_tmp.replace(DEST)
    assert sha(DEST) == hashlib.sha256(blob).hexdigest() and sha(BASE) == BASE_SHA
    write_csv('test_detections.csv', details)
    if changes:
        write_csv('test_changes.csv', [dict(frame=f, det=j, volume_A_ml=a, volume_B_ml=b) for (f,j),(a,b) in changes.items()])
    integrity = dict(decision='ACCEPT', candidate=str(DEST.relative_to(W)), candidate_sha256=sha(DEST),
                     baseline=str(BASE.relative_to(W)), baseline_sha256=sha(BASE), restored_sha256=hashlib.sha256(restored).hexdigest(),
                     frames=len(sample_ids), official_parser_rows=len(sample_ids), empty_frames=empty, detections=total,
                     changed_predictions=len(changes), frames_changed=len({f for f,j in changes}),
                     valid_probabilities=sum(d['probabilities_valid'] for d in details), valid_heights=sum(d['height_valid'] for d in details),
                     both_valid=sum(d['probabilities_valid'] and d['height_valid'] for d in details),
                     nonvolume_token_changes=0, exact_restore=True, used_test_labels=False, submitted=False,
                     source_sha256={str(p.relative_to(W)): sha(p) for p in (MASKS, STAGE2, SAMPLE, HEAD, WEIGHT, VOL/'features.py', W/'datasets/score.py')},
                     embedding_inference='frozen CPU' if not reused_embeddings else 'existing stage8 test embedding cache',
                     base_volume='existing Stage 3 CSV CatBoost volume tokens', output_rounding='exact decimal product')
    write_json('test_integrity.json', integrity)
    print('VERIFIED', json.dumps(integrity, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
