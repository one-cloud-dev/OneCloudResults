"""Порядок вычислений и сериализация официальной отправки Astramis."""

import csv
import io
import json
import math
from dataclasses import dataclass
from decimal import Decimal

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .legacy import material_features, raw_volume_ml
from .models import MATERIALS, Models, predict_take, predict_volume
from .rgbd import features, polygon_mask, read_frame

FORMS = ("bulk", "solid")
STAGES = ("rf_baseline", "dinov2_material", "catboost_volume", "strict_take", "wood_volume")


@dataclass
class Detection:
    box: list
    score: float
    form: str
    contour_xy: np.ndarray

    def prefix(self):
        return [*(f"{value:.1f}" for value in self.box), f"{self.score:.3f}", self.form]

    def mask(self, depth):
        return polygon_mask(depth, {"contour_xy": self.contour_xy})


def detect(detector, color):
    result = detector(str(color), conf=.1, imgsz=640, iou=.5, device="cpu", verbose=False)[0]
    if result.masks is None:
        return []
    boxes = result.boxes
    return [
        Detection(box, confidence, FORMS[int(cls)], polygon.round().astype("int32"))
        for box, confidence, cls, polygon in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), result.masks.xy, strict=True
        )
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2
    ]


def correct_wood_volume(token, probabilities, geometry):
    # Порядок MATERIALS сохранён из официального scorer для разрешения равенства.
    values = [float(probabilities[material]) for material in MATERIALS]
    q95 = float(geometry["q95_m"])
    valid_probabilities = all(math.isfinite(p) and 0 <= p <= 1 for p in values)
    valid_probabilities &= math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-6)
    if valid_probabilities and math.isfinite(q95) and 0 <= q95 <= .030 and geometry["base_fallback"] == 0:
        top_two = sorted(MATERIALS, key=probabilities.get, reverse=True)[:2]
        if "wood" in top_two:
            return format(Decimal(token) * Decimal("0.75"), "f")
    return token


def predict_frame(folder, models, camera, densities):
    color = folder / "color.png"
    if not color.is_file():
        color = color.with_suffix(".jpg")
    if not color.is_file():
        raise FileNotFoundError(color)

    detections = detect(models.detector, color)
    if not detections:
        return dict.fromkeys(STAGES, "none")
    depth, plane = read_frame(folder, camera)
    rgb_for_rf = cv2.cvtColor(cv2.imread(str(color)), cv2.COLOR_BGR2RGB)
    height, width = rgb_for_rf.shape[:2]
    rows = {stage: [] for stage in STAGES}
    frame_features = [
        features(depth, plane, det.mask(depth), det.box, det.form, camera)[0]
        for det in detections
    ]
    volumes = predict_volume(models.volume, pd.DataFrame(frame_features))
    with Image.open(color) as image:
        for det, geometry, corrected_ml in zip(detections, frame_features, volumes, strict=True):
            x, y, right, bottom = det.box
            crop = rgb_for_rf[max(0, int(y)):min(height, math.ceil(bottom)),
                              max(0, int(x)):min(width, math.ceil(right))]
            rf_label = str(models.legacy_material.predict([material_features(crop)])[0])
            density = densities.get((det.form, rf_label), 1.)
            raw_ml = raw_volume_ml(depth, det.mask(depth) > 0, plane, camera)
            legacy_ml = raw_ml if raw_ml is not None and raw_ml > 1 else 420.
            legacy_mass = legacy_ml * density
            mass = geometry["base_ml"] * density
            if (f"{legacy_ml:.0f}" != f"{geometry['base_ml']:.0f}"
                    or f"{legacy_mass:.0f}" != f"{mass:.0f}" or (legacy_mass <= 500) != (mass <= 500)):
                raise ValueError(f"Сверка геометрии этапов не прошла в кадре {folder.name}")

            embedding = models.encoder.embed(image, det.box)
            material = models.head_all40.classify(embedding)
            probabilities = models.head_train32.probabilities(embedding)
            take = str(predict_take(models.take, corrected_ml, mass, geometry, probabilities))

            prefix = det.prefix()
            raw_token = f"{legacy_ml:.0f}"
            mass_token = f"{legacy_mass:.0f}"
            volume_token = str(max(1, round(corrected_ml)))
            wood_token = correct_wood_volume(volume_token, probabilities, geometry)
            take_baseline = str(int(legacy_mass <= 500))
            rows["rf_baseline"].append([*prefix, rf_label, raw_token, mass_token, take_baseline])
            rows["dinov2_material"].append([*prefix, material, raw_token, mass_token, take_baseline])
            rows["catboost_volume"].append([*prefix, material, volume_token, mass_token, take_baseline])
            rows["strict_take"].append([*prefix, material, volume_token, mass_token, take])
            rows["wood_volume"].append([*prefix, material, wood_token, mass_token, take])
    return {stage: ";".join(" ".join(tokens) for tokens in predictions) for stage, predictions in rows.items()}


def render(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["image_id", "prediction"])
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def run(data, sample, model_dir, dino_source):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    camera = json.loads((data / "camera.json").read_text(encoding="utf-8"))
    with (data / "densities.csv").open(encoding="utf-8", newline="") as file:
        densities = {
            (row["form"], row["material"]): float(row["density_g_per_ml"])
            for row in csv.DictReader(file)
        }
    with sample.open(encoding="utf-8", newline="") as file:
        frame_ids = [row["image_id"] for row in csv.DictReader(file)]
    test = data / "real" / "test"
    folders = {folder.name: folder for folder in test.iterdir() if folder.is_dir()}
    if len(frame_ids) != len(folders) or set(frame_ids) != folders.keys():
        raise ValueError("Список тестовых каталогов расходится с sample_submission.csv")

    models = Models(model_dir, dino_source)
    rows = {stage: [] for stage in STAGES}
    for index, frame_id in enumerate(frame_ids, 1):
        predictions = predict_frame(folders[frame_id], models, camera, densities)
        for stage, cell in predictions.items():
            rows[stage].append((frame_id, cell))
        if index % 20 == 0 or index == len(frame_ids):
            print(f"{index}/{len(frame_ids)} кадров", flush=True)
    return {stage: render(stage_rows) for stage, stage_rows in rows.items()}
