"""Фиксированные веса и предсказания объёма и take."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from ultralytics import YOLO

from .material import DinoEncoder, LinearHead
from .rgbd import FEATURES

MATERIALS = ("metal", "wood", "plastic", "glass", "paper", "mineral")
TAKE_COLUMNS = [
    "log_volume", "log_mass", "mass_over_limit",
    *["p_" + material for material in sorted(MATERIALS)],
    "solid", "projected_m2",
]


class Models:
    def __init__(self, directory, dino_source):
        manifest = json.loads((directory / "artifacts.json").read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            with (directory / name).open("rb") as file:
                actual = hashlib.file_digest(file, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"SHA-256 {name}: {actual}, ожидалось {expected}")

        self.detector = YOLO(str(directory / "best.pt"))
        self.legacy_material = joblib.load(directory / "material.joblib")
        self.encoder = DinoEncoder(dino_source, directory / "dinov2_vits14_pretrain.pth")
        self.head_all40 = LinearHead(directory / "head_all40.npz")
        self.head_train32 = LinearHead(directory / "head_train32.npz")
        self.volume = CatBoostRegressor()
        self.volume.load_model(str(directory / "volume_train32.cbm"))
        self.take = joblib.load(directory / "take_train32_strict.joblib")


def predict_volume(model, frame):
    log_correction = model.predict(frame[FEATURES])
    base = frame.base_ml.to_numpy()
    good = (frame.base_fallback == 0) & np.isfinite(log_correction) & np.isfinite(base) & (base > 0)
    result = np.where(
        good,
        base * np.exp(np.clip(log_correction, -8, 8)),
        np.where(np.isfinite(base) & (base > 0), base, 420.),
    )
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Некорректный объём CatBoost")
    return result


def predict_take(model, corrected_ml, mass, geometry, probabilities):
    frame = pd.DataFrame([dict(
        corrected_ml=corrected_ml,
        mass=mass,
        solid=geometry["solid"],
        projected_m2=geometry["projected_m2"],
        **{"p_" + material: probability for material, probability in probabilities.items()},
    )])
    matrix = frame.assign(
        log_volume=np.log1p(frame.corrected_ml),
        log_mass=np.log1p(frame.mass),
        mass_over_limit=frame.mass / 500,
    )[TAKE_COLUMNS]
    if not np.isfinite(matrix).all().all():
        raise ValueError("Некорректные признаки take")
    return int(float(model.predict_proba(matrix)[0, 1]) > .5)
