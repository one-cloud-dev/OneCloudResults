"""Замороженный DINOv2 и две сохранённые линейные головы этапа 1."""

import math
import sys

import numpy as np
import torch
from scipy.special import softmax
from torchvision import transforms as T


class LinearHead:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as head:
            self.classes = head["classes"]
            self.weight = head["weight"]
            self.bias = head["bias"]

    def classify(self, embedding):
        return str(self.classes[(self.weight @ embedding + self.bias).argmax()])

    def probabilities(self, embedding):
        return dict(zip(self.classes, softmax(self.weight @ embedding + self.bias), strict=True))


class DinoEncoder:
    def __init__(self, source, weights):
        sys.path.insert(0, str(source.resolve()))
        from dinov2.hub.backbones import dinov2_vits14

        self.encoder = dinov2_vits14(pretrained=False)
        self.encoder.load_state_dict(
            torch.load(weights, weights_only=True, map_location="cpu"), strict=True
        )
        self.encoder.requires_grad_(False).eval()
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    @torch.inference_mode()
    def embed(self, rgb_image, box):
        x, y, right, bottom = box
        limits = (
            max(0, math.floor(x)), max(0, math.floor(y)),
            min(rgb_image.width, math.ceil(right)), min(rgb_image.height, math.ceil(bottom)),
        )
        crop = rgb_image.convert("RGB").crop(limits)
        return self.encoder(self.preprocess(crop).unsqueeze(0)).numpy()[0]
