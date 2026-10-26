"""Small pure-torch EEGNet and shallow square/log EEG decoders.

These independently implemented variants follow the main architectural ideas
of EEGNet (Lawhern et al. 2018) and Shallow ConvNet (Schirrmeister et al. 2017).
They are not claimed to be bit-identical to a particular Braindecode release.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def window_zscore(x):
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.square().mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()


class EEGNetCompact(nn.Module):
    def __init__(self, n_chans=41, n_times=480, n_classes=3, dropout=.5, f1=8, depth=2):
        super().__init__()
        f2 = f1 * depth
        self.temporal = nn.Conv2d(1, f1, (1, 61), padding=(0, 30), bias=False)
        self.bn1 = nn.BatchNorm2d(f1, momentum=.01)
        self.spatial = nn.Conv2d(f1, f2, (n_chans, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2, momentum=.01)
        self.depthwise = nn.Conv2d(f2, f2, (1, 15), padding=(0, 7), groups=f2, bias=False)
        self.pointwise = nn.Conv2d(f2, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2, momentum=.01)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(f2 * (n_times // 4 // 8), n_classes)
        self.config = dict(n_chans=n_chans, n_times=n_times, n_classes=n_classes, dropout=dropout, f1=f1, depth=depth)

    def forward(self, x, normalize=True):
        if normalize:
            x = window_zscore(x)
        x = x[:, None]
        x = self.bn1(self.temporal(x))
        x = F.elu(self.bn2(self.spatial(x)))
        x = self.dropout(F.avg_pool2d(x, (1, 4)))
        x = F.elu(self.bn3(self.pointwise(self.depthwise(x))))
        x = self.dropout(F.avg_pool2d(x, (1, 8)))
        return self.classifier(x.flatten(1))

    def constrain(self):
        with torch.no_grad():
            w = self.spatial.weight
            norm = w.flatten(1).norm(dim=1)
            w *= (1. / norm.clamp_min(1.)).view(-1, 1, 1, 1)
            w = self.classifier.weight
            norm = w.norm(dim=1, keepdim=True)
            w *= .5 / norm.clamp_min(.5)


class ShallowSquareLogNet(nn.Module):
    def __init__(self, n_chans=41, n_times=480, n_classes=3, dropout=.5, n_filters=24):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, 25), bias=True)
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters, momentum=.1)
        self.dropout = nn.Dropout(dropout)
        out_times = (n_times - 25 + 1 - 75) // 15 + 1
        self.classifier = nn.Linear(n_filters * out_times, n_classes)
        self.config = dict(n_chans=n_chans, n_times=n_times, n_classes=n_classes, dropout=dropout, n_filters=n_filters)

    def forward(self, x, normalize=True):
        if normalize:
            x = window_zscore(x)
        x = self.bn(self.spatial(self.temporal(x[:, None])))
        x = F.avg_pool2d(x.square(), (1, 75), stride=(1, 15)).clamp_min(1e-6).log()
        return self.classifier(self.dropout(x).flatten(1))

    def constrain(self):
        with torch.no_grad():
            w = self.spatial.weight
            norm = w.flatten(1).norm(dim=1)
            w *= 2. / norm.clamp_min(2.).view(-1, 1, 1, 1)
            w = self.classifier.weight
            norm = w.norm(dim=1, keepdim=True)
            w *= .5 / norm.clamp_min(.5)


def make_raw_model(kind, config=None):
    config = {} if config is None else config
    if kind == "eegnet":
        return EEGNetCompact(**config)
    if kind == "shallow":
        return ShallowSquareLogNet(**config)
    raise ValueError(kind)


class RawEEGEnsemble:
    """Deepcopy-safe stateless prediction wrapper for selected checkpoints."""
    def __init__(self, checkpoints, device="cpu"):
        self.device = str(device)
        self.models = []
        for ckpt in checkpoints:
            model = make_raw_model(ckpt["model_type"], ckpt["model_config"])
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            self.models.append(model)

    def reset_state(self):
        for model in self.models:
            model.eval()

    def predict_proba(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim != 3 or tuple(x.shape[1:]) != (41, 480):
            raise ValueError("Expected (B,41,480)")
        if len(x) == 0:
            return np.empty((0, 3), dtype=np.float32)
        with torch.inference_mode():
            p = torch.stack([torch.softmax(model(x), dim=-1) for model in self.models]).mean(0)
        return p.cpu().numpy()

    def predict(self, x):
        return self.predict_proba(x).argmax(1).astype(np.int64)

