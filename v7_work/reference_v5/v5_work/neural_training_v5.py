"""Fixed, fold-local neural ablations for Track 2 v5.

This module never opens a dataset, held-out array, prediction file, or prior
checkpoint. The caller supplies fitting rows only and owns source alignment,
artifact registration and scoring. Every returned model is the unchanged v3
SpatialLogVarianceNet; auxiliary reconstruction/domain heads are training-only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bci_work"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neural_cov import SpatialLogVarianceNet


MODEL_CONFIG = dict(n_bands=14, n_chans=41, n_filters=16, n_classes=3,
                    dropout=.5, include_diagonal=True, hidden=0)
SUPERVISED = dict(optimizer="AdamW", learning_rate=.002, weight_decay=.01,
                  batch_size=96, label_smoothing=.05, channel_gain_std=.08,
                  orthogonality_weight=.001, gradient_clip=5.,
                  cosine_horizon=120, minimum_learning_rate=.0001)
RECIPES = {
    "baseline": dict(description="Exact v3 fixed-epoch neural training"),
    "augmentation": dict(
        description="Baseline plus covariance masking and cross-subject mixup",
        channel_mask_probability=.10, paired_band_mask_probability=.10,
        channel_replacement="independent unit-variance axis",
        band_replacement="identity covariance", preserve_broadband_bank=True,
        mixup_alpha=.2, mixup_probability=.5,
        augmentation_seed_offset=300000,
    ),
    "adversarial": dict(
        description="Baseline plus a training-only subject-adversarial head",
        gradient_reversal_strength=.10, domain_loss_weight=1.,
        head_hidden=64, head_activation="GELU", head_dropout=0.,
        domain_weighting="equal fitting-subject totals",
        head_seed_offset=200000,
    ),
    "masked_ssl": dict(
        description="Fit-only masked log-power reconstruction before baseline fine-tuning",
        pretrain_epochs=30, pretrain_learning_rate=.001,
        pretrain_weight_decay=.01, pretrain_batch_size=96,
        pretrain_cosine_horizon=30, pretrain_minimum_learning_rate=.0001,
        channel_mask_probability=.15, paired_band_mask_probability=.20,
        channel_replacement="independent unit-variance axis",
        band_replacement="identity covariance", preserve_broadband_bank=True,
        reconstruction_hidden=128, reconstruction_activation="GELU",
        reconstruction_target="log diagonal covariance, fit-only coordinate standardization",
        reconstruction_loss="SmoothL1(beta=1), masked entries only",
        reconstruction_scale_floor=.1, gradient_clip=5.,
        decoder_seed_offset=100000, pretrain_order_seed_offset=100001,
        retained_parameters=["spatial_weight"],
        supervised_restart="same-seed fresh model; replace only pretrained spatial_weight",
        labels_used_in_pretraining=False,
    ),
}


def recipe_definitions():
    """Return JSON-compatible definitions without accessing EEG or labels."""
    return copy.deepcopy(dict(model=MODEL_CONFIG, supervised=SUPERVISED,
                              recipes=RECIPES))


def _subject_ids(values):
    return np.array([str(x).removeprefix("sub-") for x in values])


def mask_covariances(covariances, rng, channel_probability, band_probability):
    """PSD-preserving masks, shared between the two temporal halves.

    A masked channel's row/column is removed and its diagonal is replaced by
    one. A masked non-broadband bank becomes identity. The seventh bank is
    broadband and remains available. Returns the exact masked target entries.
    Only the supplied batch and an independent NumPy generator are consulted.
    """
    n, n_bands, n_chans, _ = covariances.shape
    if n_bands != 14 or n_chans != 41:
        raise ValueError("The fixed v5 mask expects 14 banks and 41 channels")
    channel_mask = rng.random((n, n_chans)) < channel_probability
    all_masked = channel_mask.all(axis=1)
    channel_mask[all_masked, 0] = False
    bank_mask = rng.random((n, 7)) < band_probability
    bank_mask[:, 6] = False
    bank_mask = np.tile(bank_mask, (1, 2))
    channel = torch.as_tensor(channel_mask, device=covariances.device)
    keep = (~channel).to(covariances.dtype)
    out = covariances * keep[:, None, :, None] * keep[:, None, None, :]
    out = out + torch.diag_embed(channel.to(covariances.dtype))[:, None]
    bank = torch.as_tensor(bank_mask, device=covariances.device)
    identity = torch.eye(n_chans, dtype=covariances.dtype, device=covariances.device)
    out = torch.where(bank[:, :, None, None], identity, out)
    targets = bank[:, :, None] | channel[:, None, :]
    return out, targets.flatten(1)


def _cross_subject_mixup(batch, subject_ids, rng, alpha, probability):
    """Partners are drawn only from different subjects in this fitting batch."""
    partner = np.arange(len(batch))
    lam = np.ones(len(batch), dtype=np.float32)
    for i in range(len(batch)):
        eligible = np.flatnonzero(subject_ids != subject_ids[i])
        if len(eligible) and rng.random() < probability:
            partner[i] = int(rng.choice(eligible))
            lam[i] = float(rng.beta(alpha, alpha))
    mixed = batch * torch.from_numpy(lam)[:, None, None, None]
    mixed = mixed + batch[partner] * torch.from_numpy(1. - lam)[:, None, None, None]
    return mixed, partner, torch.from_numpy(lam)


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, strength):
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def _classification_from_features(net, normalized):
    z = net.dropout(normalized)
    if net.hidden is not None:
        z = net.dropout(F.gelu(net.hidden(z)))
    return net.classifier(z)


def _masked_pretrain(net, covariances, seed):
    """No labels, loss weights, held-out values, or held-out identities accepted."""
    cfg = RECIPES["masked_ssl"]
    X = torch.from_numpy(covariances)
    target = torch.diagonal(X, dim1=-2, dim2=-1).clamp_min(1e-6).log().flatten(1)
    center = target.mean(dim=0)
    scale = target.std(dim=0, correction=0).clamp_min(cfg["reconstruction_scale_floor"])
    target = (target - center) / scale
    n_features = net.normalizer.num_features
    # The temporary head does not perturb the backbone's paired RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + cfg["decoder_seed_offset"])
        decoder = nn.Sequential(nn.Linear(n_features, cfg["reconstruction_hidden"]),
                                nn.GELU(), nn.Linear(cfg["reconstruction_hidden"], target.shape[1]))
    parameters = [net.spatial_weight]
    parameters += list(net.normalizer.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg["pretrain_learning_rate"],
                                  weight_decay=cfg["pretrain_weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["pretrain_cosine_horizon"],
        eta_min=cfg["pretrain_minimum_learning_rate"])
    rng = np.random.default_rng(seed + cfg["pretrain_order_seed_offset"])
    history = []
    for epoch in range(1, cfg["pretrain_epochs"] + 1):
        net.train(); decoder.train()
        order = rng.permutation(len(X))
        losses = []
        for start in range(0, len(X), cfg["pretrain_batch_size"]):
            idx = order[start:start + cfg["pretrain_batch_size"]]
            if len(idx) < 2:
                continue
            masked, entries = mask_covariances(
                X[idx], rng, cfg["channel_mask_probability"], cfg["paired_band_mask_probability"])
            # Guarantee a reconstruction target without consulting its value.
            if not entries.any():
                masked[:, 0] = torch.eye(41)
                entries[:, :41] = True
            reconstruction = decoder(net.normalizer(net.features(masked)))
            error = F.smooth_l1_loss(reconstruction, target[idx], beta=1., reduction="none")
            loss = (error * entries).sum() / entries.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % 10 == 0 or epoch == cfg["pretrain_epochs"]:
            record = dict(epoch=epoch, reconstruction_loss=float(np.mean(losses)))
            history.append(record)
            print("pretraining", json.dumps(record), flush=True)
    return net.spatial_weight.detach().clone(), dict(
        history=history, n_windows=len(X), labels_used=False,
        target_center=center.tolist(), target_scale=scale.tolist(),
        retained_parameters=["spatial_weight"])


def train_model(c_train, y_train, weights, subject_ids, config, seed, epochs,
                recipe="baseline", output=None, *, heldout_subject=None,
                schedule_epochs=120):
    """Train on supplied fitting rows and return (net, history, training_info).

    ``output``, when supplied, names a JSON log only. The orchestrator saves
    checkpoints together with fold-specific source references and provenance.
    The baseline follows crossvalidate_neural.fit_fixed operation-for-operation.
    """
    if recipe not in RECIPES:
        raise ValueError("Unknown fixed v5 recipe: " + str(recipe))
    if dict(config) != MODEL_CONFIG:
        raise ValueError("V5 neural ablations require the matched v3 architecture")
    if schedule_epochs != 120 or int(epochs) < 1:
        raise ValueError("Use positive fixed epochs and the unchanged 120-epoch cosine horizon")
    c_train = np.asarray(c_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    subjects = _subject_ids(subject_ids)
    if c_train.shape[1:] != (14, 41, 41) or len(c_train) < 2:
        raise ValueError("Expected at least two fitting covariance examples")
    if any(len(x) != len(c_train) for x in (y_train, weights, subjects)):
        raise ValueError("Fitting arrays have different lengths")
    if heldout_subject is not None and str(heldout_subject).removeprefix("sub-") in set(subjects):
        raise ValueError("Held-out participant reached neural fitting/pretraining")
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isin(y_train, [0, 1, 2]).all():
        raise ValueError("Invalid fitting weights or class labels")
    seed = int(seed); epochs = int(epochs)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SpatialLogVarianceNet(**config)
    ssl_info = None
    if recipe == "masked_ssl":
        pretrained, ssl_info = _masked_pretrain(net, c_train, seed)
        # Reset classifier, affine BN parameters, BN statistics and RNG to the
        # baseline initialization. Only the learned spatial filters survive.
        torch.manual_seed(seed)
        net = SpatialLogVarianceNet(**config)
        with torch.no_grad():
            net.spatial_weight.copy_(pretrained)
    parameters = list(net.parameters())
    domain_head = None
    if recipe == "adversarial":
        cfg = RECIPES[recipe]
        domain_subjects, domain_indices, counts = np.unique(subjects, return_inverse=True, return_counts=True)
        domain_targets = torch.from_numpy(domain_indices.astype(np.int64))
        domain_weights = torch.from_numpy((len(subjects) / (len(counts) * counts[domain_indices])).astype(np.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + cfg["head_seed_offset"])
            domain_head = nn.Sequential(nn.Linear(net.normalizer.num_features, cfg["head_hidden"]),
                                        nn.GELU(), nn.Linear(cfg["head_hidden"], len(domain_subjects)))
        parameters += list(domain_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_epochs, eta_min=.0001)
    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=.05)
    X, Y, W = torch.from_numpy(c_train), torch.from_numpy(y_train), torch.from_numpy(weights.astype(np.float32))
    augmentation_rng = np.random.default_rng(seed + RECIPES["augmentation"]["augmentation_seed_offset"])
    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        if domain_head is not None:
            domain_head.train()
        order = rng.permutation(len(X))
        losses, domain_losses = [], []
        for start in range(0, len(X), 96):
            idx = order[start:start + 96]
            if len(idx) < 2:
                continue
            batch = X[idx].clone()
            gains = torch.exp(torch.randn(len(idx), 1, batch.shape[-1]) * .08)
            batch = batch * gains.unsqueeze(-1) * gains.unsqueeze(-2)
            if recipe == "augmentation":
                cfg = RECIPES[recipe]
                batch, _ = mask_covariances(batch, augmentation_rng,
                                            cfg["channel_mask_probability"], cfg["paired_band_mask_probability"])
                batch, partner, lam = _cross_subject_mixup(batch, subjects[idx], augmentation_rng,
                                                          cfg["mixup_alpha"], cfg["mixup_probability"])
                logits = net(batch)
                loss = (lam * W[idx] * criterion(logits, Y[idx]) +
                        (1. - lam) * W[idx][partner] * criterion(logits, Y[idx][partner])).mean()
            elif recipe == "adversarial":
                cfg = RECIPES[recipe]
                z = net.normalizer(net.features(batch))
                logits = _classification_from_features(net, z)
                loss = (criterion(logits, Y[idx]) * W[idx]).mean()
                domain_logits = domain_head(_ReverseGradient.apply(z, cfg["gradient_reversal_strength"]))
                domain_loss = (F.cross_entropy(domain_logits, domain_targets[idx], reduction="none") * domain_weights[idx]).mean()
                loss = loss + cfg["domain_loss_weight"] * domain_loss
                domain_losses.append(float(domain_loss.detach()))
            else:
                logits = net(batch)
                loss = (criterion(logits, Y[idx]) * W[idx]).mean()
            w = F.normalize(net.spatial_weight, dim=-1)
            gram = w @ w.transpose(-1, -2)
            eye = torch.eye(gram.shape[-1])
            loss = loss + .001 * ((gram - eye) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Preserve backbone clipping; the independent head has its own cap.
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            if domain_head is not None:
                nn.utils.clip_grad_norm_(domain_head.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % 15 == 0:
            record = dict(epoch=epoch, training_loss=float(np.mean(losses)))
            if domain_losses:
                record["domain_loss"] = float(np.mean(domain_losses))
            history.append(record)
            print("training", json.dumps(record), flush=True)
    fit_subjects = sorted(set(map(str, subjects)))
    info = dict(recipe=recipe, seed=seed, epochs=epochs, schedule_epochs=schedule_epochs,
                model_config=copy.deepcopy(config), supervised=copy.deepcopy(SUPERVISED),
                recipe_config=copy.deepcopy(RECIPES[recipe]), n_fitting_windows=len(c_train),
                fitted_subjects=fit_subjects, normalization_subjects=fit_subjects,
                pretraining_subjects=fit_subjects if recipe == "masked_ssl" else [],
                heldout_subject=None if heldout_subject is None else str(heldout_subject).removeprefix("sub-"),
                checkpoint_selection="fixed epoch; no held-out metrics or early stopping",
                pretraining=ssl_info, deployment_auxiliary_heads=False)
    if output is not None:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(history=history, training_info=info), indent=2) + "\n")
    return net.eval(), history, info

