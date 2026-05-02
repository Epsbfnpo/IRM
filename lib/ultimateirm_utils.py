import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class TemporalMemoryBank:
    def __init__(self, num_classes, momentum=0.9):
        self.num_classes = num_classes
        self.momentum = momentum
        self.store = {}

    def update(self, uid_list, probs, pseudo_labels, confidences, cluster_ids=None):
        probs = probs.detach().cpu().numpy()
        pseudo_labels = pseudo_labels.detach().cpu().numpy()
        confidences = confidences.detach().cpu().numpy()
        cids = cluster_ids if cluster_ids is not None else [None] * len(uid_list)
        for i, uid in enumerate(uid_list):
            p = probs[i]
            if uid not in self.store:
                self.store[uid] = {"seen": 0, "prob_ema": np.zeros(self.num_classes), "prob_sq_ema": np.zeros(self.num_classes), "last_pseudo": -1, "flip_count": 0, "last_confidence": 0.0, "last_cluster": -1, "cluster_switches": 0}
            s = self.store[uid]
            s["seen"] += 1
            m = self.momentum
            s["prob_ema"] = m * s["prob_ema"] + (1 - m) * p
            s["prob_sq_ema"] = m * s["prob_sq_ema"] + (1 - m) * (p ** 2)
            if s["last_pseudo"] >= 0 and s["last_pseudo"] != int(pseudo_labels[i]): s["flip_count"] += 1
            s["last_pseudo"] = int(pseudo_labels[i]); s["last_confidence"] = float(confidences[i])
            if cids[i] is not None:
                cid = int(cids[i])
                if s["last_cluster"] >= 0 and s["last_cluster"] != cid: s["cluster_switches"] += 1
                s["last_cluster"] = cid

    def get_prob_var(self, uid_list):
        out = []
        for uid in uid_list:
            s = self.store.get(uid)
            if not s: out.append(1.0); continue
            v = np.maximum(0.0, s["prob_sq_ema"] - s["prob_ema"] ** 2)
            out.append(float(v.mean()))
        return torch.tensor(out, dtype=torch.float32)

    def export_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["uid", "seen", "last_pseudo", "flip_count", "last_confidence", "last_cluster", "cluster_switches"])
            for uid, s in self.store.items(): w.writerow([uid, s["seen"], s["last_pseudo"], s["flip_count"], s["last_confidence"], s["last_cluster"], s["cluster_switches"]])


def confidence_A(probs, **kwargs):
    return probs.max(dim=1)[1], probs.max(dim=1)[0]

def confidence_B(probs, uid_list, memory_bank, gamma=5.0, **kwargs):
    pseudo = probs.max(dim=1)[1]; tmp = memory_bank.get_prob_var(uid_list).to(probs.device); return pseudo, torch.exp(-gamma * tmp)

def confidence_C(probs, beta=5.0, dispersion_thresh=0.05, **kwargs):
    maxp, pseudo = probs.max(dim=1); residual = probs.clone(); residual[torch.arange(len(pseudo), device=probs.device), pseudo] = 0.0
    rv = residual.var(dim=1, unbiased=False); return pseudo, maxp * torch.exp(-beta * F.relu(rv - dispersion_thresh))

def confidence_D(probs, uid_list, memory_bank, gamma=5.0, beta=5.0, dispersion_thresh=0.05, w_max=1.0, w_tmp=1.0, w_res=1.0, **kwargs):
    maxp, pseudo = probs.max(dim=1); tmp = torch.exp(-gamma * memory_bank.get_prob_var(uid_list).to(probs.device)); residual = probs.clone(); residual[torch.arange(len(pseudo), device=probs.device), pseudo] = 0.0
    rv = residual.var(dim=1, unbiased=False); res = torch.exp(-beta * F.relu(rv - dispersion_thresh)); return pseudo, (maxp ** w_max) * (tmp ** w_tmp) * (res ** w_res)

def compute_confidence(mode, probs, uid_list=None, memory_bank=None, **kwargs):
    return {"A": confidence_A, "B": confidence_B, "C": confidence_C, "D": confidence_D}[mode](probs, uid_list=uid_list, memory_bank=memory_bank, **kwargs)

def cluster_A(features, k, seed=0, **kwargs):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10); ids = km.fit_predict(features); return ids, {"centers": km.cluster_centers_}

def cluster_B(features, k, seed=0, **kwargs):
    gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full"); ids = gmm.fit_predict(features); return ids, {"means": gmm.means_}

def cluster_C(features, k, seed=0, **kwargs):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10); km.fit(features)
    gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full", means_init=km.cluster_centers_); ids = gmm.fit_predict(features); return ids, {"means": gmm.means_}

def run_clustering(mode, features, k, seed=0, **kwargs):
    return {"A": cluster_A, "B": cluster_B, "C": cluster_C}[mode](features, k, seed=seed, **kwargs)


class GuidedTMPMasker:
    def __init__(self, patch_size=32, topk_ratio=0.4, mask_ratio=0.5):
        self.patch_size = patch_size
        self.topk_ratio = topk_ratio
        self.mask_ratio = mask_ratio

    def apply(self, x, confidence):
        # patch-level saliency proxy: channel-energy map
        b, c, h, w = x.shape
        p = self.patch_size
        out = x.clone()
        for i in range(b):
            if confidence[i] <= 0:
                continue
            xi = out[i]
            patches = xi.unfold(1, p, p).unfold(2, p, p)  # [c, ph, pw, p, p]
            ph, pw = patches.shape[1], patches.shape[2]
            energy = patches.abs().mean(dim=(0, 3, 4)).reshape(-1)
            k = max(1, int(len(energy) * self.topk_ratio))
            top_idx = torch.topk(energy, k=k).indices
            m = max(1, int(k * self.mask_ratio))
            chosen = top_idx[torch.randperm(k, device=top_idx.device)[:m]]
            for idx in chosen.tolist():
                r, col = divmod(idx, pw)
                xi[:, r * p:(r + 1) * p, col * p:(col + 1) * p] = 0.0
        return out


def compute_env_weights(unlabeled_cluster_sizes, lambda_unlabeled=1.0, size_norm=True):
    if len(unlabeled_cluster_sizes) == 0:
        return {}
    if not size_norm:
        w = lambda_unlabeled / len(unlabeled_cluster_sizes)
        return {k: w for k in unlabeled_cluster_sizes}
    inv = {k: 1.0 / max(1, v) for k, v in unlabeled_cluster_sizes.items()}
    z = sum(inv.values()) + 1e-8
    return {k: lambda_unlabeled * inv[k] / z for k in inv}
