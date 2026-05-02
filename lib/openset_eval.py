import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def _collect_scores(algorithm, loader, device):
    y_true, scores = [], []

    algorithm.eval()
    with torch.no_grad():
        for batch in loader:
            uid = None

            if len(batch) == 3:
                x, y, uid = batch
            else:
                x, y = batch

            x = x.to(device)

            if hasattr(algorithm, "score_ood"):
                s = algorithm.score_ood(x, uid=uid)
            else:
                logits = algorithm.predict(x)
                probs = torch.softmax(logits, dim=1)
                s = 1.0 - probs.max(dim=1)[0]

            scores.extend(s.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

    return np.array(y_true), np.array(scores)


def evaluate_ood(algorithm, id_loader, ood_loader, device):
    _, id_scores = _collect_scores(algorithm, id_loader, device)
    _, ood_scores = _collect_scores(algorithm, ood_loader, device)

    y = np.concatenate([
        np.zeros_like(id_scores),
        np.ones_like(ood_scores),
    ])
    s = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y, s)
    aupr = average_precision_score(y, s)

    thr = np.percentile(id_scores, 95)
    fpr95 = float((ood_scores <= thr).mean())

    return {
        "ood_auroc": float(auroc),
        "ood_aupr": float(aupr),
        "ood_fpr95": float(fpr95),
        "id_ood_score_mean": float(id_scores.mean()),
        "ood_ood_score_mean": float(ood_scores.mean()),
    }
