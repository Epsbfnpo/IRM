import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def _collect_scores(algorithm, loader, device):
    y_true, scores = [], []
    algorithm.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            logits = algorithm.predict(x.to(device))
            probs = torch.softmax(logits, dim=1)
            s = probs.max(dim=1)[0]
            scores.extend(s.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())
    return np.array(y_true), np.array(scores)


def evaluate_ood(algorithm, id_loader, ood_loader, device):
    _, id_scores = _collect_scores(algorithm, id_loader, device)
    _, ood_scores = _collect_scores(algorithm, ood_loader, device)
    y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    s = np.concatenate([1 - id_scores, 1 - ood_scores])
    auroc = roc_auc_score(y, s)
    aupr = average_precision_score(y, s)
    thr = np.percentile(id_scores, 5)
    fpr95 = float((ood_scores >= thr).mean())
    return {"ood_auroc": float(auroc), "ood_aupr": float(aupr), "ood_fpr95": fpr95, "id_conf_mean": float(id_scores.mean()), "ood_conf_mean": float(ood_scores.mean())}
