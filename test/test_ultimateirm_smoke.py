import torch
from domainbed.lib.ultimateirm_utils import TemporalMemoryBank, compute_confidence, run_clustering


def test_confidence_modes_smoke():
    probs = torch.tensor([[0.8, 0.1, 0.1], [0.4, 0.3, 0.3], [0.2, 0.7, 0.1]], dtype=torch.float32)
    bank = TemporalMemoryBank(num_classes=3, momentum=0.9)
    uid = ["a", "b", "c"]
    pseudo = torch.tensor([0, 0, 1])
    conf = torch.tensor([0.8, 0.4, 0.7])
    bank.update(uid, probs, pseudo, conf)
    for mode in ["A", "B", "C", "D"]:
        p, c = compute_confidence(mode=mode, probs=probs, uid_list=uid, memory_bank=bank, gamma=5.0, beta=5.0, dispersion_thresh=0.05, w_max=1.0, w_tmp=1.0, w_res=1.0)
        assert p.shape[0] == 3
        assert c.shape[0] == 3
        assert torch.isfinite(c).all()


def test_cluster_modes_smoke():
    x = torch.randn(16, 8).numpy()
    for mode in ["A", "B", "C"]:
        ids, meta = run_clustering(mode=mode, features=x, k=3, seed=0)
        assert len(ids) == 16
