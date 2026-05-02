import torch
from PIL import Image
import tempfile, os
from domainbed.datasets import ManifestUnlabeledDataset


def _img(path):
    Image.new('RGB', (32,32), color='red').save(path)


def test_unlabeled_dataset_returns_five_tuple():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'a.jpg'); _img(p)
        ds = ManifestUnlabeledDataset([{"path":p,"label":1,"uid":"u1"}], weak_transform=lambda x: torch.zeros(3,4,4), strong_transform=lambda x: torch.ones(3,4,4), mask_transform=lambda x: torch.full((3,4,4),2.0))
        xw,xs,xm,y,u = ds[0]
        assert xw.shape == xs.shape == xm.shape
        assert y.item() == 1 and u == 'u1'
