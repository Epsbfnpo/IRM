import json, os, tempfile
from domainbed.lib.openset_manifest import load_samples, index_samples_by_uid, load_uid_split, select_samples


def test_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        sfile = os.path.join(d, 'samples.jsonl')
        split = os.path.join(d, 'split.json')
        rows = [
            {"uid":"a:0","path":"/tmp/a.jpg","label":0,"class_name":"book","env":"A_real","source":"DomainNet","is_ood":False},
            {"uid":"b:0","path":"/tmp/b.jpg","label":-1,"class_name":"__ood__","env":"B_ood","source":"SVIRO","is_ood":True},
        ]
        with open(sfile,'w') as f:
            for r in rows: f.write(json.dumps(r)+'\n')
        with open(split,'w') as f: json.dump(["a:0"], f)
        samples = load_samples(sfile)
        idx = index_samples_by_uid(samples)
        uids = load_uid_split(split)
        sub = select_samples(idx, uids)
        assert len(sub) == 1 and sub[0]['uid'] == 'a:0'
