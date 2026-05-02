#!/usr/bin/env python3
import argparse, json, os, random, shutil
from collections import Counter
from torchvision.datasets import ImageFolder

RHO_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]
ID_CLASSES = ["book", "clock", "keyboard", "lamp", "mug", "scissors"]


def collect_imagefolder_samples(root, keep_classes=None, label_map=None, env="", source="", is_ood=False, uid_prefix=""):
    ds = ImageFolder(root=root)
    rows = []
    for i, (path, old_y) in enumerate(ds.samples):
        class_name = ds.classes[old_y]
        if keep_classes is not None and class_name not in keep_classes:
            continue
        label = -1 if is_ood else (label_map[class_name] if label_map else old_y)
        rows.append({"uid": f"{uid_prefix}:{i}", "path": os.path.abspath(path), "label": label, "class_name": "__ood__" if is_ood else class_name, "env": env, "source": source, "is_ood": is_ood})
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_samples(samples, seed, ratio=0.8):
    rng = random.Random(seed); s = list(samples); rng.shuffle(s); n = int(len(s) * ratio); return s[:n], s[n:]


def build_b_mix(id_train, ood_pool, rho, seed):
    if rho == 0.0: return list(id_train)
    n_ood = int(len(id_train) * rho / max(1e-8, 1 - rho))
    rng = random.Random(seed); pool = list(ood_pool); rng.shuffle(pool)
    return id_train + pool[:min(n_ood, len(pool))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_name", default="openset_domainnet_objects_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    out = os.path.join(args.data_root, args.output_name)
    if os.path.exists(out):
        if not args.rebuild: raise SystemExit(f"Output exists: {out} (use --rebuild)")
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    label_map = {k: i for i, k in enumerate(ID_CLASSES)}

    req = {"real": os.path.join(args.data_root, "domain_net", "real"), "painting": os.path.join(args.data_root, "domain_net", "painting"), "sketch": os.path.join(args.data_root, "domain_net", "sketch"), "clipart": os.path.join(args.data_root, "domain_net", "clipart")}
    for k, p in req.items():
        if not os.path.isdir(p): raise RuntimeError(f"Missing {k}: {p}")

    a_real = collect_imagefolder_samples(req["real"], ID_CLASSES, label_map, env="A_real", source="DomainNet", uid_prefix="A_real")
    a_paint = collect_imagefolder_samples(req["painting"], ID_CLASSES, label_map, env="A_painting", source="DomainNet", uid_prefix="A_painting")
    b_id = collect_imagefolder_samples(req["sketch"], ID_CLASSES, label_map, env="B_id", source="DomainNet", uid_prefix="B_id")
    t_clip = collect_imagefolder_samples(req["clipart"], ID_CLASSES, label_map, env="T_clipart", source="DomainNet", uid_prefix="T_clipart")

    terra = collect_imagefolder_samples(os.path.join(args.data_root, "terra_incognita", "location_100"), env="B_ood", source="TerraIncognita", is_ood=True, uid_prefix="Terra")
    sviro = collect_imagefolder_samples(os.path.join(args.data_root, "sviro", "aclass"), env="B_ood", source="SVIRO", is_ood=True, uid_prefix="SVIRO")
    spaw = []
    sp_root = os.path.join(args.data_root, "spawrious224")
    for split in sorted(os.listdir(sp_root)):
        sroot = os.path.join(sp_root, split)
        if not os.path.isdir(sroot): continue
        for loc in sorted(os.listdir(sroot)):
            lroot = os.path.join(sroot, loc)
            if os.path.isdir(lroot): spaw.extend(collect_imagefolder_samples(lroot, env="B_ood", source="Spawrious224", is_ood=True, uid_prefix=f"Spaw_{split}_{loc}"))

    for n, x in [("A_real", a_real), ("A_painting", a_paint), ("B_id", b_id), ("T_clipart", t_clip), ("Terra", terra), ("SVIRO", sviro), ("Spaw", spaw)]:
        print(n, len(x))
        if len(x) == 0: raise RuntimeError(f"empty source: {n}")

    b_id_train, b_id_eval = split_samples(b_id, args.seed)
    ood_train, ood_eval = split_samples(terra + sviro + spaw, args.seed + 1)
    for s in b_id_eval: s["env"] = "B_id_eval"
    for s in ood_eval: s["env"] = "B_ood_eval"
    all_samples = a_real + a_paint + b_id + t_clip + terra + sviro + spaw
    if len({s['uid'] for s in all_samples}) != len(all_samples): raise RuntimeError("UID collision")

    splits = {"A_real_train": [s["uid"] for s in a_real], "A_painting_train": [s["uid"] for s in a_paint], "B_id_eval": [s["uid"] for s in b_id_eval], "B_ood_eval": [s["uid"] for s in ood_eval], "T_clipart_eval": [s["uid"] for s in t_clip]}
    for rho in RHO_VALUES:
        mix = build_b_mix(b_id_train, ood_train, rho, args.seed + 10)
        splits[f"B_mix_train_rho{rho:.2f}"] = [s["uid"] for s in mix]

    by_uid = {s["uid"]: s for s in all_samples}
    stats = {"splits": {}}
    for n, uids in splits.items():
        rows = [by_uid[u] for u in uids]
        id_n = sum(1 for r in rows if not r["is_ood"]); ood_n = len(rows) - id_n
        stats["splits"][n] = {"n": len(rows), "class_counts": dict(Counter(str(r["label"]) for r in rows)), "env_counts": dict(Counter(r["env"] for r in rows)), "id_count": id_n, "ood_count": ood_n, "ood_ratio": (ood_n / len(rows) if len(rows) else 0.0)}

    write_jsonl(os.path.join(out, "samples.jsonl"), all_samples)
    for n, uids in splits.items():
        with open(os.path.join(out, "splits", f"{n}.json"), "w", encoding="utf-8") as f: json.dump(uids, f)
    with open(os.path.join(out, "class_map.json"), "w") as f: json.dump(label_map, f, indent=2)
    with open(os.path.join(out, "build_config.json"), "w") as f: json.dump({"id_classes": ID_CLASSES, "seed": args.seed, "rho_values": RHO_VALUES}, f, indent=2)
    with open(os.path.join(out, "stats.json"), "w") as f: json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
