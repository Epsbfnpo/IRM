#!/usr/bin/env python3
import argparse, json, os, random
from collections import Counter
from datasets import collect_imagefolder_samples, wrap_samples

RHO_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9]
ID_CLASSES = ["book", "clock", "keyboard", "lamp", "mug", "scissors"]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_split(path, uid_list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(uid_list, f)


def _split(samples, seed, train_ratio=0.8):
    rng = random.Random(seed)
    s = list(samples)
    rng.shuffle(s)
    n = int(len(s) * train_ratio)
    return s[:n], s[n:]


def build_b_mix(id_train, ood_pool, rho, seed):
    if rho == 0.0:
        return list(id_train)
    n_id = len(id_train)
    n_ood = int(n_id * rho / max(1e-8, 1 - rho))
    rng = random.Random(seed)
    pool = list(ood_pool)
    rng.shuffle(pool)
    return id_train + pool[:min(len(pool), n_ood)]


def build_stats(samples, splits):
    by_uid = {s["uid"]: s for s in samples}
    out = {"env_counts": {}, "env_class_counts": {}, "splits": {}}
    for name, uids in splits.items():
        env_cnt = Counter(by_uid[u]["env"] for u in uids)
        cls_cnt = Counter(str(by_uid[u]["label"]) for u in uids)
        out["splits"][name] = {"n": len(uids), "env_counts": dict(env_cnt), "class_counts": dict(cls_cnt)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_name", default="openset_domainnet_objects_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    out = os.path.join(args.data_root, args.output_name)
    splits_dir = os.path.join(out, "splits")
    if os.path.exists(out) and not args.rebuild:
        raise SystemExit(f"Output exists: {out}. Use --rebuild")
    os.makedirs(splits_dir, exist_ok=True)

    label_map = {k: i for i, k in enumerate(ID_CLASSES)}
    dn = os.path.join(args.data_root, "domain_net")
    raw_real = collect_imagefolder_samples(os.path.join(dn, "real"), ID_CLASSES, label_map)
    raw_paint = collect_imagefolder_samples(os.path.join(dn, "painting"), ID_CLASSES, label_map)
    raw_sketch = collect_imagefolder_samples(os.path.join(dn, "sketch"), ID_CLASSES, label_map)
    raw_clipart = collect_imagefolder_samples(os.path.join(dn, "clipart"), ID_CLASSES, label_map)

    a_real = wrap_samples(raw_real, "A_real", "DomainNet", False, "A_real")
    a_paint = wrap_samples(raw_paint, "A_painting", "DomainNet", False, "A_painting")
    b_id = wrap_samples(raw_sketch, "B_id", "DomainNet", False, "B_id")
    t_clip = wrap_samples(raw_clipart, "T_clipart", "DomainNet", False, "T_clipart")

    terra = wrap_samples([(p, -1) for p, _ in collect_imagefolder_samples(os.path.join(args.data_root, "terra_incognita", "location_100"))], "B_ood", "TerraIncognita", True, "Terra")
    sviro = wrap_samples([(p, -1) for p, _ in collect_imagefolder_samples(os.path.join(args.data_root, "sviro", "aclass"))], "B_ood", "SVIRO", True, "SVIRO")
    spaw = []
    sp_root = os.path.join(args.data_root, "spawrious224")
    for split in sorted(os.listdir(sp_root)):
        for loc in sorted(os.listdir(os.path.join(sp_root, split))):
            loc_root = os.path.join(sp_root, split, loc)
            if os.path.isdir(loc_root):
                spaw.extend(collect_imagefolder_samples(loc_root))
    spaw = wrap_samples([(p, -1) for p, _ in spaw], "B_ood", "Spawrious224", True, "Spaw")

    for n, xs in [("Terra", terra), ("SVIRO", sviro), ("Spaw", spaw)]:
        print(n, len(xs))
        if len(xs) == 0:
            raise RuntimeError(f"{n} source is empty")

    b_id_train, b_id_eval = _split(b_id, args.seed)
    ood_train, ood_eval = _split(terra + sviro + spaw, args.seed + 1)
    for s in b_id_eval: s["env"] = "B_id_eval"
    for s in ood_eval: s["env"] = "B_ood_eval"

    all_samples = a_real + a_paint + b_id + t_clip + terra + sviro + spaw
    uids = [s["uid"] for s in all_samples]
    if len(set(uids)) != len(uids):
        raise RuntimeError("UID must be unique")

    splits = {
        "A_real_train": [s["uid"] for s in a_real],
        "A_painting_train": [s["uid"] for s in a_paint],
        "B_id_eval": [s["uid"] for s in b_id_eval],
        "B_ood_eval": [s["uid"] for s in ood_eval],
        "T_clipart_eval": [s["uid"] for s in t_clip],
    }
    for rho in RHO_VALUES:
        mix = build_b_mix(b_id_train, ood_train, rho, args.seed + 10)
        splits[f"B_mix_train_rho{rho:.2f}"] = [s["uid"] for s in mix]

    write_jsonl(os.path.join(out, "samples.jsonl"), all_samples)
    for k, v in splits.items():
        write_split(os.path.join(splits_dir, f"{k}.json"), v)
    with open(os.path.join(out, "class_map.json"), "w") as f: json.dump(label_map, f, indent=2)
    with open(os.path.join(out, "build_config.json"), "w") as f: json.dump({"id_classes": ID_CLASSES, "seed": args.seed, "rho_values": RHO_VALUES}, f, indent=2)
    with open(os.path.join(out, "stats.json"), "w") as f: json.dump(build_stats(all_samples, splits), f, indent=2)

if __name__ == "__main__":
    main()
