import json
from typing import Dict, List, TypedDict


class OpenSetSample(TypedDict):
    uid: str
    path: str
    label: int
    class_name: str
    env: str
    source: str
    is_ood: bool


def load_samples(jsonl_path: str) -> List[OpenSetSample]:
    rows: List[OpenSetSample] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_uid_split(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "uids" in data:
        return list(data["uids"])
    return list(data)


def index_samples_by_uid(samples: List[OpenSetSample]) -> Dict[str, OpenSetSample]:
    out: Dict[str, OpenSetSample] = {}
    for s in samples:
        uid = s["uid"]
        if uid in out:
            raise ValueError(f"Duplicate uid detected: {uid}")
        out[uid] = s
    return out


def select_samples(samples_by_uid: Dict[str, OpenSetSample], uid_list: List[str]) -> List[OpenSetSample]:
    missing = [u for u in uid_list if u not in samples_by_uid]
    if missing:
        raise KeyError(f"Missing uids in manifest: {missing[:5]}")
    return [samples_by_uid[u] for u in uid_list]
