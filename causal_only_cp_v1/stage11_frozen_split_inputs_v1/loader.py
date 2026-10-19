"""Leakage-aware loader for Stage-11 recovered inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLIT_MAP = {"train": "train", "scale_fit": "scale_fit", "tuning_calibration": "tuning_calibration",
             "validation": "validation", "final_calibration": "final_calibration", "test": "test"}


def load_split(replicate_index: int, seed: int, scenario: str, split: str, allow_test: bool = False) -> pd.DataFrame:
    if split not in SPLIT_MAP:
        raise ValueError(split)
    if split == "test" and not allow_test:
        raise PermissionError("test labels are sealed; pass allow_test=True only after CP is frozen")
    path = HERE / f"replicate_{replicate_index:02d}_seed_{seed}" / scenario / ("test_sealed.csv" if split == "test" else "development.csv")
    frame = pd.read_csv(path, keep_default_na=False)
    role = frame["split_role"].eq(split)
    result = frame.loc[role].copy()
    if split == "test":
        assert result["split"].eq("test").all()
    if len(result) == 0:
        raise ValueError(f"no rows for {split}: {path}")
    return result


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        raise SystemExit("use --self-test")
    checks = []
    for scenario in ("S2", "S4"):
        for split in ("scale_fit", "tuning_calibration", "final_calibration"):
            first = load_split(1, 20270908, scenario, split)
            second = load_split(1, 20270908, scenario, split)
            checks.append({"scenario": scenario, "split": split, "rows": len(first),
                           "hash_first": frame_hash(first), "hash_second": frame_hash(second),
                           "identical": first.equals(second)})
        first = load_split(1, 20270908, scenario, "test", allow_test=True)
        second = load_split(1, 20270908, scenario, "test", allow_test=True)
        checks.append({"scenario": scenario, "split": "test", "rows": len(first),
                       "hash_first": frame_hash(first), "hash_second": frame_hash(second),
                       "identical": first.equals(second), "sealed_access": True})
    print(json.dumps({"status": "PASS" if all(x["identical"] for x in checks) else "FAIL", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
