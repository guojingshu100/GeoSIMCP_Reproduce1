"""Integration-only verification runner for cmi_adapter_v1; does not load project data."""

import numpy as np

from src.cmi_adapter_v1 import direct_score


def main() -> None:
    seed = 20260910
    rng = np.random.default_rng(seed)
    x = rng.normal(size=32)
    y = x + rng.normal(size=32)
    array = np.vstack([x, y])
    xyz = np.array([0, 1])
    first, kwargs = direct_score(array, xyz, seed)
    second, _ = direct_score(array, xyz, seed)
    if first != second:
        raise AssertionError("identical input and seed were not reproducible")
    print({"score": first, "constructor_kwargs": kwargs, "shape": list(array.shape), "xyz": xyz.tolist()})


if __name__ == "__main__":
    main()
