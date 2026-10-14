"""Compare a fresh data-only reproduction with archived revision evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(reference, reproduced):
    checked = []
    reference_files = {p.name for p in reference.iterdir() if p.is_file()}
    reproduced_files = {p.name for p in reproduced.iterdir() if p.is_file()}
    assert reference_files == reproduced_files, (reference_files - reproduced_files, reproduced_files - reference_files)
    for name in sorted(reference_files):
        original, fresh = reference / name, reproduced / name
        if original.suffix == ".csv":
            left, right = pd.read_csv(original), pd.read_csv(fresh)
            assert list(left.columns) == list(right.columns) and left.shape == right.shape, name
            for column in left:
                if "runtime" in column:
                    continue
                if pd.api.types.is_numeric_dtype(left[column]):
                    np.testing.assert_allclose(left[column].to_numpy(), right[column].to_numpy(),
                        rtol=0, atol=1e-10, err_msg=f"{name}: {column}")
                else:
                    assert left[column].fillna("<missing>").equals(right[column].fillna("<missing>")), (name, column)
        elif original.suffix == ".json":
            assert json.loads(original.read_text(encoding="utf-8")) == json.loads(fresh.read_text(encoding="utf-8")), name
        else:
            assert original.read_bytes() == fresh.read_bytes(), name
        checked.append(name)
    print(json.dumps({"passed": True, "files_checked": len(checked),
        "absolute_numeric_tolerance": 1e-10, "excluded_columns": "runtime measurements only",
        "files": checked}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parent / "revision_results")
    parser.add_argument("--reproduced", type=Path, required=True)
    arguments = parser.parse_args()
    compare(arguments.reference.resolve(), arguments.reproduced.resolve())
