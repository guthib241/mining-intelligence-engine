import numpy as np
import pytest

from mie.data import DATASETS, load, manifest, validate_all
from mie.infrastructure import DataFailure


def test_manifests_have_required_fields():
    for k in DATASETS:
        m = manifest(k)
        for f in ("source", "schema", "validation", "limitations", "authority"):
            assert f in m, (k, f)
        assert any(x in m for x in ("sha256", "sha256_archive"))
def test_unknown_dataset_is_data_failure():
    with pytest.raises(DataFailure):
        load("NOPE")
@pytest.mark.slow
def test_headers_epoch_structure():
    z = load("BTCHDR-2025-12-14")
    D = z["difficulty"]
    assert (np.diff(D[:2016 * 300:2016]) != 0).mean() > 0.9  # difficulty changes at epoch boundaries
    v = validate_all()
    assert v["BTCHDR-2025-12-14"]["status"] == "OK"
