import numpy as np

from newsdim.dims import DIMS, DIM_COUNT, DimScores
from newsdim.train.trainer import LinearHead


def test_dims_constants():
    assert len(DIMS) == 8
    assert DIM_COUNT == 8
    assert DIMS == ("mom", "stab", "horz", "eng", "hype", "sent", "sec", "pol")


def test_dim_scores_to_dict():
    s = DimScores(mom=1, stab=-2, eng=3)
    d = s.to_dict()
    assert d == {"mom": 1, "stab": -2, "horz": 0, "eng": 3, "hype": 0, "sent": 0, "sec": 0, "pol": 0}


def test_dim_scores_from_dict():
    d = {"mom": 1, "stab": -2, "eng": 3, "horz": 0, "hype": 0, "sent": 0, "sec": 0, "pol": 0}
    s = DimScores.from_dict(d)
    assert s.mom == 1
    assert s.stab == -2
    assert s.eng == 3
    assert s.horz == 0


def test_dim_scores_roundtrip():
    s = DimScores(mom=-3, stab=2, horz=1, eng=0, hype=3, sent=-1, sec=-2, pol=2)
    assert DimScores.from_dict(s.to_dict()) == s


def test_dim_scores_to_array():
    s = DimScores(mom=1, stab=-2, horz=0, eng=3, hype=0, sent=-1, sec=2, pol=0)
    arr = s.to_array()
    assert arr == [1, -2, 0, 3, 0, -1, 2, 0]


def test_dim_scores_from_array():
    arr = [1, -2, 0, 3, 0, -1, 2, 0]
    s = DimScores.from_array(arr)
    assert s.mom == 1
    assert s.stab == -2
    assert s.eng == 3


def test_linear_head_predict():
    head = LinearHead(
        weight=np.zeros((768, 8), dtype=np.float32),
        bias=np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    )
    X = np.zeros((3, 768), dtype=np.float32)
    pred = head.predict(X)
    assert pred.shape == (3, 8)
    assert list(pred[0]) == [0, 0, 0, 1, 0, 0, 0, 0]


def test_linear_head_save_load(tmp_path):
    w = np.random.randn(768, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    head = LinearHead(weight=w, bias=b)
    path = tmp_path / "test.npz"
    head.save(path)
    loaded = LinearHead.load(path)
    np.testing.assert_array_equal(loaded.weight, w)
    np.testing.assert_array_equal(loaded.bias, b)
