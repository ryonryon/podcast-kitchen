import numpy as np
import render as R


def test_apply_cuts_length_and_timemap():
    sr = 1000; n = 10000
    x = np.arange(n, dtype=np.float64)
    out, spans, joins = R.apply_cuts({"a": x.copy(), "b": x.copy()}, [(2000, 3000, ["e1"]), (6000, 6500, ["e2"])], sr, 10)
    assert len(out["a"]) == n - 1500 and np.array_equal(out["a"], out["b"])
    tm = {"spans": spans, "joins": joins}
    assert spans == [[0.0, 2.0, 0.0], [3.0, 6.0, 2.0], [6.5, 10.0, 5.0]]
    assert R.map_time(tm, 1.0) == 1.0 and R.map_time(tm, 4.0) == 3.0 and R.map_time(tm, 2.5) == 2.0
    assert R.map_time(tm, 8.0) == 6.5 and joins[1]["out_s"] == 5.0
    # away from joins the samples are untouched
    assert out["a"][1000] == 1000 and out["a"][4000] == 5000


def test_crossfade_equal_power_constant_signal():
    x = np.ones(5000)
    out, _, _ = R.apply_cuts({"a": x}, [(2000, 3000, ["e"])], 1000, 20)
    assert out["a"].min() >= 0.99 and out["a"].max() <= 1.42


def test_laughter_local_is_not_a_candidate_key():
    assert "laughter_local" in R.EXTRA_EDIT_TYPES


def test_merge_overlapping():
    m = R.merge([(5, 8, ["b"]), (1, 3, ["a"]), (2, 4, ["c"])])
    assert [tuple(i[:2]) for i in m] == [(1, 4), (5, 8)]
