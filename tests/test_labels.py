import pytest
import label_store as L

CANDS = [{"key": "k1", "type": "long_pause", "track": "all", "start": 1, "end": 2, "default_action": "cut", "reason": "", "excerpt": "", "confidence": .9},
         {"key": "k2", "type": "overlap_yield", "track": "host", "start": 3, "end": 4, "default_action": "mute", "reason": "", "excerpt": "", "confidence": .6},
         {"key": "k3", "type": "false_start", "track": "all", "start": 5, "end": 6, "default_action": "review", "reason": "", "excerpt": "", "confidence": .5}]


def test_seed_approves_only_auto_types(cfg):
    labels = {}
    assert L.seed(CANDS, labels, cfg) == 3
    assert labels["k1"]["approved"] and not labels["k2"]["approved"] and not labels["k3"]["approved"]


def test_seed_keeps_existing_labels(cfg):
    labels = {"k1": {"action": "keep", "labeler": "human", "approved": True, "note": ""}}
    L.seed(CANDS, labels, cfg)
    assert labels["k1"]["action"] == "keep" and labels["k1"]["labeler"] == "human"


def test_build_fails_on_unlabeled():
    with pytest.raises(SystemExit):
        L.build(CANDS, {"k1": {"action": "cut", "labeler": "rule", "approved": True}})


def test_build_schema_and_review_never_approved(cfg):
    labels = {}; L.seed(CANDS, labels, cfg); labels["k3"]["approved"] = True
    ed = L.build(CANDS, labels)
    assert set(ed[0]) == {"id", "key", "type", "track", "start", "end", "action", "approved", "labeler", "reason", "excerpt", "confidence", "note"}
    assert ed[2]["approved"] is False


def test_laughter_edits_are_one_track_mutes(tmp_path):
    import json
    (tmp_path / "laughter_edits.json").write_text(json.dumps([
        {"track": "guest_b", "start": 12.0, "end": 13.5, "action": "mute", "kind": "laughter", "approved": True, "note": "pure laugh interval"}]))
    rows = L.laughter_local(tmp_path)
    assert rows[0]["type"] == "laughter_local" and rows[0]["action"] == "mute" and rows[0]["approved"] and rows[0]["track"] == "guest_b"
    (tmp_path / "laughter_edits.json").write_text(json.dumps([
        {"track": "guest_a", "start": 1.0, "end": 2.0, "kind": "laughter", "action": "mute"}]))
    assert L.laughter_local(tmp_path)[0]["approved"] is False
    for bad in (
        [{"track": "all", "start": 1, "end": 2, "action": "mute", "kind": "laughter"}],
        [{"track": "guest_a", "start": 1, "end": 2, "action": "cut", "kind": "laughter"}],
        [{"track": "guest_a", "start": 1, "end": 2, "action": "mute", "kind": "speech"}],
        [{"track": "guest_a", "start": 2, "end": 1, "action": "mute", "kind": "laughter"}],
        {"track": "guest_a"},
    ):
        (tmp_path / "laughter_edits.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            L.laughter_local(tmp_path)


def test_manual_edits_appended(tmp_path):
    import json
    import label_store as LS
    (tmp_path / "manual_edits.json").write_text(json.dumps([{"start": 10.0, "end": 12.5, "action": "cut", "approved": True, "note": "retake"},
                                                           {"start": 20.0, "end": 21.0, "action": "review", "approved": True}]))
    m = LS.manual(tmp_path)
    assert [x["approved"] for x in m] == [True, False] and m[0]["type"] == "manual" and m[0]["track"] == "all"
