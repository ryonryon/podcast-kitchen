import numpy as np
import pytest
import ending_bgm as B

SR = 8000
P = {**B.DEF, "lead_in_s": 3.0, "fade_in_s": 2.0, "bed_db": -20.0, "ramp_s": 1.0, "tail_db": -4.0, "tail_s": 5.0, "fade_out_s": 3.0,
     "fade_out_floor_db": -60.0, "end_silence_s": 1.5}


def test_find_phrase_last_occurrence_in_window_uses_token_start():
    segs = [(10.0, "ということで始めます", [(10.0, "という"), (10.4, "ことで")]),
            (90.0, "はいということで今日は", [(90.0, "はい"), (90.5, "という"), (90.9, "ことで")]),
            (95.0, "ありがとうございました", [])]
    t, ph, text = B.find_phrase(segs, ["ということで"], 100.0, 30.0)
    assert t == pytest.approx(90.5) and ph == "ということで"
    assert B.find_phrase(segs, ["ということで"], 200.0, 30.0) is None       # outside the window -> skip
    assert B.find_phrase(segs, ["それでは"], 100.0, 30.0) is None


def test_last_speech_end_ignores_short_islands():
    assert B.last_speech_end([(1, 5), (6, 9.5), (9.8, 9.9)], 12.0, 0.2) == 9.5


def tone(sec, f=220, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return amp * np.sin(2 * np.pi * f * t)


def db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def test_mix_bgm_envelope_levels_and_extension():
    voice = np.concatenate([tone(20, 300), np.zeros(int(0.5 * SR))])       # speech ends at 20 s, file 20.5 s
    bgm = tone(60, 500, 0.5)
    vdb = B.active_rms_db(voice[:20 * SR], SR)
    y, info = B.mix_bgm(voice, bgm, SR, 12.0, 20.0, P, vdb)
    assert info["fade_in_start_s"] == pytest.approx(9.0)
    assert info["fade_out_end_s"] == pytest.approx(28.0, abs=0.01)
    assert info["end_s"] == pytest.approx(29.5, abs=0.01) and info["extended_s"] == pytest.approx(9.0, abs=0.01)
    added = y.copy(); added[:len(voice)] -= voice
    seg = lambda a, b: added[int(a * SR):int(b * SR)]
    assert np.max(np.abs(seg(0, 8.99))) == 0                                  # nothing before the lead-in
    assert db(seg(12, 19)) - vdb == pytest.approx(-20, abs=0.5)               # ducked bed under the voices
    assert db(seg(21.5, 24.5)) - vdb == pytest.approx(-4, abs=0.5)            # BGM-only level after the last speech
    assert db(seg(9.0, 9.5)) < db(seg(10.5, 11.0))                            # fade-in
    assert np.all(y[-int(1.5 * SR):] == 0)                                    # end_silence_s of digital silence


def make_ep(tmp_path, text="ということで以上です", bgm_seconds=40, **bgm_cfg):
    import json, sys, yaml, soundfile as sf
    from conftest import ROOT, seg
    ep = tmp_path / "ep"; (ep / "work").mkdir(parents=True)
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    sf.write(str(tmp_path / "bgm.wav"), np.stack([tone(bgm_seconds, 500, 0.5)] * 2, 1), SR)
    cfg["ending_bgm"].update({"file": str(tmp_path / "bgm.wav"), "mode": "phrase_tail", **bgm_cfg})
    (ep / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    (ep / "manifest.json").write_text(json.dumps({"speakers": ["a"]}))
    v = np.concatenate([tone(20, 300), np.zeros(SR // 2)])
    sf.write(str(ep / "work/mix.wav"), np.stack([v, v], 1), SR, subtype="FLOAT")
    (ep / "work/timemap.json").write_text(json.dumps({"spans": [[0, 20.5, 0]], "edited_duration_s": 20.5}))
    (ep / "work/words.json").write_text(json.dumps({"a": [seg(14.0, 16.0, text)]}, ensure_ascii=False))
    (ep / "work/vad.json").write_text(json.dumps({"speakers": {"a": [[0.5, 20.0]]}}))
    return ep, len(v)


def run(ep, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["ending_bgm.py", "--ep", str(ep)])
    B.main()


def frames(p):
    import soundfile as sf
    return sf.info(str(p)).frames


def test_main_applies_extends_and_is_idempotent(tmp_path, monkeypatch):
    import json
    ep, n = make_ep(tmp_path)
    run(ep, monkeypatch)
    log = json.loads((ep / "logs/ending_bgm.json").read_text())
    assert log["applied"] and log["phrase_s"] == pytest.approx(14.0) and log["fade_in_start_s"] == pytest.approx(11.0)
    n1 = frames(ep / "work/mix.wav"); assert n1 > n
    run(ep, monkeypatch)                                   # re-run: rebuilt from mix_voice.wav, not stacked
    assert frames(ep / "work/mix.wav") == n1 and frames(ep / "work/mix_voice.wav") == n
    assert json.loads((ep / "work/timemap.json").read_text())["edited_duration_s"] == pytest.approx(n1 / SR)


@pytest.mark.parametrize("kw,reason", [({"enabled": False}, "disabled"), ({"file": "/nonexistent/x.mp3"}, "not found"),
                                       ({"phrases": ["それでは"]}, "no closing phrase")])
def test_main_skips_cleanly(tmp_path, monkeypatch, capsys, kw, reason):
    import json
    ep, n = make_ep(tmp_path, **kw)
    run(ep, monkeypatch)
    log = json.loads((ep / "logs/ending_bgm.json").read_text())
    assert not log["applied"] and reason in log["reason"] and "WARNING" in capsys.readouterr().err
    assert frames(ep / "work/mix.wav") == n                 # voice mix passed through unchanged


def test_qa_bed_click_score_separates_music_from_click():
    from qa import click_score, click_score_bed
    rs = np.random.default_rng(1); t = np.arange(SR * 2) / SR
    music = 0.2 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * t))) + 0.01 * rs.standard_normal(len(t))
    i = SR
    assert click_score_bed(music, i, SR) < 2.5
    clicked = music.copy(); clicked[i:] += 0.3
    assert click_score_bed(clicked, i, SR) > 2.5


def test_fade_out_is_db_linear_monotonic_and_reaches_floor():
    p = {**P, "fade_out_s": 7.0, "end_silence_s": 1.5}
    g, n = B.envelope(10 * SR, SR, 1.0, 10.0, p)               # tail hold 10..15 s, fade 15..22 s, silence 22..23.5 s
    assert n == int(np.ceil(23.5 * SR))
    fade = g[15 * SR:22 * SR]
    assert np.all(np.diff(fade) <= 1e-12)                      # monotonic non-increasing
    gdb = 20 * np.log10(fade)
    assert gdb[0] == pytest.approx(-4.0, abs=0.01)
    assert gdb[-1] <= -60.0 + 0.01                             # reaches the floor at the fade end
    mid = 20 * np.log10(g[int(18.5 * SR)])
    assert mid == pytest.approx(-32.0, abs=0.05)               # linear in dB: halfway = midpoint dB
    assert np.all(g[22 * SR:] == 0) and len(g[22 * SR:]) == int(1.5 * SR)


def _full(**kw):
    p = {**B.DEF, "mode": "full_asset", "fade_in_s": 0.0, "ramp_s": 0.0, "tail_fade_ms": 0.0,
         "end_silence_s": 0.0, "bed_db": -20.0, "tail_db": -4.0, "tail_s": 0.3, "fade_out_s": 0.2}
    p.update(kw)
    return p


def test_full_asset_end_anchors_are_speech_or_episode_not_a_cropped_tail():
    voice = np.concatenate([tone(5, 300), np.zeros(2 * SR)])          # speech ends at 5; episode file ends at 7
    music = tone(2, 500, 0.5)
    vdb = B.active_rms_db(voice[:5 * SR], SR)
    y_ep, info_ep = B.mix_full_asset(voice, music, None, SR, 5.0, 7.0, 5.0, _full(end_anchor="episode_end"), vdb)
    y_sp, info_sp = B.mix_full_asset(voice, music, None, SR, 5.0, 7.0, 5.0, _full(end_anchor="speech_end"), vdb)
    assert info_ep["end_anchor_met"] and info_ep["music_end_s"] == pytest.approx(7.0)
    assert info_ep["music_samples"] == 2 * SR and info_ep["cropped"] is False and info_ep["looped"] is False
    assert info_sp["end_anchor_met"] and info_sp["music_end_s"] == pytest.approx(5.0)
    assert info_sp["music_start_s"] == pytest.approx(3.0)
    # tail_s is 0.3s; the payload is still the whole 2s either way
    assert info_ep["music_end_s"] - info_ep["music_start_s"] == pytest.approx(2.0)
    assert info_sp["music_payload_s"] == pytest.approx(2.0)
    base = np.zeros_like(y_ep); base[:len(voice)] = voice
    added = y_ep - base
    assert np.max(np.abs(added[:int(info_ep["music_start_s"] * SR)])) == 0
    assert np.count_nonzero(np.abs(added) > 1e-8) <= 2 * SR + 2


def test_full_asset_keeps_overlong_payload_and_natural_tail_level():
    voice = tone(4, 300)
    music = tone(10, 500, 0.5)
    vdb = B.active_rms_db(voice, SR)
    y, info = B.mix_full_asset(voice, music, None, SR, 4.0, 4.0, 4.0, _full(end_anchor="episode_end"), vdb)
    assert info["end_anchor_met"] is False and info["anchor_shift_s"] == pytest.approx(6.0)
    assert info["music_samples"] == 10 * SR and info["music_start_s"] == pytest.approx(0.0)
    assert info["music_end_s"] == pytest.approx(10.0) and info["cropped"] is False and info["looped"] is False
    assert len(y) == 10 * SR
    base = np.zeros_like(y); base[:len(voice)] = voice
    tail = (y - base)[int(8 * SR):int(9 * SR)]
    assert db(tail) - vdb == pytest.approx(-4.0, abs=0.6)               # natural tail held, not faded to the floor
    assert np.max(np.abs((y - base)[int(9.0 * SR):int(9.8 * SR)])) > 1e-3


def test_full_asset_aligns_outro_end_to_music_end_without_looping():
    voice = np.concatenate([tone(8, 300), np.zeros(2 * SR)])
    music, outro = tone(4, 500, 0.5), tone(2, 700, 0.4)
    vdb = B.active_rms_db(voice[:8 * SR], SR)
    y, info = B.mix_full_asset(voice, music, outro, SR, 8.0, 10.0, 8.0, _full(end_anchor="episode_end"), vdb)
    assert info["ends_aligned"] and info["music_end_s"] == pytest.approx(10.0) and info["outro_end_s"] == pytest.approx(10.0)
    assert info["music_samples"] == 4 * SR and info["outro_samples"] == 2 * SR
    assert info["music_start_s"] == pytest.approx(6.0) and info["outro_start_s"] == pytest.approx(8.0)
    assert info["looped"] is False and info["cropped"] is False
    base = np.zeros_like(y); base[:len(voice)] = voice
    added = y - base
    assert np.max(np.abs(added[:int(6 * SR)])) == 0
    assert np.max(np.abs(added[int(6 * SR):int(8 * SR)])) > 0


def test_explicit_starts_keep_both_payloads_when_ends_differ():
    voice = np.zeros(8 * SR)
    music, outro = tone(3, 500, 0.5), tone(1, 700, 0.4)
    y, info = B.mix_full_asset(voice, music, outro, SR, 6.0, 8.0, 6.0,
                              _full(music_start_s=1.0, outro_start_s=1.0), 0.0)
    assert info["ends_aligned"] is False and info["end_anchor_met"] is False
    assert info["music_samples"] == 3 * SR and info["outro_samples"] == 1 * SR
    assert info["music_end_s"] == pytest.approx(4.0) and info["outro_end_s"] == pytest.approx(2.0)
    assert len(y) >= 4 * SR


def test_music_start_end_aligns_outro():
    voice = np.zeros(8 * SR)
    _, info = B.mix_full_asset(voice, tone(3, 500, 0.5), tone(1, 700, 0.4), SR, 6.0, 8.0, 6.0,
                              _full(music_start_s=2.0), 0.0)
    assert info["ends_aligned"] and info["music_end_s"] == pytest.approx(5.0) and info["outro_end_s"] == pytest.approx(5.0)
    assert info["outro_start_s"] == pytest.approx(4.0) and info["outro_samples"] == SR


def test_main_full_asset_ignores_missing_phrase_and_is_idempotent(tmp_path, monkeypatch):
    import json
    ep, n = make_ep(tmp_path, text="ありがとうございました", bgm_seconds=5, mode="full_asset", end_anchor="episode_end",
                    fade_in_s=0.0, ramp_s=0.0, tail_fade_ms=0.0)
    run(ep, monkeypatch)
    log = json.loads((ep / "logs/ending_bgm.json").read_text())
    assert log["applied"] and log["mode"] == "full_asset" and log["cropped"] is False and log["looped"] is False
    assert log["end_anchor"] == "episode_end" and log["music_end_s"] == pytest.approx(20.5)
    assert log["speech_end_s"] == pytest.approx(20.0)          # final speech, earlier than the episode end
    assert log["episode_end_in_s"] == pytest.approx(20.5)
    assert log["music_samples"] == 5 * SR
    n1 = frames(ep / "work/mix.wav")
    assert n1 == pytest.approx(n + int(round(1.5 * SR)), abs=2)
    run(ep, monkeypatch)
    assert frames(ep / "work/mix.wav") == n1


def test_main_full_asset_missing_outro_skips(tmp_path, monkeypatch, capsys):
    import json
    ep, n = make_ep(tmp_path, mode="full_asset", outro_file="/nonexistent/outro.wav")
    run(ep, monkeypatch)
    log = json.loads((ep / "logs/ending_bgm.json").read_text())
    assert not log["applied"] and "outro" in log["reason"] and frames(ep / "work/mix.wav") == n
    assert "WARNING" in capsys.readouterr().err
