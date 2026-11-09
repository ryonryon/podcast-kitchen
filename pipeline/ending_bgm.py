"""Step 5a: ending BGM (render stage, after mix, before loudness). Not the mid-show jingle.

mode phrase_tail (legacy, also the default when an episode config has no mode key): last closing phrase
(ending_bgm.phrases) within search_window_s of the voice-mix end, fade in lead_in_s before it, bed_db under the
voices, ramp to tail_db after the last speech end, hold tail_s, fade_out_s down to fade_out_floor_db, then
end_silence_s. A longer music file is not played past that envelope.

mode full_asset: one play of the music payload (from bgm_offset_s through EOF) and, when outro_file is set, one
play of that second payload. No loop. tail_s and fade_out_s do not crop either file. The shared end aims at
end_anchor (speech_end = last qualifying VAD end; episode_end = voice-mix duration before this step extends it;
explicit = end_anchor_s). If a full payload would start before start_floor_s, both assets shift later together.
Under speech the bed stays at bed_db; after gain_speech_end_s (default: that VAD end) it rises to tail_db and
holds through the last payload sample when tail_fade_ms is 0. end_silence_s is digital silence after the content,
not a replacement for the music tail.

Rewrites work/mix.wav (voice-only mix kept as work/mix_voice.wav). Skips with a warning when disabled, a file is
missing, or full_asset placement is invalid. Log: logs/ending_bgm.json; timemap.json gets ending_bgm."""
import argparse, os, subprocess, sys
from pathlib import Path
import numpy as np
from common import dump, load, load_cfg, log_step, r, read_wav, speakers

DEF = {"enabled": True, "mode": "phrase_tail", "file": "../../assets/ending_bgm.mp3", "phrases": ["ということで"],
       "search_window_s": 120.0, "lead_in_s": 3.0, "fade_in_s": 2.0, "bed_db": -20.0, "ramp_s": 1.5, "tail_db": -4.0,
       "tail_s": 5.0, "fade_out_s": 7.0, "fade_out_floor_db": -60.0, "end_silence_s": 1.5, "bgm_offset_s": 0.0,
       "voice_ref_window_s": 60.0, "min_island_s": 0.2, "end_anchor": "episode_end", "end_anchor_s": None,
       "music_start_s": None, "outro_file": "", "outro_offset_s": 0.0, "outro_start_s": None, "outro_bed_db": None,
       "outro_tail_db": None, "start_floor_s": 0.0, "gain_speech_end_s": None, "tail_fade_ms": 0.0}


def map_time(tm, t):
    for a, b, o in tm["spans"]:
        if t < a:
            return o
        if t <= b:
            return o + (t - a)
    a, b, o = tm["spans"][-1]
    return o + (b - a)


def find_phrase(segs, phrases, dur, window):
    """segs: [(edited_start, text, [(edited_tok_start or None, surface), ...])]. -> (time, phrase, text) of the last
    occurrence starting within `window` s of the end, else None. The start is the token where the phrase begins
    (last known token start at or before it; the segment start if tokens carry no timing)."""
    best = None
    for st, text, toks in segs:
        for ph in phrases:
            i = text.rfind(ph)
            if i < 0:
                continue
            t, n = st, 0
            for ts, sf in toks:
                if ts is not None and n <= i:
                    t = ts
                n += len(sf)
                if n > i:
                    break
            if t >= dur - window and (best is None or t > best[0]):
                best = (t, ph, text)
    return best


def last_speech_end(intervals, dur, min_len=0.2):
    ends = [b for a, b in intervals if b - a >= min_len and a < dur]
    return min(dur, max(ends)) if ends else None


def active_rms_db(x, sr):
    fr = int(0.05 * sr); m = len(x) // fr
    if m == 0:
        return -120.0
    e = np.mean(x[: m * fr].reshape(m, fr) ** 2, axis=1); db = 10 * np.log10(e + 1e-20)
    act = e[db > np.max(db) - 30]
    return float(10 * np.log10(np.mean(act) + 1e-20))


def envelope(n, sr, t_in, t_end, p):
    """-> (amplitude gain per sample relative to the voice-matched BGM, total length). Silence before t_in, linear
    fade-in to bed_db, bed until t_end, ramp to tail_db over ramp_s, hold until t_end + tail_s, fade-out linear in dB
    from tail_db to fade_out_floor_db over fade_out_s, then 0 (incl. end_silence_s of digital silence)."""
    t_ph_end = t_end + p["tail_s"]; t_stop = t_ph_end + p["fade_out_s"]
    n_total = max(n, int(np.ceil((t_stop + p["end_silence_s"]) * sr)))
    t = np.arange(n_total) / sr
    bed, tail = p["bed_db"], p["tail_db"]
    lin = lambda a, b, ga, gb: ga + (gb - ga) * np.clip((t - a) / max(b - a, 1e-9), 0, 1)
    fi = (t >= t_in) & (t < t_in + p["fade_in_s"])     # fade in from silence (in amplitude)
    g_amp = np.zeros(n_total)
    g_amp[fi] = 10 ** (bed / 20) * (t[fi] - t_in) / p["fade_in_s"]
    on = (t >= t_in + p["fade_in_s"]) & (t < t_end)
    g_amp[on] = 10 ** (bed / 20)
    up = (t >= t_end) & (t < t_ph_end)
    g_amp[up] = 10 ** (lin(t_end, t_end + p["ramp_s"], bed, tail)[up] / 20)
    fo = (t >= t_ph_end) & (t < t_stop)
    g_amp[fo] = 10 ** ((tail + (p["fade_out_floor_db"] - tail) * (t[fo] - t_ph_end) / p["fade_out_s"]) / 20)
    return g_amp, n_total


def mix_bgm(voice, bgm, sr, t_phrase, t_end, p, voice_db):
    """voice: mono mix; bgm: mono BGM; levels: bed/tail are dB relative to voice_db (active voice RMS)."""
    t_in = max(0.0, t_phrase - p["lead_in_s"])
    t_end = max(t_end, t_in + p["fade_in_s"])
    g, n_total = envelope(len(voice), sr, t_in, t_end, p)
    b = bgm[int(p["bgm_offset_s"] * sr):]
    norm = 10 ** ((voice_db - active_rms_db(b[: int(30 * sr)], sr)) / 20)  # BGM active level -> voice level
    i0 = int(t_in * sr); seg = np.zeros(n_total); k = min(len(b), n_total - i0)
    seg[i0:i0 + k] = b[:k] * norm
    out = np.zeros(n_total); out[:len(voice)] = voice
    out += seg * g
    return out, {"fade_in_start_s": r(t_in), "last_speech_end_s": r(t_end), "tail_end_s": r(t_end + p["tail_s"]),
                 "fade_out_end_s": r(t_end + p["tail_s"] + p["fade_out_s"]), "end_s": r(n_total / sr), "extended_s": r((n_total - len(voice)) / sr), "bgm_short": bool(k < n_total - i0)}


def _payload(x, sr, offset_s):
    """Samples from offset through EOF. No loop and no further crop."""
    if x is None:
        return None
    i = int(round(max(0.0, float(offset_s or 0.0)) * sr))
    if i >= len(x):
        return np.zeros(0, dtype=np.float64)
    return np.asarray(x[i:], dtype=np.float64)


def place_full_payloads(music_n, outro_n, sr, speech_end, episode_end, p):
    """Place full music (and optional outro) payloads. Returns (plan, error).

    plan times are seconds. shift > 0 means the requested end could not be met without starting before
    start_floor_s; both assets move later by the same amount and nothing is cropped or looped.
    speech_end is the last qualifying speech end (final speech). episode_end is the voice-mix duration
    before this step extends the file (current episode end, which can be later than the last word)."""
    floor = float(p.get("start_floor_s") or 0.0)
    anchor = p.get("end_anchor") or "episode_end"
    ms_cfg, os_cfg = p.get("music_start_s"), p.get("outro_start_s")
    music_dur = music_n / sr
    has_outro = bool(outro_n)
    outro_dur = (outro_n / sr) if has_outro else 0.0

    def shift_pair(ms, os_):
        earliest = ms if os_ is None else min(ms, os_)
        shift = max(0.0, floor - earliest)
        return ms + shift, (None if os_ is None else os_ + shift), shift

    if ms_cfg is not None and (not has_outro or os_cfg is not None):
        ms, os_ = float(ms_cfg), (float(os_cfg) if has_outro else None)
        if ms < floor or (os_ is not None and os_ < floor):
            return None, "explicit start is before start_floor_s"
        me = ms + music_dur
        oe = None if os_ is None else os_ + outro_dur
        aligned = True if oe is None else abs(me - oe) <= max(1.0 / sr, 1e-3)
        return {"music_start": ms, "outro_start": os_, "aligned": aligned, "shift": 0.0,
                "requested_end": None, "how": "explicit_starts"}, None
    if ms_cfg is not None and has_outro:
        ms0 = float(ms_cfg)
        if ms0 < floor:
            return None, "explicit start is before start_floor_s"
        me = ms0 + music_dur
        ms, os_, shift = shift_pair(ms0, me - outro_dur)
        return {"music_start": ms, "outro_start": os_, "aligned": True, "shift": shift,
                "requested_end": me, "how": "music_start"}, None
    if os_cfg is not None and has_outro and ms_cfg is None:
        os0 = float(os_cfg)
        if os0 < floor:
            return None, "explicit start is before start_floor_s"
        oe = os0 + outro_dur
        ms, os_, shift = shift_pair(oe - music_dur, os0)
        return {"music_start": ms, "outro_start": os_, "aligned": True, "shift": shift,
                "requested_end": oe, "how": "outro_start"}, None
    if anchor == "speech_end":
        requested = float(speech_end)
    elif anchor == "episode_end":
        requested = float(episode_end)
    elif anchor == "explicit":
        if p.get("end_anchor_s") is None:
            return None, "end_anchor explicit requires end_anchor_s"
        requested = float(p["end_anchor_s"])
    else:
        return None, f"unknown end_anchor {anchor}"
    ms, os_, shift = shift_pair(requested - music_dur, (requested - outro_dur) if has_outro else None)
    return {"music_start": ms, "outro_start": os_, "aligned": True, "shift": shift,
            "requested_end": requested, "how": anchor}, None


def payload_gain(n, sr, t0, gain_at, fade_s, ramp_s, bed_db, tail_db, tail_fade_ms):
    """Per-sample gain for one full payload. Does not change n. tail_fade_ms 0 leaves the last samples at the hold level."""
    if n <= 0:
        return np.zeros(0)
    t = t0 + np.arange(n) / sr
    level_db = np.full(n, bed_db if gain_at > t0 else tail_db)
    if gain_at > t0:
        after = t >= gain_at
        if ramp_s <= 1e-9:
            level_db[after] = tail_db
        else:
            u = np.clip((t - gain_at) / ramp_s, 0, 1)
            level_db = bed_db + (tail_db - bed_db) * u
            level_db[t < gain_at] = bed_db
    amp = 10 ** (level_db / 20.0)
    if fade_s > 1e-9:
        amp = amp * np.clip((np.arange(n) / sr) / fade_s, 0, 1)
    fade_n = int(round(float(tail_fade_ms or 0.0) / 1000.0 * sr))
    if fade_n > 0:
        fade_n = min(fade_n, n)
        amp[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, endpoint=True)
    return amp


def mix_full_asset(voice, music, outro, sr, speech_end, episode_end, gain_at, p, voice_db, voice_t0=0.0):
    """Mix full music and optional outro payloads onto `voice` (timeline starts at voice_t0).

    speech_end is the anchor value for end_anchor speech_end (final speech). episode_end is the voice-mix
    duration before extension (current episode end). gain_at is only where the bed rises toward tail_db.
    Returns (audio, info) or (None, error)."""
    mp = _payload(music, sr, p.get("bgm_offset_s") or 0.0)
    op = _payload(outro, sr, p.get("outro_offset_s") or 0.0) if outro is not None else None
    if len(mp) == 0:
        return None, "music payload empty"
    if outro is not None and (op is None or len(op) == 0):
        return None, "outro payload empty"
    placed, err = place_full_payloads(len(mp), 0 if op is None else len(op), sr, speech_end, episode_end, p)
    if err:
        return None, err

    def norm_of(pay):
        ref = pay[:int(30 * sr)] if len(pay) else pay
        return 10 ** ((voice_db - active_rms_db(ref, sr)) / 20.0)

    music_i = int(round((placed["music_start"] - voice_t0) * sr))
    if op is not None and placed["aligned"]:
        outro_i = music_i + len(mp) - len(op)
    elif op is not None:
        outro_i = int(round((placed["outro_start"] - voice_t0) * sr))
    else:
        outro_i = None
    if music_i < 0 or (outro_i is not None and outro_i < 0):
        return None, "placement starts before the voice slice"
    silence_n = int(round(float(p.get("end_silence_s") or 0.0) * sr))
    content_end = max(len(voice), music_i + len(mp), (outro_i + len(op)) if outro_i is not None else 0)
    n_total = content_end + silence_n
    out = np.zeros(n_total)
    out[:min(len(voice), n_total)] = voice[:min(len(voice), n_total)]

    def add(pay, i0, bed, tail):
        g = payload_gain(len(pay), sr, voice_t0 + i0 / sr, gain_at, float(p.get("fade_in_s") or 0.0),
                         float(p.get("ramp_s") or 0.0), bed, tail, float(p.get("tail_fade_ms") or 0.0))
        out[i0:i0 + len(pay)] += pay * norm_of(pay) * g

    bed, tail = float(p["bed_db"]), float(p["tail_db"])
    add(mp, music_i, bed, tail)
    if op is not None:
        ob = bed if p.get("outro_bed_db") is None else float(p["outro_bed_db"])
        ot = tail if p.get("outro_tail_db") is None else float(p["outro_tail_db"])
        add(op, outro_i, ob, ot)
    music_end = voice_t0 + (music_i + len(mp)) / sr
    outro_end = None if outro_i is None else voice_t0 + (outro_i + len(op)) / sr
    asset_end = music_end if outro_end is None else max(music_end, outro_end)
    info = {"mode": "full_asset", "applied": True, "end_anchor": p.get("end_anchor") or "episode_end",
            "end_anchor_met": bool(placed["shift"] <= 1e-6 and placed["aligned"]), "anchor_shift_s": r(placed["shift"], 4),
            "requested_end_s": None if placed["requested_end"] is None else r(placed["requested_end"]),
            "how": placed["how"], "speech_end_s": r(speech_end), "gain_speech_end_s": r(gain_at),
            "episode_end_in_s": r(episode_end), "music_start_s": r(voice_t0 + music_i / sr), "music_end_s": r(music_end),
            "music_payload_s": r(len(mp) / sr), "music_samples": int(len(mp)),
            "outro_start_s": None if outro_end is None else r(voice_t0 + outro_i / sr),
            "outro_end_s": None if outro_end is None else r(outro_end),
            "outro_payload_s": 0.0 if op is None else r(len(op) / sr), "outro_samples": 0 if op is None else int(len(op)),
            "ends_aligned": bool(placed["aligned"] if op is not None else True), "cropped": False, "looped": False,
            "shared_end_s": r(asset_end), "fade_in_start_s": r(voice_t0 + min(music_i, outro_i if outro_i is not None else music_i) / sr),
            "last_speech_end_s": r(gain_at), "tail_end_s": r(voice_t0 + content_end / sr), "fade_out_end_s": r(asset_end),
            "end_s": r(voice_t0 + n_total / sr), "extended_s": r((n_total - len(voice)) / sr), "bgm_short": False}
    return out, info


def decode(path, sr):
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f64le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float64).copy()


def main():
    import soundfile as sf
    ap = argparse.ArgumentParser(); ap.add_argument("--ep", required=True); a = ap.parse_args()
    ep = Path(a.ep); cfg = load_cfg(ep); p = {**DEF, **(cfg.get("ending_bgm") or {})}
    mix_p, voice_p, mark = ep / "work/mix.wav", ep / "work/mix_voice.wav", ep / "work/.ending_bgm"
    # mix.wav written by this step before (mark = its mtime) -> the voice mix is mix_voice.wav; else render just wrote it
    ours = mark.exists() and mix_p.exists() and mark.read_text().strip() == str(mix_p.stat().st_mtime_ns)
    if not ours:
        if not mix_p.exists():
            raise SystemExit("ending_bgm: work/mix.wav missing; run render first")
        os.replace(mix_p, voice_p)
    info_v = sf.info(str(voice_p)); sr, n, ch = info_v.samplerate, info_v.frames, info_v.channels; dur = n / sr

    def write(tail=None, t0=0):
        """copy the voice mix head blockwise, then the processed mono tail from sample t0 (or the rest unchanged)."""
        blk = sr * 60
        with sf.SoundFile(str(voice_p)) as src, sf.SoundFile(str(mix_p), "w", sr, ch, subtype="FLOAT") as dst:
            stop = t0 if tail is not None else n
            for s0 in range(0, stop, blk):
                dst.write(src.read(min(blk, stop - s0), dtype="float64", always_2d=True))
            if tail is not None:
                dst.write(np.repeat(tail[:, None], ch, 1))
        mark.write_text(str(mix_p.stat().st_mtime_ns))

    def skip(msg):
        print(f"WARNING ending_bgm: {msg}; skipped", file=sys.stderr)
        write(); dump(ep / "logs/ending_bgm.json", {"applied": False, "reason": msg}); log_step(ep, "ending_bgm", {"applied": False, "reason": msg})

    if not p["enabled"]:
        return skip("disabled")
    mode = p.get("mode") or "phrase_tail"
    if mode not in ("phrase_tail", "full_asset"):
        return skip(f"unknown mode {mode}")
    f = Path(p["file"]); f = f if f.is_absolute() else (ep / f).resolve()
    if not f.exists():
        return skip(f"BGM file not found: {p['file']}")
    tm = load(ep / "work/timemap.json"); vad = load(ep / "work/vad.json")
    iv = [(map_time(tm, a_), map_time(tm, b_)) for s in speakers(ep) for a_, b_ in vad["speakers"][s]]
    speech_end = last_speech_end(iv, dur, p["min_island_s"])
    params = {k: p[k] for k in DEF if k not in ("file", "outro_file")}

    if mode == "full_asset":
        raw_outro = str(p.get("outro_file") or "").strip()
        outro = None
        if raw_outro:
            op = Path(raw_outro); op = op if op.is_absolute() else (ep / op).resolve()
            if not op.exists():
                return skip("outro file not found")
            outro = decode(op, sr)
        anchor = p.get("end_anchor") or "episode_end"
        if anchor == "speech_end" and speech_end is None and p.get("music_start_s") is None and p.get("outro_start_s") is None:
            return skip("no speech end for speech_end anchor")
        anchor_speech = float(speech_end) if speech_end is not None else 0.0
        gain_at = anchor_speech if p.get("gain_speech_end_s") is None else float(p["gain_speech_end_s"])
        music = decode(f, sr)
        mp = _payload(music, sr, p.get("bgm_offset_s") or 0.0)
        op = _payload(outro, sr, p.get("outro_offset_s") or 0.0) if outro is not None else None
        if len(mp) == 0:
            return skip("music payload empty")
        if outro is not None and (op is None or len(op) == 0):
            return skip("outro payload empty")
        placed, err = place_full_payloads(len(mp), 0 if op is None else len(op), sr, anchor_speech, dur, p)
        if err:
            return skip(err)
        starts = [placed["music_start"]]
        if placed["outro_start"] is not None:
            starts.append(placed["outro_start"])
        t0s = max(0.0, min(min(starts), gain_at) - 1.0); i0 = int(round(t0s * sr))
        with sf.SoundFile(str(voice_p)) as src:
            src.seek(i0); tail = src.read(dtype="float64", always_2d=True)[:, 0]
        ref_b = int(round((min(gain_at, dur) - t0s) * sr))
        ref_a = max(0, int(round((min(gain_at, dur) - p["voice_ref_window_s"] - t0s) * sr)))
        vdb = active_rms_db(tail[ref_a:max(ref_a + 1, ref_b)], sr)
        y, info = mix_full_asset(tail, music, outro, sr, anchor_speech, dur, gain_at, p, vdb, t0s)
        if y is None:
            return skip(info)
        write(y, i0)
        info.update({"voice_active_db": r(vdb, 1), "params": params, "speech_end_s": None if speech_end is None else r(speech_end)})
        dump(ep / "logs/ending_bgm.json", info)
        tm["ending_bgm"] = info; tm["edited_duration_s"] = r(info["end_s"], 6); dump(ep / "work/timemap.json", tm)
        log_step(ep, "ending_bgm", {k: info[k] for k in ("mode", "end_anchor", "end_anchor_met", "music_end_s", "shared_end_s", "end_s")})
        print("ending_bgm", {k: v for k, v in info.items() if k != "params"})
        return

    words = load(ep / "work/words.json")
    segs = []
    for s in speakers(ep):
        for w in words.get(s, []):
            if w.get("status") == "rejected":
                continue
            toks = [(map_time(tm, t["start"]) if t.get("start") is not None else None, t.get("surface", "")) for t in w.get("words", [])]
            segs.append((map_time(tm, w["start"]), w.get("text", ""), toks))
    hit = find_phrase(segs, p["phrases"], dur, p["search_window_s"])
    if hit is None:
        return skip(f"no closing phrase {p['phrases']} in the last {p['search_window_s']} s")
    t_end = max(speech_end or dur, hit[0])
    t0s = max(0.0, min(hit[0] - p["lead_in_s"], t_end - p["voice_ref_window_s"]) - 1.0); i0 = int(t0s * sr)
    with sf.SoundFile(str(voice_p)) as src:
        src.seek(i0); tail = src.read(dtype="float64", always_2d=True)[:, 0]
    ref = tail[max(0, int((t_end - p["voice_ref_window_s"] - t0s) * sr)):int((t_end - t0s) * sr)]
    vdb = active_rms_db(ref, sr)
    y, info = mix_bgm(tail, decode(f, sr), sr, hit[0] - t0s, t_end - t0s, p, vdb)
    for k in ("fade_in_start_s", "last_speech_end_s", "tail_end_s", "fade_out_end_s", "end_s"):
        info[k] = r(info[k] + t0s)
    write(y, i0)
    info.update({"applied": True, "mode": "phrase_tail", "phrase": hit[1], "phrase_s": r(hit[0]), "phrase_text": hit[2],
                 "voice_active_db": r(vdb, 1), "params": params})
    dump(ep / "logs/ending_bgm.json", info)
    tm["ending_bgm"] = info; tm["edited_duration_s"] = r(info["end_s"], 6); dump(ep / "work/timemap.json", tm)
    if info["bgm_short"]:
        print("WARNING ending_bgm: BGM shorter than the needed span", file=sys.stderr)
    log_step(ep, "ending_bgm", {k: info[k] for k in ("phrase_s", "last_speech_end_s", "end_s", "extended_s")})
    print("ending_bgm", {k: v for k, v in info.items() if k != "params"})


if __name__ == "__main__":
    main()
