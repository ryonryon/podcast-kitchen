"""Step 5a: apply approved edits from work/edits.json to every track in sync, mix (dual mono).

Order: track-local (duck, mute->room tone) then timeline cuts (track 'all'), identical on every track.
Join: each cut boundary is snapped to the lowest-energy 5 ms frame of the summed tracks within
render.snap_window_ms; at each join every track gets the same equal-power crossfade of
render.crossfade_ms (per-track definition, no speaker-change special case).
Room tone: quietest render.roomtone_s VAD-negative window of each track, looped with crossfades.
Input: work/gate/<spk>.wav when gate.enabled (else work/level). Outputs work/render/<spk>.wav, work/mix.wav, work/timemap.json. Fails on unlabeled candidates."""
import argparse
from pathlib import Path
import numpy as np
from common import dump, load, load_cfg, log_step, read_wav, speakers, write_wav, r

APPLY = ("cut", "duck", "mute")
# Episode-local / structural edits are not required to have a candidates.json key.
EXTRA_EDIT_TYPES = ("manual", "opening_chatter", "edge_trim", "laughter_local")


def roomtone(x, sr, isl, secs, rise_db=15, floor_min_db=-80):
    """Quietest VAD-free window of `secs` that is also burst-free (no 10 ms frame more than rise_db above the
    track noise floor, the gate QA burst threshold) -> (tone, start_s). Falls back to the quietest VAD-free window,
    then the quietest window regardless."""
    n = int(secs * sr); hop = int(0.1 * sr)
    mask = np.ones(len(x), bool)
    for s, e in isl:
        mask[max(0, int((s - 0.1) * sr)):int((e + 0.1) * sr)] = False
    m = len(x) // hop; k = max(1, n // hop)
    fe = np.zeros(m); fpk = np.zeros(m); fr = max(1, int(0.01 * sr))
    for c0 in range(0, m, 100000):
        c1 = min(m, c0 + 100000); blk = x[c0 * hop:c1 * hop].astype(np.float64)
        fe[c0:c1] = (blk ** 2).reshape(c1 - c0, hop).sum(1)
        p10 = (blk[: (len(blk) // fr) * fr] ** 2).reshape(-1, fr).mean(1)
        per = hop // fr
        fpk[c0:c1] = p10[: (c1 - c0) * per].reshape(c1 - c0, per).max(1)
    db10 = 10 * np.log10(fpk + 1e-12)
    fdb = 10 * np.log10(fe / hop + 1e-12)
    ref = np.percentile(fdb[fdb > -119], 10) if np.any(fdb > -119) else -120
    hot = db10 > max(ref, floor_min_db) + rise_db
    fb = (~mask[: m * hop]).reshape(m, hop).any(1)
    cs = np.concatenate([[0], np.cumsum(fe)]); bad = np.concatenate([[0], np.cumsum(fb)])
    hc = np.concatenate([[0], np.cumsum(hot)])
    best, bi, best2, bi2 = None, 0, None, 0
    for j in range(0, max(1, m - k)):
        if bad[j + k] - bad[j]:
            continue
        e = cs[j + k] - cs[j]
        if best2 is None or e < best2:
            best2, bi2 = e, j * hop
        if hc[j + k] - hc[j]:
            continue
        if best is None or e < best:
            best, bi = e, j * hop
    if best is None:
        best, bi = best2, bi2
    if best is None:  # no VAD-negative window: quietest window regardless
        e = cs[k:] - cs[:-k]; bi = int(np.argmin(e)) * hop
    return x[bi:bi + n].copy(), bi / sr


def loop(tone, n, sr, fade):
    """Loop tone to n samples with equal-power crossfades of `fade` samples."""
    if n <= 0:
        return np.zeros(0)
    t = np.linspace(0, np.pi / 2, fade)
    out = tone.copy()
    while len(out) < n:
        a = out[-fade:] * np.cos(t) + tone[:fade] * np.sin(t)
        out = np.concatenate([out[:-fade], a, tone[fade:]])
    return out[:n]


def xfade_replace(x, s, e, repl, fade):
    """Replace x[s:e] by repl (len e-s) with equal-power crossfades at both edges."""
    y = x.copy(); seg = repl.copy()
    f = min(fade, (e - s) // 2)
    if f > 0:
        t = np.linspace(0, np.pi / 2, f)
        seg[:f] = x[s:s + f] * np.cos(t) + seg[:f] * np.sin(t)
        seg[-f:] = seg[-f:] * np.cos(t) + x[e - f:e] * np.sin(t)
    y[s:e] = seg
    return y


def duck(x, s, e, db, fade):
    g = np.ones(len(x)); lo = 10 ** (db / 20)
    g[s:e] = lo
    f1 = min(fade, s); f2 = min(fade, len(x) - e)
    if f1: g[s - f1:s] = np.linspace(1, lo, f1)
    if f2: g[e:e + f2] = np.linspace(lo, 1, f2)
    return x * g


def write_dual_mono(path, mono, sr, block=1 << 20):
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(path), "w", sr, 2, "FLOAT") as f:
        for i in range(0, len(mono), block):
            b = mono[i:i + block]; f.write(np.stack([b, b], axis=1))


def merge(iv):
    out = []
    for s, e, ids in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e); out[-1][2] += ids
        else:
            out.append([s, e, list(ids)])
    return out


def snap(t, energy, sr, win):
    fr = int(0.005 * sr); i = int(t * sr) // fr; w = int(win * sr) // fr
    lo, hi = max(0, i - w), min(len(energy) - 1, i + w)
    k = lo + int(np.argmin(energy[lo:hi + 1]))
    return k * fr + fr // 2


def apply_cuts(tracks, cuts, sr, cf):
    """cuts: merged sorted [(s, e, ids)] in samples. Equal-power crossfade of cf samples centred on each
    join, identical for every track. Output length = n - sum(e - s). Returns tracks, timemap spans, joins."""
    h = cf // 2
    t = np.linspace(0, np.pi / 2, 2 * h) if h else None
    n = len(next(iter(tracks.values())))
    cuts = [(max(s, h), min(e, n - h), ids) for s, e, ids in cuts]
    out = {}
    for k, x in tracks.items():
        pieces, cur = [], 0
        for s, e, _ in cuts:
            pieces.append(x[cur:s - h])
            if h:
                pieces.append(x[s - h:s + h] * np.cos(t) + x[e - h:e + h] * np.sin(t))
            cur = e + h
        pieces.append(x[cur:n])
        out[k] = np.concatenate(pieces)
    spans, joins, removed, prev = [], [], 0, 0
    for s, e, ids in cuts:
        spans.append([r(prev / sr, 6), r(s / sr, 6), r((prev - removed) / sr, 6)])
        joins.append({"out_s": r((s - removed) / sr, 4), "raw_cut": [r(s / sr, 4), r(e / sr, 4)], "ids": ids})
        removed += e - s; prev = e
    spans.append([r(prev / sr, 6), r(n / sr, 6), r((prev - removed) / sr, 6)])
    return out, spans, joins


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ep", required=True); a = ap.parse_args()
    ep = Path(a.ep); cfg = load_cfg(ep); c = cfg["render"]
    cands = load(ep / "work/candidates.json"); labels = load(ep / "labels.json") if (ep / "labels.json").exists() else {}
    missing = [x["key"] for x in cands if x["key"] not in labels]
    if missing:
        raise SystemExit(f"render: {len(missing)} unlabeled candidates; refusing to render")
    edits = load(ep / "work/edits.json")
    if {e["key"] for e in edits if e["type"] not in EXTRA_EDIT_TYPES} != {x["key"] for x in cands}:
        raise SystemExit("render: edits.json is stale vs candidates.json; run label_store.py build")
    vad = load(ep / "work/vad.json")
    spk = speakers(ep)
    tracks = {}
    sr = None
    gated = (cfg.get("gate") or {}).get("enabled") and (ep / "work/gate/.done").exists()
    tones = {}
    for s in spk:  # room tone always from the ungated levelled track (a gated track's quiet parts are digital silence)
        x, sr = read_wav(ep / "work/level" / f"{s}.wav")
        tones[s] = roomtone(x, sr, vad["speakers"][s], c["roomtone_s"], cfg.get("gate", {}).get("transient_rise_db", 15), cfg.get("gate", {}).get("noise_floor_min_db", -80)); del x
    for s in spk:
        x, sr = read_wav(ep / ("work/gate" if gated else "work/level") / f"{s}.wav"); tracks[s] = x
    n = min(len(x) for x in tracks.values()); tracks = {k: v[:n] for k, v in tracks.items()}
    fade_d = int(c["duck_fade_ms"] / 1000 * sr); cf = int(c["crossfade_ms"] / 1000 * sr)
    applied = [e for e in edits if e["approved"] and e["action"] in APPLY]
    for e in applied:
        if e["track"] == "all":
            continue
        s0, s1 = int(e["start"] * sr), int(e["end"] * sr)
        x = tracks[e["track"]]
        if e["action"] == "duck":
            tracks[e["track"]] = duck(x, s0, s1, c["duck_db"], fade_d)
        elif e["action"] == "mute":
            tracks[e["track"]] = xfade_replace(x, s0, s1, loop(tones[e["track"]][0], s1 - s0, sr, cf), cf)
        elif e["action"] == "cut":
            raise SystemExit(f"{e['id']}: cut must be on track 'all' (use mute for one track)")
    mix = sum(tracks.values())
    fr = int(0.005 * sr); m = len(mix) // fr
    energy = np.mean(mix[: m * fr].reshape(m, fr) ** 2, axis=1)
    raw_cuts = []
    for e in applied:
        if e["track"] == "all" and e["action"] == "cut":
            s0 = snap(e["start"], energy, sr, c["snap_window_ms"] / 1000); s1 = snap(e["end"], energy, sr, c["snap_window_ms"] / 1000)
            if s1 - s0 > 2 * cf:
                raw_cuts.append((s0, s1, [e["id"]]))
    cuts = merge(raw_cuts)
    del mix, energy
    out, spans, joins = apply_cuts(tracks, cuts, sr, cf)
    tracks.clear()  # memory: raw tracks no longer needed (full episodes OOM otherwise)
    n_out = len(next(iter(out.values())))
    mono = np.zeros(n_out)
    for s in spk:
        write_wav(ep / "work/render" / f"{s}.wav", out[s], sr); mono += out[s]; out[s] = None
    write_dual_mono(ep / "work/mix.wav", mono, sr)  # dual mono: loudness measured as delivered
    removed = sum(e - s for s, e, _ in cuts) / sr
    dump(ep / "work/timemap.json", {"sr": sr, "spans": spans, "joins": joins, "raw_duration_s": r(n / sr, 6),
                                     "edited_duration_s": r(n_out / sr, 6), "removed_s": r(removed, 3),
                                     "roomtone_at_s": {s: r(tones[s][1]) for s in spk}})
    log_step(ep, "render", {"applied": len(applied), "timeline_cuts": len(cuts), "removed_s": r(removed)})
    print("render", {"applied": len(applied), "timeline_cuts": len(cuts), "removed_s": r(removed)})


def map_time(tm, t):
    """raw seconds -> edited seconds (inside a cut maps to the join)."""
    for a, b, o in tm["spans"]:
        if t < a:
            return o
        if t <= b:
            return o + (t - a)
    a, b, o = tm["spans"][-1]
    return o + (b - a)


if __name__ == "__main__":
    main()
