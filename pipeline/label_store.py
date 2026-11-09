"""Step 4b: persistent labels and the single edit list.

labels.json (episode root, survives reruns) is keyed by candidate key = hash(type, track, times, excerpt, context).
Labels are never regenerated: `seed` only adds missing keys with the rule default.
  seed                       add rule labels for new candidates
  set KEY ACTION [--approve] [--by agent|human] [--note ...]
  approve --types a,b        bulk-approve current actions (test harness / explicit user instruction only)
  build                      write work/edits.json (fails if any candidate lacks a label)
  status                     counts
edits.json entry: {id, key, type, track, start, end, action, approved, labeler, reason, excerpt, confidence, note}
manual_edits.json (optional, episode root): [{start, end, track, action, approved, note}] for edits no rule can find
(e.g. a retake the ASR merged away). They are appended as type "manual" with key m-<start>-<end>.
laughter_edits.json (optional, episode root, never committed): laughter-only mutes the rules did not apply.
Each item is {track, start, end, action: "mute", kind: "laughter", approved: true|false, note?}.
One speaker track (not "all"); the timeline is not cut. Applied as type "laughter_local".
Structural edits (config `structure`, built every time, not labels):
  opening_chatter  cut [0, theme start - lead_s): theme start = first raw/transcript.csv line containing
                   manifest/config structure.opening_chatter.start_text, else the first line at >= min_s matching
                   start_markers; snapped to the VAD onset of that speech. Off when no transcript or no match.
  edge_trim        cut leading/trailing silence so at most structure.edge_silence_s remains (VAD-based).
Render applies an entry only if approved is true and action in {cut, duck, mute}."""
import argparse, sys
from pathlib import Path
from common import dump, load, load_cfg, r

ACTIONS = ("cut", "duck", "mute", "keep", "review")


def seed(cands, labels, cfg):
    auto = set(cfg["labels"]["auto_approve_types"]); n = 0
    for c in cands:
        if c["key"] in labels:
            continue
        a = c["default_action"]
        labels[c["key"]] = {"action": a, "labeler": "rule", "approved": a in ("cut", "duck", "mute", "keep") and c["type"] in auto, "note": ""}
        n += 1
    return n


def build(cands, labels):
    missing = [c["key"] for c in cands if c["key"] not in labels]
    if missing:
        raise SystemExit(f"unlabeled candidates: {len(missing)} (e.g. {missing[:3]}); run label_store.py seed")
    out = []
    for i, c in enumerate(cands):
        L = labels[c["key"]]
        if L["action"] not in ACTIONS:
            raise SystemExit(f"bad action {L['action']} for {c['key']}")
        out.append({"id": f"e{i:04d}", "key": c["key"], "type": c["type"], "track": c["track"], "start": c["start"], "end": c["end"],
                    "action": L["action"], "approved": bool(L["approved"]) and L["action"] != "review", "labeler": L["labeler"],
                    "reason": c["reason"], "excerpt": c["excerpt"], "confidence": c["confidence"], "note": L.get("note", "")})
    return out


def manual(ep):
    p = ep / "manual_edits.json"
    out = []
    for i, m in enumerate(load(p) if p.exists() else []):
        if m["action"] not in ACTIONS or not m["end"] > m["start"]:
            raise SystemExit(f"bad manual edit {m}")
        out.append({"id": f"m{i:04d}", "key": f"m-{m['start']:.3f}-{m['end']:.3f}", "type": "manual", "track": m.get("track", "all"),
                    "start": m["start"], "end": m["end"], "action": m["action"], "approved": bool(m.get("approved")) and m["action"] != "review",
                    "labeler": m.get("by", "agent"), "reason": "manual edit", "excerpt": m.get("excerpt", ""), "confidence": 1.0, "note": m.get("note", "")})
    return out


def laughter_local(ep):
    """Episode-local pure-laughter mutes. Rejects anything that is not one-track mute of kind laughter."""
    p = ep / "laughter_edits.json"
    if not p.exists():
        return []
    rows = load(p)
    if not isinstance(rows, list):
        raise SystemExit("laughter_edits.json must be a list")
    out = []
    for i, m in enumerate(rows):
        if not isinstance(m, dict):
            raise SystemExit(f"laughter_edits.json[{i}] must be an object")
        track = m.get("track")
        if m.get("kind") != "laughter" or m.get("action", "mute") != "mute" or not isinstance(track, str) or track.strip() in ("", "all"):
            raise SystemExit(f"laughter_edits.json[{i}] must be kind=laughter, action=mute, and one speaker track")
        if "approved" in m and not isinstance(m["approved"], bool):
            raise SystemExit(f"laughter_edits.json[{i}] approved must be a boolean")
        note = m.get("note", "")
        if not isinstance(note, str):
            raise SystemExit(f"laughter_edits.json[{i}] note must be a string")
        try:
            start, end = r(float(m["start"])), r(float(m["end"]))
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"laughter_edits.json[{i}] needs numeric start and end")
        if start < 0 or not end > start:
            raise SystemExit(f"laughter_edits.json[{i}] end must be after start")
        out.append({"id": f"l{i:04d}", "key": f"laugh-{i}-{track}-{start:.3f}-{end:.3f}", "type": "laughter_local",
                    "track": track, "start": start, "end": end, "action": "mute", "approved": m.get("approved") is True,
                    "labeler": "human", "reason": "episode-local overlapping-laughter mute; one speaker; timeline kept",
                    "excerpt": "", "confidence": 1.0, "note": note})
    return out


def structural(ep, cfg):
    S = cfg.get("structure") or {}
    vad = load(ep / "work/vad.json"); U = vad["union"]; dur = vad["duration_s"]; out = []
    if not U:
        return out
    pad = S.get("edge_silence_s", 0.4)
    head = max(0.0, U[0][0] - pad); oc = S.get("opening_chatter") or {}
    man = load(ep / "manifest.json"); oc = {**oc, **(man.get("opening_chatter") or {})}
    tp = ep / "raw/transcript.csv"
    if oc.get("enabled") and tp.exists():
        import re
        from zencastr_transcript import read_rows
        rows = sorted(read_rows(tp))
        off = cfg["transcript"].get("offset_s", 0.0)
        hit = None
        if oc.get("start_text"):
            hit = next((r_ for r_ in rows if oc["start_text"] in r_[2]), None)
        else:
            pat = re.compile("|".join(oc.get("start_markers", [])))
            hit = next((r_ for r_ in rows if r_[0] >= oc.get("min_s", 0) and r_[0] <= oc.get("max_s", 600) and pat.search(r_[2])), None)
        if hit:
            t = hit[0] + off
            onset = next((a for a, b in U if b > t - 0.5), t)
            cut = max(0.0, min(t, onset) - oc.get("lead_s", 0.3))
            if cut > head:
                head = cut
                out.append({"id": "s0000", "key": f"s-opening-{cut:.3f}", "type": "opening_chatter", "track": "all", "start": 0.0, "end": round(cut, 3),
                            "action": "cut", "approved": True, "labeler": "rule", "reason": "opening small talk before main theme",
                            "excerpt": hit[2], "confidence": 1.0, "note": f"theme starts at transcript {hit[0]:.2f}s"})
    if not out and head > 0.01:
        out.append({"id": "s0001", "key": f"s-head-{head:.3f}", "type": "edge_trim", "track": "all", "start": 0.0, "end": round(head, 3),
                    "action": "cut", "approved": True, "labeler": "rule", "reason": "leading silence", "excerpt": "", "confidence": 1.0, "note": ""})
    tail = U[-1][1] + pad
    if dur - tail > 0.01:
        out.append({"id": "s0002", "key": f"s-tail-{tail:.3f}", "type": "edge_trim", "track": "all", "start": round(tail, 3), "end": dur,
                    "action": "cut", "approved": True, "labeler": "rule", "reason": "trailing silence", "excerpt": "", "confidence": 1.0, "note": ""})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ep", required=True)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("seed"); sp.add_parser("build"); sp.add_parser("status")
    s = sp.add_parser("set"); s.add_argument("key"); s.add_argument("action", choices=ACTIONS)
    s.add_argument("--approve", action="store_true"); s.add_argument("--by", default="agent"); s.add_argument("--note", default="")
    b = sp.add_parser("approve"); b.add_argument("--types", required=True)
    a = ap.parse_args()
    ep = Path(a.ep); cfg = load_cfg(ep)
    cands = load(ep / "work/candidates.json")
    lp = ep / "labels.json"
    labels = load(lp) if lp.exists() else {}
    if a.cmd == "seed":
        n = seed(cands, labels, cfg); dump(lp, labels); print("seeded", n, "total", len(labels))
    elif a.cmd == "set":
        if a.key not in labels:
            sys.exit("unknown key")
        labels[a.key] = {"action": a.action, "labeler": a.by, "approved": a.approve, "note": a.note}; dump(lp, labels)
    elif a.cmd == "approve":
        types = set(a.types.split(",")); n = 0
        for c in cands:
            L = labels.get(c["key"])
            if L and c["type"] in types and L["action"] != "review" and not L["approved"]:
                L["approved"] = True; n += 1
        dump(lp, labels); print("approved", n)
    elif a.cmd == "build":
        e = build(cands, labels) + manual(ep) + laughter_local(ep) + structural(ep, cfg); dump(ep / "work/edits.json", e)
        print("edits", len(e), "applied", sum(1 for x in e if x["approved"] and x["action"] in ("cut", "duck", "mute")))
    else:
        from collections import Counter
        print(Counter((labels[c["key"]]["action"], labels[c["key"]]["approved"]) for c in cands if c["key"] in labels))


if __name__ == "__main__":
    main()
