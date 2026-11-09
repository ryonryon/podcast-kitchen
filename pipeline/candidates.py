"""Step 4a: deterministic rule candidates -> work/candidates.json (raw seconds, after sync).

Types (config.candidates, config.text): long_pause, filler, backchannel, overlap_yield, overlap_restart,
overlap_abandon, mutual_yield, false_start, overlap_laughter,
crosstalk, mistake. Every candidate carries `default_action` from a measurable proxy; anything the
rules cannot decide is `review`. Timeline cuts use track 'all'; duck/mute act on one track only.
overlap_laughter never cuts the timeline and never mutes speech, questions, backchannels, or mixed speech/laughter."""
import argparse
import re
from pathlib import Path
from common import dump, key, load, load_cfg, log_step, speakers, r

TIMELINE = "all"


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def speech(isl, t0, t1):
    return sum(overlap(s, e, t0, t1) for s, e in isl)


def norm(s):
    return s.replace("〜", "ー").replace("~", "ー").strip("、。!?！？ ")


class Ctx:
    def __init__(self, vad, words, cfg, roles=None):
        self.vad, self.words, self.cfg = vad, words, cfg
        self.roles = roles if isinstance(roles, dict) else {}
        self.c, self.t = cfg["candidates"], cfg["text"]
        self.dur = vad["duration_s"]
        self.spk = sorted(vad["speakers"])
        self.segs = sorted(((g["start"], g["end"], s, g) for s in self.spk for g in words.get(s, [])), key=lambda x: (x[0], x[2]))
        self.laughs = self._laughs()

    def others(self, spk, t0, t1):
        return sum(speech(self.vad["speakers"][o], t0, t1) for o in self.spk if o != spk)

    def context(self, t0, t1):
        cs = self.c["context_s"]
        return " / ".join(f"{s}: {g['text']}" for a, b, s, g in self.segs if b >= t0 - cs and a <= t1 + cs)

    def excerpt(self, spk, t0, t1):
        segs = [g for a, b, s, g in self.segs if (spk in (s, TIMELINE)) and overlap(a, b, t0 - 0.05, t1 + 0.05) > 0]
        return " ".join(g["text"] for g in segs)[:80]

    def _laughs(self):
        """Laugh proxy: text markers, or short VAD islands with no aligned text (laugh/backchannel/filler)."""
        z = []
        for a, b, s, g in self.segs:
            if any(m in g["text"] for m in self.t["laugh_markers"]):
                z.append((a, b, s, "text"))
        for s in self.spk:
            for a, b in self.acoustic(s):
                z.append((a, b, s, "acoustic"))
        return sorted(z)

    def acoustic(self, s):
        """Short VAD islands with no ASR text, or covered only by aligner-rejected segments."""
        out = []
        for a, b in self.vad["speakers"][s]:
            if b - a > self.c["acoustic_island_max_s"]:
                continue
            txt = [g for g in self.words.get(s, []) if g["status"] != "rejected" and overlap(g["start"], g["end"], a, b) > 0]
            if not txt:
                out.append((a, b))
        return out

    def near_laugh(self, t0, t1, pad):
        return any(overlap(a, b, t0 - pad, t1 + pad) > 0 for a, b, _, k in self.laughs if k == "text")

    def protected_edge(self, t0, t1):
        p = self.c["protect_head_tail_s"]
        return t0 < p or t1 > self.dur - p


def cand(ctx, typ, track, t0, t1, action, reason, conf):
    ex = ctx.excerpt(track, t0, t1)
    con = ctx.context(t0, t1)
    return {"key": key(typ, track, r(t0), r(t1), ex, con), "type": typ, "track": track, "start": r(t0), "end": r(t1),
            "default_action": action, "reason": reason, "excerpt": ex, "context": con, "confidence": r(conf, 2)}


def long_pauses(ctx):
    lp = ctx.c["long_pause"]; u = ctx.vad["union"]; out = []
    for (a0, g0), (g1, b1) in zip(u, u[1:]):
        gap = g1 - g0
        prev = max(ctx.spk, key=lambda s: max([e for s0, e in ctx.vad["speakers"][s] if e <= g0 + 1e-6] or [-1]))
        nxt = min(ctx.spk, key=lambda s: min([s0 for s0, e in ctx.vad["speakers"][s] if s0 >= g1 - 1e-6] or [1e9]))
        thr = lp["within_turn_min_s"] if prev == nxt else lp["between_speakers_min_s"]
        if gap <= thr:
            continue
        h = lp["target_s"] / 2
        t0, t1 = g0 + h, g1 - h
        after = [z for z in ctx.laughs if g1 - 0.05 <= z[0] <= g1 + lp["laugh_lookahead_s"]]
        if after:
            act, why = "review", f"gap {gap:.2f}s followed by possible laugh/reaction ({after[0][3]}) -> possible punchline pause"
        else:
            act, why = "cut", f"gap {gap:.2f}s ({'within turn' if prev == nxt else 'speaker change'}) -> {lp['target_s']}s"
        out.append(cand(ctx, "long_pause", TIMELINE, t0, t1, act, why, 0.9))
    return out


def fillers(ctx):
    out = []; iso = ctx.c["filler_isolation_s"]; fl = set(ctx.t["fillers"])
    for s in ctx.spk:
        for g in ctx.words.get(s, []):
            ws = g["words"]
            for i, w in enumerate(ws):
                if w["start"] is None or w["pos"] not in ctx.t["filler_pos"] or norm(w["surface"]) not in fl:
                    continue
                prev_end = max([x["end"] for x in ws[:i] if x["end"] is not None] or [g["start"] - 1])
                next_start = min([x["start"] for x in ws[i + 1:] if x["start"] is not None] or [g["end"] + 1])
                if w["start"] - prev_end < iso or next_start - w["end"] < iso:
                    continue  # embedded in running speech: cutting would change rhythm
                t0, t1 = w["start"], w["end"]
                if g["status"] != "ok" or (w["score"] or 0) < ctx.cfg["align"]["min_word_score"]:
                    act, why = "review", f"filler '{w['surface']}' but alignment score low"
                elif ctx.protected_edge(t0, t1) or ctx.near_laugh(t0, t1, ctx.c["laugh_protect_s"]):
                    act, why = "review", "filler in protected head/tail or near laugh"
                elif ctx.others(s, t0, t1) > 0:
                    out.append(cand(ctx, "filler", s, t0, t1, "mute", f"filler '{w['surface']}' over other speech -> mute own track", w["score"])); continue
                elif speech(ctx.vad["speakers"][s], t1, t1 + iso) >= iso - 1e-6:
                    out.append(cand(ctx, "filler", s, t0, t1, "mute", f"filler '{w['surface']}' runs straight into speech (VAD) -> mute, a cut would click", w["score"])); continue
                else:
                    act, why = "cut", f"isolated filler '{w['surface']}' ({w['pos']})"
                out.append(cand(ctx, "filler", TIMELINE if act == "cut" else s, t0, t1, act, why, w["score"] or 0))
    return out


def backchannels(ctx):
    out = []; bc = set(ctx.t["backchannels"]); last = {}
    for s in ctx.spk:
        for g in ctx.words.get(s, []):
            toks = [norm(w["surface"]) for w in g["words"]]
            if not toks or g["end"] - g["start"] > ctx.c["backchannel_max_s"]:
                continue
            if not (norm(g["text"]) in bc or all(t in bc or t in ("うん", "はい", "そう") for t in toks)):
                continue
            t0, t1 = g["start"], g["end"]
            if ctx.others(s, t0, t1) < 0.5 * (t1 - t0):
                continue  # not over someone else's speech: it is a turn, not a backchannel
            if ctx.near_laugh(t0, t1, ctx.c["laugh_protect_s"]):
                act, why = "keep", "reaction near laugh"
            elif t0 - last.get(s, -1e9) >= ctx.c["backchannel_keep_every_s"]:
                act, why = "keep", f"first backchannel in {ctx.c['backchannel_keep_every_s']}s window"
                last[s] = t0
            elif g["status"] != "ok":
                act, why = "review", "excess backchannel, alignment score low"
            else:
                act, why = "duck", f"excess backchannel -> duck {ctx.cfg['render']['duck_db']} dB on own track"
            out.append(cand(ctx, "backchannel", s, t0, t1, act, why, g["score"]))
    # acoustic: speech islands with no aligned text (kotoba drops fillers/相槌) -> review only
    for s in ctx.spk:
        for a, b in ctx.acoustic(s):
            typ = "backchannel" if ctx.others(s, a, b) > 0.5 * (b - a) else "filler"
            out.append(cand(ctx, typ, s, a, b, "review", "acoustic: VAD speech with no aligned text (could be laugh)", 0.3))
    return out


def overlap_yield(ctx):
    """Two onsets within overlap_onset_s; the yielder (turn start: silent >= yield_pre_silence_s before) stops within
    yield_max_s while the other keeps >= continue_min_s of speech. Only the yielder's own track is muted, and only
    when the yielder then says a yield marker (どうぞ/ごめん...); otherwise review (it may be a finished phrase)."""
    out = []; c = ctx.c; bc = set(ctx.t["backchannels"]); marks = ctx.t["yield_markers"]
    V = ctx.vad["speakers"]
    for y in ctx.spk:
        prev_end = -1e9
        for a0, a1 in V[y]:
            pre, prev_end = a0 - prev_end, a1
            if a1 - a0 > c["yield_max_s"] or pre < c["yield_pre_silence_s"]:
                continue
            txt = ctx.excerpt(y, a0, a1)
            if not txt or norm(txt) in bc or any(m in txt for m in marks):
                continue
            for k in ctx.spk:
                if k == y:
                    continue
                hit = [(b0, b1) for b0, b1 in V[k] if abs(b0 - a0) <= c["overlap_onset_s"]]
                if not hit or speech(V[k], hit[0][0], hit[0][0] + c["continue_window_s"]) < c["continue_min_s"]:
                    continue
                after = ctx.excerpt(y, a1, a1 + c["yield_marker_window_s"])
                marked = any(m in after for m in marks)
                prot = ctx.protected_edge(a0, a1) or ctx.near_laugh(a0, a1, c["laugh_protect_s"])
                # real-episode QA: yielders said ごめん after COMPLETE phrases (おめでとうございます, 使ってくれる人多いから),
                # so a marker does not prove the phrase was aborted -> never auto-mute; marker only raises confidence
                act = "review"
                why = (f"{y} yields to {k} (onsets {abs(hit[0][0]-a0):.2f}s apart" + (", then yield marker" if marked else ", no yield marker")
                       + f"); only {y}'s aborted phrase on {y}'s track")
                out.append(cand(ctx, "overlap_yield", y, a0, a1, act, why, 0.8 if marked else 0.4))
                break
    return out


CONTENT_POS = ("名詞", "動詞", "形容詞", "形状詞", "副詞")


def reading(w):
    """Reading-normalized token: katakana reading (or surface), long-vowel marks unified, small tsu dropped."""
    k = (w.get("kana") or w.get("surface") or "")
    k = "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in k)   # hiragana -> katakana
    return norm(k).replace("ッ", "")


def content_words(ctx, g):
    stop = set(ctx.t["backchannels"]) | set(ctx.t["fillers"])
    light = {reading({"kana": k}) for k in ctx.c.get("overlap_restart", {}).get("light_words", [])}
    return {reading(w) for w in g.get("words", []) if reading(w) not in light} & {reading(w) for w in g.get("words", []) if w.get("kana") or w.get("surface")
            if (w.get("pos") or "名詞").split("-")[0] in CONTENT_POS and norm(w.get("surface", "")) not in stop
            and len(reading(w)) >= 2}


def overlap_restart(ctx):
    """Speaker X starts a short phrase (<= overlap_restart.max_s) and another speaker starts within
    overlap_onset_s and overlaps it; X then restarts within restart_window_s with a phrase sharing >= 1 content
    word (reading-normalized) -> X's aborted phrase is muted on X's track only (timeline intact, other speaker kept).
    Auto-decided (labels.auto_approve_types); never touches head/tail or laughs."""
    c = ctx.c.get("overlap_restart", {})
    if not c.get("enabled", True):
        return []
    mx, win, ons = c.get("max_s", 2.0), c.get("restart_window_s", 2.5), ctx.c["overlap_onset_s"]
    out = []
    for x in ctx.spk:
        gs = [g for g in ctx.words.get(x, []) if g["status"] == "ok"]
        for i, g in enumerate(gs):
            a0, a1 = g["start"], g["end"]
            if a1 - a0 > mx or norm(g["text"]) in set(ctx.t["backchannels"]):
                continue
            other = [(o, h) for o in ctx.spk if o != x for h in ctx.words.get(o, [])
                     if h["status"] == "ok" and abs(h["start"] - a0) <= ons and overlap(h["start"], h["end"], a0, a1) > 0]
            lead = c.get("interject_lead_s", 0.5)
            inter = [] if other else [(o, h) for o in ctx.spk if o != x for h in ctx.words.get(o, [])
                                      if h["status"] == "ok" and h["start"] <= a0 - lead and h["end"] > a0
                                      and norm(h["text"]) not in set(ctx.t["backchannels"])]
            if inter and re.search(c.get("interject_end_re", r"(か|\?|？|。|！|!)\s*$"), g["text"].strip()):
                continue  # 「2022年か」「前半?後半?」「すごいね。」: a finished reaction, not an aborted phrase
            other = other or inter
            if not other:
                continue
            nxt = next((h for h in gs[i + 1:] if h["start"] >= a1 - 1e-6), None)
            if nxt is None or nxt["start"] - a1 > win:
                continue
            shared = content_words(ctx, g) & content_words(ctx, nxt)
            if not shared:
                continue
            if ctx.protected_edge(a0, a1) or ctx.near_laugh(a0, a1, ctx.c["laugh_protect_s"]):
                continue
            o, h = other[0]
            how = (f"inside {o}'s ongoing line (started {a0-h['start']:.2f}s earlier)" if inter
                   else f"as {o} started ({abs(h['start']-a0):.2f}s apart)")
            why = (f"{x} aborted 「{g['text']}」 {how}, restarted "
                   f"{nxt['start']-a1:.2f}s later 「{nxt['text']}」 (shared: {'/'.join(sorted(shared))}); mute {x}'s track only")
            out.append(cand(ctx, "overlap_restart", x, a0, a1, "mute", why, 0.8))
    return out


def overlap_abandon(ctx):
    """X's short phrase (<= max_s, not a backchannel/filler/laugh) lies inside Y's ongoing substantive line, Y keeps
    talking >= continue_after_s past X's end, X stays silent for silent_after_s, and the phrase is not
    sentence-ending -> X gave up: mute X's track only (timeline and Y untouched). Auto-approved."""
    c = ctx.c.get("overlap_abandon", {})
    if not c.get("enabled", True):
        return []
    mx, cont, sil = c.get("max_s", 2.0), c.get("continue_after_s", 1.0), c.get("silent_after_s", 3.0)
    end_re = c.get("sentence_end_re", r"(。|！|!|\?|？|か|です|ます|でした|ました|よね|ね|よ|な|かな|だ|た|る|い|う)\s*$")
    light = set(ctx.t["backchannels"]) | set(ctx.t["fillers"]) | {"うん", "はい", "そう"}
    V = ctx.vad["speakers"]; out = []
    for x in ctx.spk:
        gs = [g for g in ctx.words.get(x, []) if g["status"] == "ok"]
        for g in gs:
            a0, a1, txt = g["start"], g["end"], g["text"].strip()
            if a1 - a0 > mx or not norm(txt) or norm(txt) in light or all(norm(w["surface"]) in light for w in g["words"]):
                continue
            if any(m in txt for m in ctx.t["laugh_markers"]) or re.search(end_re, txt):
                continue
            ys = [(y, h) for y in ctx.spk if y != x for h in ctx.words.get(y, []) if h["status"] == "ok"
                  and h["start"] < a0 and h["end"] >= a1 + cont and norm(h["text"]) not in light]
            if not ys:
                continue
            if any(h["start"] > a1 + 1e-6 and h["start"] < a1 + sil for h in ctx.words.get(x, []) if h["status"] != "rejected"):
                continue
            if speech(V[x], a1 + c.get("vad_tail_s", 0.2), a1 + sil) > c.get("vad_tolerance_s", 0.15):
                continue
            if ctx.protected_edge(a0, a1) or ctx.near_laugh(a0, a1, ctx.c["laugh_protect_s"]):
                continue
            y, h = ys[0]
            why = (f"{x} abandoned 「{txt}」 inside {y}'s line 「{h['text'][:30]}」 ({y} continues {h['end']-a1:.2f}s, "
                   f"{x} silent {sil}s after); mute {x}'s track only")
            out.append(cand(ctx, "overlap_abandon", x, a0, a1, "mute", why, 0.7))
    return out


def mutual_yield(ctx):
    """Both speakers start within overlap_onset_s, both stop within yield_max_s, one of them says a mutual_yield
    marker (どうぞ/お先に) within yield_marker_window_s, and the OTHER restarts -> timeline cut from the first onset
    to gap_s before the restart (the collision, the どうぞ and any aizuchi between go; a natural gap remains).
    Review instead when a third speaker talks inside, or head/tail/laugh is involved. Auto-approved."""
    c = ctx.c.get("mutual_yield", {})
    if not c.get("enabled", True):
        return []
    ons, ym, mw = ctx.c["overlap_onset_s"], ctx.c["yield_max_s"], ctx.c["yield_marker_window_s"]
    marks, gap, rw = c.get("markers", ["どうぞ", "お先に"]), c.get("gap_s", 0.4), c.get("restart_window_s", 3.0)
    light = set(ctx.t["backchannels"]) | {"うん", "はい", "そう"}
    G = {s: [g for g in ctx.words.get(s, []) if g["status"] != "rejected"] for s in ctx.spk}
    out = []
    for i, a in enumerate(ctx.spk):
        for b in ctx.spk[i + 1:]:
            for ga in G[a]:
                if ga["end"] - ga["start"] > ym or any(m in ga["text"] for m in marks):
                    continue
                for gb in G[b]:
                    if abs(gb["start"] - ga["start"]) > ons or gb["end"] - gb["start"] > ym or any(m in gb["text"] for m in marks):
                        continue
                    t0, stop = min(ga["start"], gb["start"]), max(ga["end"], gb["end"])
                    mk = sorted([(m["start"], s, m) for s in (a, b) for m in G[s]
                                 if any(k in m["text"] for k in marks) and stop - 0.05 <= m["start"] <= stop + mw], key=lambda z: z[0])
                    if not mk:
                        continue
                    _, giver, m = mk[0]
                    other = b if giver == a else a
                    rs = next((h for h in G[other] if h["start"] >= m["start"] and norm(h["text"]) not in light
                               and not any(k in h["text"] for k in marks)), None)
                    if rs is None or rs["start"] - m["end"] > rw:
                        continue
                    t1 = rs["start"] - gap
                    if t1 - t0 < 0.3:
                        continue
                    third = sum(speech(ctx.vad["speakers"][s], t0, t1) for s in ctx.spk if s not in (a, b))
                    rev = third > 0.1 or ctx.protected_edge(t0, t1) or ctx.near_laugh(t0, t1, ctx.c["laugh_protect_s"])
                    why = (f"{a}/{b} collided ({abs(gb['start']-ga['start']):.2f}s apart), {giver} said 「{m['text']}」, {other} restarted "
                           f"「{rs['text'][:30]}」 -> cut collision to {gap}s before restart" + ("; third speaker/protected -> review" if rev else ""))
                    out.append(cand(ctx, "mutual_yield", TIMELINE, t0, t1, "review" if rev else "cut", why, 0.8))
                    break
    return out


def false_starts(ctx):
    out = []; c = ctx.c
    global BC_FL
    BC_FL = set(ctx.t["backchannels"]) | set(ctx.t["fillers"]) | {"うん", "はい", "そう"}
    for s in ctx.spk:
        ws = [w for g in ctx.words.get(s, []) if g["status"] == "ok" for w in g["words"] if w["start"] is not None and w["kana"]]
        i = 0
        while i < len(ws):
            hit = None
            for n in (3, 2):
                if i + n > len(ws):
                    continue
                k1 = "".join(w["kana"] for w in ws[i:i + n])
                if len(k1) < c["false_start_min_kana"]:
                    continue
                if all(norm(w["surface"]) in BC_FL for w in ws[i:i + n]):
                    continue  # repeated はい/うん/えー is not a false start
                for j in range(i + n, min(len(ws), i + n + 3)):  # restart within 2 words of the attempt
                    if ws[j]["start"] - ws[i]["start"] > c["false_start_window_s"]:
                        break
                    if "".join(w["kana"] for w in ws[j:j + n]) == k1 and ws[j]["start"] - ws[i + n - 1]["end"] < 1.5:
                        # only an immediate restart (nothing but fillers between) is a confident false start;
                        # 'X-というか Y-というか' / 'AIなのか人間なのか' are parallel structure, cutting changes meaning
                        imm = all(norm(w["surface"]) in BC_FL for w in ws[i + n:j])
                        hit = (ws[i]["start"], ws[j]["start"], imm); break
                if hit:
                    break
            if hit:
                t0, t1, imm = hit
                tl = imm and ctx.others(s, t0, t1) == 0 and not ctx.protected_edge(t0, t1) and not ctx.near_laugh(t0, t1, c["laugh_protect_s"])
                out.append(cand(ctx, "false_start", TIMELINE if tl else s, t0, t1, "cut" if tl else "review",
                                "immediate restart; first attempt" if imm else "repeated phrase with words between (parallel structure?)", 0.5 if imm else 0.3))
                i = next((m for m, w in enumerate(ws) if w["start"] >= t1), len(ws))
            else:
                i += 1
    return out


def crosstalk(ctx):
    out = []; V = ctx.vad["speakers"]
    for i, a in enumerate(ctx.spk):
        for b in ctx.spk[i + 1:]:
            for a0, a1 in V[a]:
                for b0, b1 in V[b]:
                    o0, o1 = max(a0, b0), min(a1, b1)
                    if o1 - o0 >= ctx.c["crosstalk_min_overlap_s"]:
                        out.append(cand(ctx, "crosstalk", TIMELINE, o0, o1, "review", f"{a}+{b} overlap {o1-o0:.1f}s", 0.5))
    return out


def mistakes(ctx):
    return [cand(ctx, "mistake", TIMELINE, a, b, "review", "edit marker phrase", 0.5)
            for a, b, s, g in ctx.segs if any(m in g["text"] for m in ctx.t["mistake_markers"])]


_LAUGH_STRIP = set(" \t\n\r、。！!・…「」『』（）()[]【】〜~ー－-っッ\"'“”．.,　")
_W_TRANS = str.maketrans({"Ｗ": "w", "ｗ": "w", "W": "w"})
_LAUGH_WHY = {
    "edge": "overlapping pure laughter in protected head/tail; human review, no mute",
    "too_many": "overlapping pure laughter above max_decided_speakers; human review, no mute",
    "pair_review": "overlapping pure laughter; two speakers and no configured winner; human review, no mute",
    "pair_keep": "overlapping pure laughter; configured two-speaker winner kept on this track",
    "pair_mute": "overlapping pure laughter; mute this track's laughter interval only; timeline kept",
    "triple_review": "overlapping pure laughter; no unambiguous unique speaker who is neither interviewee nor primary host; human review, no mute",
    "triple_keep": "overlapping pure laughter; unique speaker who is neither interviewee nor primary host kept",
    "triple_mute": "overlapping pure laughter; mute this track's laughter interval only; unique third kept; timeline kept",
    "span_conflict": "overlapping pure laughter; full_span win/lose conflict; human review, no mute",
}


def is_question(text):
    t = str(text or "")
    return "?" in t or "？" in t


def _laugh_consume(s, markers):
    if s == "":
        return True
    for m in sorted(markers, key=len, reverse=True):
        if not m or not s.startswith(m):
            continue
        if _laugh_consume(s[len(m):], markers):
            return True
        if len(set(m)) == 1:
            k = len(m)
            while k < len(s) and s[k] == m[0]:
                k += 1
            if k > len(m) and _laugh_consume(s[k:], markers):
                return True
    return False


def is_pure_laugh(text, markers, limit=40):
    """True when the whole string is laugh markers only (repetitions allowed). Questions and other text are not."""
    raw = str(text or "").strip()
    if not raw or is_question(raw) or len(raw) > limit:
        return False
    s = "".join(ch for ch in raw if ch not in _LAUGH_STRIP)
    if not s or len(s) > limit:
        return False
    s = s.translate(_W_TRANS)
    norms = []
    for m in markers or []:
        mm = str(m).translate(_W_TRANS)
        if mm:
            norms.append(mm)
    return bool(norms) and _laugh_consume(s, norms)


def _clean_roles(roles):
    out = {}
    if not isinstance(roles, dict):
        return out
    for k, v in roles.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _role_table(ol):
    raw = ol.get("two_speaker_role_winners") or {}
    table = {}
    if not isinstance(raw, dict):
        return table
    for k, v in raw.items():
        parts = [p.strip() for p in str(k).split("+")]
        if len(parts) != 2 or not parts[0] or not parts[1] or not isinstance(v, str) or not v.strip():
            continue
        table["+".join(sorted(parts))] = v.strip()
    return table


def _pair_winner(speakers, roles, ol):
    """None unless two_speaker is role_map and the pair has exactly one configured winning role."""
    if str(ol.get("two_speaker", "review")) != "role_map":
        return None
    sp = list(speakers)
    if len(sp) != 2:
        return None
    r1, r2 = roles.get(sp[0]), roles.get(sp[1])
    if not r1 or not r2:
        return None
    win = _role_table(ol).get("+".join(sorted([r1, r2])))
    if not win:
        return None
    got = [s for s in sp if roles.get(s) == win]
    return got[0] if len(got) == 1 else None


def _triple_winner(speakers, roles, ol):
    """The one speaker whose role is neither primary host nor interviewee, or None (fail safe)."""
    ph = str(ol.get("primary_host_role", "primary_host"))
    iv = str(ol.get("interviewee_role", "interviewee"))
    sp = list(speakers)
    if len(sp) != 3:
        return None
    thirds = []
    for s in sp:
        role = roles.get(s)
        if not role:
            return None
        if role != ph and role != iv:
            thirds.append(s)
    return thirds[0] if len(thirds) == 1 else None


def _pure_laugh_spans(ctx, ol):
    markers = ctx.t.get("laugh_markers") or []
    limit = int(ol.get("max_laugh_chars", 40))
    min_span = float(ol.get("min_span_s", 0.15))
    merge_gap = float(ol.get("merge_gap_s", 0.25))
    max_speech = float(ol.get("max_speech_overlap_s", 0.0))
    require_ok = bool(ol.get("require_status_ok", True))
    ms = ol.get("min_word_score")
    if ms is None:
        ms = ctx.cfg.get("align", {}).get("min_word_score", 0.35)
    ms = float(ms)
    spans = []
    for s in ctx.spk:
        raw = []
        for g in ctx.words.get(s, []) or []:
            if require_ok and g.get("status") != "ok":
                continue
            if is_question(g.get("text")):
                continue
            words = [w for w in (g.get("words") or []) if w.get("start") is not None and w.get("end") is not None]
            if not words:
                if is_pure_laugh(g.get("text"), markers, limit) and float(g.get("score") or 0) >= ms:
                    a, b = float(g["start"]), float(g["end"])
                    if b - a >= min_span:
                        raw.append((a, b))
                continue
            laugh_w, speech_w = [], []
            for w in words:
                surf = w.get("surface") or ""
                pure = (not is_question(surf)) and is_pure_laugh(surf, markers, limit) and float(w.get("score") or 0) >= ms
                (laugh_w if pure else speech_w).append(w)
            for w in laugh_w:
                a, b = float(w["start"]), float(w["end"])
                if any(overlap(a, b, float(o["start"]), float(o["end"])) > max_speech for o in speech_w):
                    continue
                if b - a >= min_span:
                    raw.append((a, b))
        raw.sort()
        merged = []
        for a, b in raw:
            if merged and a <= merged[-1][1] + merge_gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            if b - a >= min_span:
                spans.append({"speaker": s, "start": a, "end": b})
    return spans


def _laugh_regions(spans, min_overlap):
    events = []
    for i, sp in enumerate(spans):
        events.append((sp["start"], 1, i))
        events.append((sp["end"], 0, i))
    events.sort(key=lambda e: (e[0], e[1]))
    active, prev_t, prev_sp, raw = set(), None, set(), []
    for t, kind, i in events:
        if prev_t is not None and t > prev_t and len(prev_sp) >= 2 and (t - prev_t) >= min_overlap - 1e-9:
            raw.append({"start": prev_t, "end": t, "speakers": set(prev_sp)})
        if kind == 0:
            active.discard(i)
        else:
            active.add(i)
        prev_t = t
        prev_sp = {spans[j]["speaker"] for j in active}
    merged = []
    for reg in raw:
        if merged and merged[-1]["speakers"] == reg["speakers"] and reg["start"] <= merged[-1]["end"] + 1e-9:
            merged[-1]["end"] = reg["end"]
        else:
            merged.append(reg)
    return merged


def _decide_laugh_region(ctx, reg, roles, ol):
    spks = reg["speakers"]
    n = len(spks)
    if ol.get("respect_edge_protect", True) and ctx.protected_edge(reg["start"], reg["end"]):
        return None, "edge"
    if n > int(ol.get("max_decided_speakers", 3)):
        return None, "too_many"
    if n == 3:
        w = _triple_winner(spks, roles, ol)
        return w, ("triple_keep" if w else "triple_review")
    if n == 2:
        w = _pair_winner(spks, roles, ol)
        return w, ("pair_keep" if w else "pair_review")
    return None, "too_many"


def _merge_laugh_pieces(chunks, gap):
    chunks = sorted(chunks, key=lambda z: (z[0], z[1]))
    out = []
    for a, b, why in chunks:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b, why])
    return out


def _laugh_action_why(base, action):
    if action == "review":
        return base if base in _LAUGH_WHY else "pair_review"
    if str(base).startswith("triple"):
        return "triple_keep" if action == "keep" else "triple_mute"
    if str(base).startswith("pair"):
        return "pair_keep" if action == "keep" else "pair_mute"
    return base


def _laugh_cand(ctx, track, t0, t1, action, reason, conf):
    ex = ctx.excerpt(track, t0, t1)
    con = ctx.context(t0, t1)
    return {"key": key("overlap_laughter", track, r(t0), r(t1), action, ex, con), "type": "overlap_laughter",
            "track": track, "start": r(t0), "end": r(t1), "default_action": action, "reason": reason,
            "excerpt": ex, "context": con, "confidence": r(conf, 2)}


def overlap_laughter(ctx):
    """Overlapping pure laughter keeps one laugh. Mute only a loser's laughter interval (never a timeline cut).

    Two speakers: no built-in winner (`two_speaker: review`, or `role_map` with an explicit pair entry).
    Three speakers at once: keep the unique role that is neither primary host nor interviewee; otherwise review.
    Speech, questions, ordinary backchannels, mixed speech/laughter, and ambiguous detection are not muted."""
    ol = ctx.c.get("overlap_laughter") or {}
    if not ol.get("enabled", False):
        return []
    spans = _pure_laugh_spans(ctx, ol)
    if len({sp["speaker"] for sp in spans}) < 2:
        return []
    regions = _laugh_regions(spans, float(ol.get("min_overlap_s", 0.2)))
    if not regions:
        return []
    roles = _clean_roles(ctx.roles)
    for reg in regions:
        reg["winner"], reg["why"] = _decide_laugh_region(ctx, reg, roles, ol)
    extent = ol.get("mute_extent", "overlap_only")
    if extent not in ("overlap_only", "full_span"):
        extent = "overlap_only"
    gap = float(ol.get("merge_gap_s", 0.25))
    floor = float(ol.get("min_span_s", 0.15))
    pieces = {}

    def add(spk, action, a, b, why):
        pieces.setdefault((spk, action), []).append((a, b, why))

    if extent == "full_span":
        for sp in spans:
            related = [reg for reg in regions if sp["speaker"] in reg["speakers"]
                       and overlap(sp["start"], sp["end"], reg["start"], reg["end"]) > 0]
            if not related:
                continue
            wins = [reg["winner"] == sp["speaker"] for reg in related]
            loses = [reg["winner"] is not None and reg["winner"] != sp["speaker"] for reg in related]
            if any(reg["winner"] is None for reg in related) or (any(wins) and any(loses)):
                why = "span_conflict" if any(wins) and any(loses) else related[0]["why"]
                add(sp["speaker"], "review", sp["start"], sp["end"], why)
            elif any(loses):
                add(sp["speaker"], "mute", sp["start"], sp["end"], related[0]["why"])
            elif any(wins):
                add(sp["speaker"], "keep", sp["start"], sp["end"], related[0]["why"])
    else:
        for reg in regions:
            for sp in spans:
                if sp["speaker"] not in reg["speakers"]:
                    continue
                a, b = max(sp["start"], reg["start"]), min(sp["end"], reg["end"])
                if b - a <= 1e-9:
                    continue
                if reg["winner"] is None:
                    act = "review"
                elif reg["winner"] == sp["speaker"]:
                    act = "keep"
                else:
                    act = "mute"
                add(sp["speaker"], act, a, b, reg["why"])
    out = []
    for (spk, action), chunks in pieces.items():
        for a, b, why in _merge_laugh_pieces(chunks, gap):
            if b - a < floor - 1e-9:
                continue
            if action not in ("mute", "keep", "review") or spk == TIMELINE:
                continue
            code = _laugh_action_why(why, action)
            conf = 0.45 if action == "review" else 0.86
            out.append(_laugh_cand(ctx, spk, a, b, action, _LAUGH_WHY[code], conf))
    return out


def speaker_roles(ep, cfg):
    """Episode manifest only. Missing or non-dict map -> no roles (triple rule reviews)."""
    p = Path(ep) / "manifest.json"
    if not p.exists():
        return {}
    man = load(p)
    if not isinstance(man, dict):
        return {}
    key_name = ((cfg.get("candidates") or {}).get("overlap_laughter") or {}).get("roles_manifest_key", "speaker_roles")
    roles = man.get(key_name) or {}
    return roles if isinstance(roles, dict) else {}


def build(vad, words, cfg, roles=None):
    ctx = Ctx(vad, words, cfg, roles)
    my = mutual_yield(ctx); ors0 = overlap_restart(ctx)
    ab = [c for c in overlap_abandon(ctx) if not any(o["track"] == c["track"] and overlap(o["start"], o["end"], c["start"], c["end"]) > 0 for o in ors0)]
    cs = (long_pauses(ctx) + fillers(ctx) + backchannels(ctx) + overlap_yield(ctx) + ors0 + ab + false_starts(ctx)
          + crosstalk(ctx) + mistakes(ctx) + overlap_laughter(ctx))
    # a mutual_yield cut removes the whole collision: drop the smaller edits inside it (incl. overlap_yield, so
    # keep_overlap_yield cannot keep it)
    myc = [m for m in my if m["default_action"] == "cut"]
    cs = my + [c for c in cs if not any(overlap(m["start"], m["end"], c["start"], c["end"]) > 0 and (
        c["type"] in ("overlap_yield", "overlap_restart", "overlap_abandon", "backchannel", "filler")
        or (c["start"] >= m["start"] and c["end"] <= m["end"])) for m in myc)]
    # an aborted phrase detected as overlap_restart is decided (auto mute): drop the overlap_yield on the same
    # track/span so keep_overlap_yield cannot auto-keep it
    ors = [c for c in cs if c["type"] == "overlap_restart"]
    cs = [c for c in cs if not (c["type"] == "overlap_yield" and any(
        o["track"] == c["track"] and overlap(o["start"], o["end"], c["start"], c["end"]) > 0 for o in ors))]
    seen, out = set(), []
    for c in sorted(cs, key=lambda c: (c["start"], c["type"], c["track"], c["end"])):
        if c["key"] not in seen:
            seen.add(c["key"]); out.append(c)
    return auto_decide(out, cfg["candidates"].get("auto_decide", {}))


def auto_decide(cs, ad):
    """Resolve low-information review items to `keep` (no edit) so the human list stays short.
    Only ever review -> keep: an auto-decision can never cut meaning or a laugh.
    overlap_laughter is never rewritten (a review stays a review; a mute stays a mute)."""
    if not ad.get("enabled", False):
        return cs
    for c in cs:
        if c["default_action"] != "review" or c.get("type") == "overlap_laughter":
            continue
        r, why = c["reason"], None
        if c["type"] == "overlap_yield" and ad.get("keep_overlap_yield", True):
            why = "overlap left as recorded"
        elif "could be laugh" in r:
            why = "untranscribed sound, possibly laugh"
        elif "alignment score low" in r and c["confidence"] < ad.get("min_conf", 0.5):
            why = "weak alignment, not confident"
        elif c["type"] == "false_start" and "parallel structure" in r:
            why = "repetition with words between, likely rhetorical"
        elif c["type"] == "long_pause" and "punchline" in r:
            why = "pause before laugh/reaction kept for timing"
        if why:
            c["default_action"] = "keep"; c["reason"] = f"{r} [auto-keep: {why}]"
    return cs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ep", required=True); a = ap.parse_args()
    ep = Path(a.ep); cfg = load_cfg(ep)
    cs = build(load(ep / "work/vad.json"), load(ep / "work/words.json"), cfg, speaker_roles(ep, cfg))
    dump(ep / "work/candidates.json", cs)
    stats = {}
    for c in cs:
        stats.setdefault(c["type"], {}).setdefault(c["default_action"], 0)
        stats[c["type"]][c["default_action"]] += 1
    log_step(ep, "candidates", stats)
    print("candidates", stats)


if __name__ == "__main__":
    main()
