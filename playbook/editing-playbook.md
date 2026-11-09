# 編集プレイブック（エンドツーエンド）

パラメータの唯一の正は [config.yaml](../config.yaml)。本書・スキルに数値を書く場合は config のキー名を併記し、食い違えば config が優先。実行は `run_pipeline.sh <ep> prep …` → レビュー → `run_pipeline.sh <ep> finish`。どのステップも失敗した時点で停止する。

## 0. 素材（Zencastr）
1. Library → エピソードの ⋮ → View Tracks → 話者別 MP3 をダウンロード（ミックスではなく話者別）。文字起こしがあれば TXT。
2. ingest は元ファイルを **コピー**（移動しない）して `raw/` に置く。元素材は消さない。

## 1. 前処理（pipeline/ingest.py, sync.py, clean_*.py, level.py）
- デコード: `aresample=resampler=soxr` で 48 kHz / 24bit / mono（`ingest.*`）。LAME/Xing タグのエンコーダ遅延・パディングを読み、ギャップレス長を検証。開始オフセットを `logs/probe.json` に記録し `ingest.max_start_offset_s` 超で停止。トラック長差は `ingest.max_length_diff_s`。ffmpeg / libmp3lame のバージョンは `env.ffmpeg_version` / `env.lame_version` と一致しなければ停止。
- 同期: 最長トラックを基準に、10 ms ログエネルギー包絡（かぶり音）の相互相関。ピーク信頼度 `sync.min_peak_confidence` 未満なら補正しない。オフセット `sync.min_offset_ms`、ドリフト `sync.max_drift_ppm`（ppm）超のみ補正。
- 整音: DeepFilterNet3（`clean.dfn_atten_lim_db`）→ pedalboard（HPF → de-ess → Compressor → Limiter, `clean.*`）。afftdn は使わない。
- レベラー: 窓 `level.window_s`、ゲート `level.gate_dbfs`、最大ゲイン ±`level.max_gain_db`、平滑 `level.smoothing_s`、位置はダイナミクスの後・VAD の前。その後 `level.target_lufs` に静的ゲイン。

## 2. VAD・文字起こし・アライメント
- VAD は Silero のみ（`vad.*` 固定）。無音判定は **話者別 VAD の和集合** の一つだけ。
- 文字起こし: Zencastr の TSV（話者・発話開始時刻）を一次ソースとし、`zencastr_transcript.py` で話者トラックに割り当て、各発話を MMS_FA で整列して単語時刻を得る。kotoba ASR は既定で使わない（`asr.fallback: gaps` で未カバー区間のみ）。Zencastr からは毎回 TXT と CSV の両方を書き出す。CSV が無い場合のみ全編 ASR。
- ASR（フォールバック／CSV なし時）: kotoba-whisper-v2.0（HF revision 固定, float32, beam 5, temperature `[0.0]` のみ）。フォールバック相当（低 logprob・高圧縮率）は `logs/asr_fallbacks.json` に記録。
- 漢字→かな: fugashi + unidic-lite で分かち書き・品詞、読みは pyopenjtalk（文単位）→ uroman → MMS_FA。セグメントスコア `align.min_segment_score` 未満は `low` とし、その上の候補は review。

## 3. 編集種別（`work/candidates.json` → `labels.json` → `work/edits.json`）
`edits.json` は唯一の編集リスト: `{id, key, type, track, start, end, action(cut|duck|mute|keep|review), approved, labeler, reason, excerpt, confidence, note}`（raw 秒）。`approved: true` かつ action が review 以外のものだけが適用される。

| type | 検出 | 処理 |
|---|---|---|
| `long_pause`（長すぎる沈黙と間） | 話者別 VAD 和集合のギャップ。同一話者内 > `candidates.long_pause.within_turn_min_s`(1.2 s)、話者交代 > `between_speakers_min_s`(1.5 s) | 中央を切り `target_s`(0.7 s) を残す（タイムライン cut、ルームトーン＋クロスフェード）。直後 `laugh_lookahead_s`(2 s) 以内に笑い/反応があれば **review**（オチ前・意図的な間は詰めない） |
| `filler` | 形態素の品詞が `text.filler_pos`（フィラー/感動詞）かつ `text.fillers` に含まれ、前後に間がある。音響: テキストの無い短い VAD 島・アライナ棄却区間 | 他者無発話なら cut、他者発話中なら自トラック mute。低スコア・保護区間・音響のみは review |
| `backchannel` | `text.backchannels` のみの短い発話が他者発話中 | 10 s に 1 つ・笑いへの反応は keep、他は自トラックのみ `render.duck_db`(-20 dB) duck。タイムラインは切らない |
| `overlap_yield` | 発話開始差 ≤ `overlap_onset_s`、譲った側は直前 `yield_pre_silence_s` 無音（ターン開始）かつ島 ≤ `yield_max_s`、相手が `continue_window_s` 内に ≥ `continue_min_s` 発話 | 直後に譲り表現（`text.yield_markers`）があれば **譲った側の言いかけだけ** を自トラックで mute（ルームトーン）。無ければ review。継続話者は無傷 |
| `false_start` | かな n-gram の 2 語以内の繰り返し | 他者無発話なら cut 候補、それ以外は review |
| `crosstalk` | 2 名以上 ≥ `crosstalk_min_s` 重なり | review |
| `mistake` | `text.mistake_markers`（「ここカットで」等） | review |
| `overlap_laughter` | 純然たる笑いの重なり（`candidates.overlap_laughter`） | 3人同時は interviewee でも primary host でもない唯一の話者を残し、他の笑い区間だけ自トラック mute。2人は `two_speaker` が review なら人間。タイムラインは切らない |

自動承認は `labels.auto_approve_types` のみ。他は agent/人間のラベルが必要。

## 4. 絶対にカットしないもの
意味が変わるカット／ジョーク・笑い・その反応／冒頭・末尾 `candidates.protect_head_tail_s` 秒の内容（該当候補は review）。ブレス処理は行わない（既知の課題）。重なった純然たる笑いだけは `overlap_laughter` に従い、負けた話者の笑い区間を自トラックで mute できる（タイムラインは切らない）。

## 5. ラベルとレビュー
- `labels.json` はキー = hash(候補 + 前後文脈)。再実行で同じ候補には同じラベルが再利用され、`seed` は既存ラベルを上書きしない。未ラベル候補があると render は停止。
- `review_export.py` → Audacity ラベル（`work/review_labels.txt`）→ 先頭に `OK`/`NG`/`cut`/`duck`/`mute`/`keep` を付けて `review_import.py`。
- 判断が数値化できないもの（意味・笑い・意図的な間）は review に回し、文章判断で自動適用しない。

## 5b. スピーチゲート（pipeline/gate.py、config `gate`）
- クリーニング・レベル調整後、各トラックを**自分の発話区間以外ではミュート**（`attenuation_db`、既定 -inf、例 -40）。発話区間 = Silero VAD ∪ 笑い ∪ 相槌（これらは決してゲートしない）。文字起こしの単語スパンは**自分の VAD 島と重なる部分だけ**を発話とみなし、VAD を超えたアライメントのにじみは捨てる（20261026 の 0:15 の鼻すすりは 6.7 秒ににじんだ単語に守られていた）。前 150 ms / 後 250 ms のパッドは単語端ではなく **VAD の発話端** 基準、300 ms 未満の隙間は結合、10–20 ms フェード。
- 過渡音（鼻すすり・咳・咳払い）の判定は **低い周期性（voicing < `voicing_max`）かつ ノイズ的な高域**（2 kHz 以上のエネルギー比 > `hb_share_min` または重心 > `centroid_min_hz`）の両方が必要。息っぽい/かすれた語尾（20261026 の「ね」の減衰は低域）は過渡音にしない。有声部から単調に下がり続ける区間（有声の減衰）はミュートしない。単語端と一致する VAD 終端の後 `word_end_keep_s` は保護。
- VAD 島の中も長さ・文字に関係なく 100 ms 窓で走査し、前後 `surround_s` に有声窓がない高域・無声窓はミュート（語中の「さ」などの摩擦音は有声に挟まれるので残る）。無声窓が `island_unvoiced_frac` 以上の島では単語を確かな支えとみなさず、`island_sniff_max_s` 以下なら島ごとミュート（20261026 の ASR 幻覚「ありがとうございました。」の上の鼻すすり）。
- 咳・咳払い（帯域に依存しない、鼻すすり判定とは別ルール `burst_*`）：確かな文字も単語トークンもない ≤ 0.6 s の島で、ノイズフロア付近（余裕 = clamp(自トラック非発話 p90 − フロア, 10, 15) dB）から 40 ms 以内に自トラック SNR × 0.4 以上立ち上がり（トラックごとに自動計算、logs/gate.json `burst_thresholds`。20261026 では host 35 dB / guest_a 26.4 dB / guest_b 30.4 dB）、ピッチが不安定（フレーム有声度中央値 < 0.5 または f0 跳び > 15%）、島全体の有声度 < 0.6（「んー」等の安定したハミングは残す）、かつ自分の有声発話終わりから ≥ 0.2 s（減衰は残す）ならミュート。20261026 の 5:20 付近の低域の咳払い（重心 約250 Hz で高域の鼻すすり判定には掛からない）が対象。
- 確かな文字（単語の半分以上がその島の中、かつ島が主に有声）がない短い VAD 島（≤ `transient_max_s`）で無声（自己相関ピーク < `voicing_max`）またはスペクトル平坦（> `flatness_min`）なもの＝鼻すすり・咳・咳払いは VAD が反応してもゲート（笑い・相槌は除く）。
- 発話区間内の過渡音検出：パッド部・結合した隙間など有声 VAD に支えられていない部分は 10 ms フレームでバースト（ノイズフロア + `transient_rise_db`）を探し、サブ窓ごとに分類して無声ならミュート。有声 VAD 島の頭/尻の無声連続 ≥ `edge_tail_min_s`（VAD に貼り付いた鼻すすり・息）もミュートし、発話側 `edge_keep_s` は残す（語末の「す」などを守る）。笑い・相槌とそのパッドは対象外。
- 全トラックがゲートされる区間は `fill_track` のルームトーンで埋め（ルームトーン窓は VAD なし かつ **バーストなし** の最も静かな窓。採取位置を `logs/gate.json` の `roomtone_source` に記録）、ミックスがデジタル無音にならないようにする。QA 指標：非発話区間のバースト数（ゲート前/後、`logs/gate.json`）。

## 6. レンダリング（pipeline/render.py）
- タイムライン cut は全トラック同時、ミックスの最小エネルギー 5 ms 点へ ±`render.snap_window_ms` でスナップ、`render.crossfade_ms` の等パワークロスフェード（全トラック同一）。
- duck/mute はトラック内のみ（`render.duck_fade_ms`）。mute はルームトーン置換。
- ルームトーン: 各トラックの VAD 外で最も静かな `render.roomtone_s` 秒窓をクロスフェードでループ。
- `work/timemap.json`（raw→編集後）を保存。出力はモノラルミックスを L/R 同一にした **デュアルモノ**。

## 6a. ジングル（pipeline/jingle.py、render の後・ending_bgm の前）
- 1エピソードに **1回だけ**。音源はリポジトリ外（`jingle.file`）。コミットしない。
- 位置: チャプター境界（テーマ転換）の間を優先し、エピソード中央に最も近いチャプターから探す。無ければ最も自然な間（文末・話者交代・最長の無音）。
- 中央優先: まず編集後の尺の 30〜70%（`prefer_range_frac`）で探し、無ければ `widen_step_frac`（10%）ずつ左右に広げる。同じ範囲内ではチャプター境界 > 自然な間。終盤のチャプター直前（話題の締め）は不自然になりやすい。
- 除外: 冒頭2分（`min_start_s`）、終盤（`exclude_end_s`、エンディング BGM の探索範囲以上）、文の途中、質問の直後（答えと切り離さない）、相槌・返事の直前、被りの近く（`avoid_margin_s`）。
- 笑い・相槌は避けずに **待つ**: 文末の後に続く笑い・相槌・短い反応（どのトラックでも）が終わり、全トラックが静かになった所に入れる（最大 `max_tail_s` 4秒）。その前に本題の発話が始まったら候補から外す。何も削らない。
- 流れ: 直前の発話終わり → 無音 0.4 秒 → ジングル（声の実効レベルに合わせ `gain_db`）→ 無音 0.2 秒 → 再開。時間を足すだけで発話は削らない。
- timemap をずらすので、チャプター・文字起こし・エンディング BGM・QA は自動で追従。QA はジングルの無音を長い間として数えない。
- トラックシフト（`track_shift`）: チャプター境界で次の話者が他の人の笑い・相槌・文末に被せて話し始めている場合、**その話者のトラックだけ**を発話開始直前（`split_pad_s`、無音の中）で切り、他のトラックは尾を最後まで鳴らす → 全員静かになったら 0.4 秒 → ジングル → 0.2 秒 → 切った所から全トラック再開。何も削らない。待ちは最大 `max_shift_s`（3 秒）。被せた発話が他の人の本題の発話と被る（`max_tail_overlap_s` 超）なら不可。トラック別ステム（work/render）から作り直すので mono ミックスでは行わない。チャプター境界扱いなので普通の間より優先。
- 再実行しても二重に入らない（`work/mix_prejingle.wav` から戻す）。`--dry-run` で位置だけ確認できる。

## 6b. エンディング BGM（pipeline/ending_bgm.py、jingle の後・loudness の前）
- ミッドショーのジングルではない。`jingle.file` には置かない。音源もエピソードごとのアンカーもリポジトリに入れない。
- `mode: phrase_tail`（`mode` が無い古いエピソード config）: 末尾 `search_window_s` 以内の最後の締めフレーズ（`phrases`）の `lead_in_s` 前からフェードイン。声の下は `bed_db`。最後の発話終了後 `ramp_s` で `tail_db` へ上げ、発話終了から `tail_s` を BGM のみ、`fade_out_s` で dB 直線に `fade_out_floor_db` まで落とし、さらに `end_silence_s`。この包絡より長い音楽はそこで止まる。フレーズが無いときはスキップ。
- `mode: full_asset`（リポジトリ config の既定）: 音楽は `bgm_offset_s` から EOF まで 1 回。ループしない。`tail_s` / `fade_out_s` では切らない。`tail_fade_ms: 0` なら自然な末尾をホールドしたまま残す。`outro_file` は別資産で、同じく末尾まで 1 回鳴らし、終わりを音楽の終わりに揃える。両方の開始時刻を明示して終わりがずれるときは切らず、`ends_aligned: false`。締めフレーズは見ない。
- 終端アンカーは編集後タイムライン。`speech_end` = 最後の有効 VAD 終端（最終発話。ミックス末尾の余白より前になり得る）。`episode_end` = 延長前のボイスミックス長（現在のエピソード終端。最終発話の後の無音を含み得る）。`explicit` = エピソード config の `end_anchor_s`。フル尺が `start_floor_s` より前からはみ出すときは両方を同じだけ後ろへずらし、サンプルは捨てない（`end_anchor_met: false`、`anchor_shift_s`）。
- 発話中は `bed_db`。`gain_speech_end_s`（null ならその VAD 終端）のあと `tail_db` まで上げ、ペイロードの最後まで保持する。この時刻はファイル終端ではない。`end_silence_s` はペイロードの後のデジタル無音。QA の末尾判定は `tail_end_s` まで。
- 無効・音源なしは警告してスキップ。再実行しても重ならない（`work/mix_voice.wav` から作り直す）。
- QA：BGM 区間内のつなぎ目は、局所 |diff| の 99.9 パーセンタイル比（`qa.click_ratio_bgm` 2.5）でクリック判定（音楽のアタックを誤検出しない）。

## 7. ラウドネス（pipeline/loudness.py：唯一のラウドネスツール）
- ffmpeg loudnorm 2 パス、`linear=true`。線形にならなければ停止。予測 TP が `loudness.max_true_peak_dbtp - limiter_headroom_db` を超える場合のみ事前に alimiter。
- 目標 **-16 LUFS ±`qa.lufs_tolerance`、TP ≤ -1.0 dBTP**（納品物の MP3 を QA で測定）。

## 8. チャプター（pipeline/chapters.py）
- 再 ASR はしない。raw 文字起こしからタイトルと raw 時刻を `chapters_raw.json` に書き、timemap で編集後時刻へ変換。
- `out/chapters.json`（Podcasting 2.0）と MP3 の ID3 CHAP/CTOC を書く。`chapters.min_gap_s` 未満の間隔は停止。

## 9. QA（pipeline/qa.py）
- MP3 の LUFS・TP（クリック検査は MP3 を一時 WAV にデコードして実施、終了後削除）、先頭/末尾無音 ≤ `qa.edge_max_silence_s`（0.5 s）、クリック = ±2 ms の最大サンプル差 / 周辺 ±50 ms の中央値 > `qa.click_ratio` かつ > `qa.click_min_abs`、レベル跳び > `qa.level_jump_db`、冒頭/末尾無音、long_pause の件数・削減秒・最長残存ギャップ。`logs/qa.md` に記録、失敗で exit 1。最後に人間が通し試聴。

## 10. 納品物（リポジトリ外）
- **MP3 のみ納品**：`out/<slug>.mp3`（192 kbps CBR / 44.1 kHz stereo）と `out/chapters.json`。WAV マスターは作らない。レンダー中の一時 WAV（normalized.tmp / prelimited / mix / render/*.wav）はエンコード後に削除（`output.delete_render_wavs`）。

## 構造編集（常時適用）
- **opening_chatter**：本題開始前の雑談はカット。Zencastr 文字起こしで挨拶後に最初に番組/テーマを切り出す行（`structure.opening_chatter.start_markers`、例「今日は」「周年」「始めた」）を探し、その発話の VAD 立ち上がり − `lead_s` まで削除。外れる場合は manifest.json の `opening_chatter.start_text` に正確な行を書いて固定。カット点は前後の行を引用して報告する。
- **edge_trim**：先頭/末尾の無音は `structure.edge_silence_s`（0.3 s）まで詰める。
- ショーノート（概要・トピック・リンク・mm:ss チャプター）。リンク検索は任意で、確認していないものは **未確認** と明記。
- 種別ごとの件数と削減秒、未適用の review 一覧。アップロード・公開・送信はしない。

## リポジトリ運用（絶対ルール）
- スキル・ポリシー・config・playbook の変更はすべてこのリポジトリにコミットし、**必ず GitHub に反映する**。コミット後に sha を報告し、push 権限/承認がある環境で push する（ローカルだけに残さない）。


### overlap_restart（被って言い直し）
話者 X の短い発話（≤ `candidates.overlap_restart.max_s` = 2.0 s）に別の話者が `overlap_onset_s`（0.5 s）以内に話し始めて被り、X が `restart_window_s`（2.5 s）以内に、読み正規化した内容語（`light_words` 以外）を 1 つ以上共有するフレーズで言い直した場合、X の言いかけを **X のトラックだけ自動ミュート**。タイムラインは切らず、相手の発話は残す。auto-approve 対象で、同じ区間の overlap_yield（keep_overlap_yield で自動 keep）は出さない。後に残る間は既存の long_pause が詰める。回帰: 20261026 host 151.43–153.15「なんか最近言われてるのは」→ 154.88「最近ちょろっと聴いたのは」（guest_a 151.62–153.92 は残す）。

**割り込み型**（overlap_onset 不一致でも可）：X のフレーズが、相手の内容のある発話（`interject_lead_s` = 0.5 s 以上前に開始）の最中に始まり、上記と同じ長さ・言い直し・共有内容語の条件を満たす場合も X のトラックだけミュート。ただし `interject_end_re`（か / ? / 。！）で終わるフレーズ（「2022年か」「前半?後半?」「すごいね。」）は完結した反応なので除外。回帰: 20261026 host 1624.51「ハードルね」→ 1627.91「01のハードルが…」（v8 26:19）。

### overlap_abandon（言いかけて諦め）
X の短い発話（≤ `candidates.overlap_abandon.max_s` = 2.0 s、相槌・フィラー・笑いでない）が相手 Y の進行中の内容発話の中にあり、Y が X の終了後 `continue_after_s`（1.0 s）以上話し続け、X が `silent_after_s`（3.0 s）沈黙し、`sentence_end_re`（。?か・です/ます/ね/よ/た/る 等）で終わらない場合、X のトラックだけ自動ミュート。auto-approve 対象。

### overlap_laughter（重なった笑いを一人だけ残す）
純然たる笑い（`text.laugh_markers` だけでできた区間。繰り返し可）が `min_overlap_s` 以上重なったとき、残すのは一人。負けた話者の笑い区間だけを自トラックで mute し、タイムラインは切らない。パラメータは `candidates.overlap_laughter`。

- 了解できる発話、質問、通常の相槌、発話と笑いが混ざった区間は、このルールでは無音にしない。音響だけの区間・低スコア・質問符を含む区間も自動 mute しない。
- 役割名は generic（`primary_host_role` / `interviewee_role` / それ以外）。トラックへの対応はエピソード `manifest.json` の `speaker_roles`（キー名は `roles_manifest_key`）だけ。
- 2人: リポジトリはどちらを残すか決めない。既定 `two_speaker: review`。`two_speaker: role_map` かつ `two_speaker_role_winners` にその役割ペアがあるときだけ自動。無ければレビュー。
- 3人が同時: interviewee でも primary host でもない話者がちょうど一人ならその笑いを残し、他の二人の笑い区間だけ mute。一意でなければ mute しない。4人以上（`max_decided_speakers`）もレビュー。時間的に連鎖しているだけのペアはトリプルにしない。
- 保存済みの `labels.json` は `seed` が上書きしない。検出を回さずに足す場合はエピソードの `laughter_edits.json`（リポジトリに置かない）:

```json
[{"track": "guest_b", "start": 12.0, "end": 13.5, "action": "mute", "kind": "laughter", "approved": true, "note": "pure laugh interval of the losing speaker"}]
```

`action` は mute、`kind` は laughter、`track` は一人（`all` 不可）。`approved: true` のときだけレンダーに乗る。`python pipeline/label_store.py --ep <ep> build` は既存ラベルを書き換えない。

### mutual_yield（お見合い・譲り合い）
2 人の立ち上がりが `overlap_onset_s`（0.5 s）以内、両者とも `yield_max_s`（1.5 s）以内に止まり、片方が `mutual_yield.markers`（どうぞ/お先に）を `yield_marker_window_s`（3 s）以内に言い、もう片方が `restart_window_s`（3 s）以内に話し直した場合、最初の立ち上がりから言い直しの `gap_s`（0.4 s）手前までを **タイムラインカット**（衝突・どうぞ・間の相槌ごと除去）。keep_overlap_yield の対象外で auto-approve。第三者が区間内で話す／頭尾・笑い付近は review。回帰: 20261026 host 1609.06「私なんか」/ guest_a「まあでもなんか」→ host「どうぞ」→ guest_a 1612.26 再開（v8 26:03）。
