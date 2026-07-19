# Open Video KG on RunPod Design

## Goal

`runpod.etzhayyim.com` を、公開取得可能な動画ソースを対象にしたマルチモーダル抽出基盤へ拡張し、動画・音声・字幕から evidence-first で knowledge graph を生成する。

前提:

- 最初の対象は **コンプライアンス上オープンに取得できるサイトのみ**
- DRM 回避、ログイン必須領域の取得、利用規約違反の大量収集は対象外
- 収集結果は既存の `collector -> resources -> entity -> resourceflow -> intel` 系に接続する
- RunPod は GPU が必要な重い処理に限定し、サイト巡回や権限制御は別 actor/worker に分離する

## Current Repo Status

現時点の repo 上の確認結果:

- `YouTube` は専用設計あり
  - `60-apps/etzhayyim-project-youtube/CLAUDE.md`
  - `60-apps/etzhayyim-project-youtube/wit/youtube/package.wit`
- `TikTok / Baidu / Douyin / Bilibili` は同等レベルの actor/project は未確認
- `RunPod` は現状 `Ollama` を載せた推論 gateway が中心
  - `60-apps/etzhayyim-project-runpod/CLAUDE.md`
  - `60-apps/etzhayyim-project-runpod/serve/handler.py`
  - `60-apps/etzhayyim-project-runpod/serve/worker-gateway.ts`

従って、現状は:

- **YouTube は設計済み**
- **TikTok / Baidu 系の動画収集 actor は未設計**
- **動画/音声 -> KG の RunPod 設計はこれから追加する段階**

## Compliance-First Scope

Phase 1 で対象にするのは次のようなソース。

1. 公式 API か公開メタデータ取得手段がある
2. robots / terms / rate limit を守れる
3. ログイン不要で取得できる
4. DRM 保護コンテンツを含まない
5. 取得目的と保存粒度を説明できる

推奨対象:

- YouTube 公開チャンネル
  - 優先取得: metadata, captions, thumbnails, comments
  - 本体動画の保存は原則しない。必要時のみ短期キャッシュ
- Vimeo 公開動画
- Dailymotion 公開動画
- Internet Archive 動画
- 各国政府・大学・研究機関・企業 IR の公開動画ページ
- 自社保有またはライセンス確認済みの Blob/Object Storage 上の動画

Phase 1 で除外:

- TikTok / Douyin の大規模クロール
- Baidu 系動画サービスの無差別収集
- 会員限定 / 地域制限 / DRM 付き配信
- 利用規約上グレーなダウンロード

## System Boundary

役割分離:

- Site actor
  - 対象 URL 発見
  - allow / deny 判定
  - robots / terms / domain policy 参照
  - metadata/caption URL の取得
- Fetch actor
  - HTML, JSON, caption, manifest の取得
  - 必要最小限の media segment 取得
- RunPod multimedia workers
  - 音声抽出
  - ASR
  - diarization
  - OCR
  - scene segmentation
  - embedding / triplet extraction
- Graph ingest actor
  - evidence 正規化
  - entity/relation/resourceflow への書き込み

RunPod に入れないもの:

- 収集可否の法務判断
- robots / terms の最終判定
- 長期 state orchestration
- 大量の HTML クロール制御

## Target Pipeline

```text
Domain discovery / seed
  -> compliance policy check
  -> page fetch / API fetch
  -> video manifest + metadata + captions discovery
  -> EvidenceCreated(video-raw)
  -> RunPod preprocess job
      -> ffmpeg demux
      -> keyframe sampling
      -> audio extraction
      -> caption normalization
  -> RunPod understanding job
      -> ASR
      -> diarization
      -> OCR
      -> scene/topic segmentation
      -> entity/relation/event extraction
  -> EvidenceNormalized
  -> collector_evidence_current
  -> entity/resource/resourceflow/intel projection
  -> graph.write
```

## Canonical Evidence Model

最低限の evidence 単位を分ける。

- `VideoSource`
  - source URL, canonical URL, platform, license hint, fetched_at
- `VideoAsset`
  - duration, fps, width, height, codec, language hints
- `VideoCaptionTrack`
  - source=`publisher|platform|asr`
  - lang, confidence, segments[]
- `VideoAudioSegment`
  - start_ms, end_ms, speaker_id?, transcript, confidence
- `VideoScene`
  - scene_id, start_ms, end_ms, keyframe_ref, OCR text, visual summary
- `VideoClaim`
  - subject, predicate, object / event / temporal anchor
- `VideoEntityMention`
  - entity text span, source modality, offsets, confidence
- `VideoProcessingJob`
  - model versions, job id, gpu type, runtime_sec, cost_estimate

原則:

- 元データと派生データを分離
- publisher captions を ASR より優先
- relation は evidence への参照付きで保存
- 低信頼抽出は graph に直接昇格せず evidence に留める

## Knowledge Graph Projection

既存の `crawler -> collector -> entity -> resourceflow -> intel` の流れに合わせ、動画由来データを以下へ投影する。

- `collector`
  - 動画、字幕、音声セグメント、OCR、コメントを evidence として保持
- `entity`
  - 人物、組織、製品、地名、番組、チャンネル、トピック
- `resource`
  - 再利用価値の高い transcript, summary, chapter, thumbnail-set
- `resourceflow`
  - `video -> scene -> transcript chunk -> extracted entity/relation`
- `intel`
  - anomaly, risk, trend, narrative shift, mention spike

Graph へ昇格する relation 例:

- `CHANNEL PUBLISHED VIDEO`
- `VIDEO MENTIONS ENTITY`
- `SPEAKER SPOKE_IN SEGMENT`
- `SEGMENT ASSERTS CLAIM`
- `SCENE CONTAINS OCR_TEXT`
- `VIDEO DERIVED_FROM SOURCE_URL`

## RunPod Service Split

現行 `serve/handler.py` は LLM 推論向け単体 worker なので、動画系は serverless endpoint を分離する。

推奨 endpoint:

1. `runpod-video-preprocess`
   - ffmpeg
   - scene cut
   - thumbnail extraction
   - audio demux
2. `runpod-audio-asr`
   - Whisper / faster-whisper
   - VAD
   - optional diarization
3. `runpod-video-understanding`
   - OCR
   - image/video captioning
   - frame embedding
4. `runpod-kg-extract`
   - transcript + OCR + metadata から entity/relation/event 抽出
   - JSON schema strict output
5. `runpod-llm-gateway`
   - 現行の `Ollama` 系 gateway

理由:

- 動画前処理と LLM 会話推論では GPU / memory profile が違う
- cold start コストを用途別に分離できる
- 障害点を局所化できる

## Model Strategy

Phase 1 推奨:

- 音声認識
  - `faster-whisper large-v3` か同等
- diarization
  - `pyannote` 系 or 同等
- OCR
  - PaddleOCR / EasyOCR / Surya 系
- vision summary
  - 軽量 VLM を keyframe 単位に適用
- KG extraction
  - schema constrained LLM
  - relation taxonomy は限定列挙

重要方針:

- いきなり end-to-end で動画丸ごと LLM に入れない
- まず字幕/OCR/metadata を構造化し、その後 relation 抽出
- コスト最適化のため、LLM は scene summary と transcript chunk に対して使う

## Job Contract

RunPod に渡す job input 例:

```json
{
  "job_type": "video_kg_extract",
  "source": {
    "platform": "youtube",
    "canonical_url": "https://www.youtube.com/watch?v=...",
    "license": "public"
  },
  "media": {
    "video_url": "signed-url-or-cache-ref",
    "caption_tracks": [
      { "lang": "ja", "url": "..." }
    ]
  },
  "processing": {
    "extract_audio": true,
    "run_asr": true,
    "run_ocr": true,
    "sample_fps": 0.25,
    "chunk_sec": 30
  },
  "output_schema_version": "video-kg-v1"
}
```

返却例:

```json
{
  "status": "completed",
  "artifacts": {
    "transcript_json": "s3://...",
    "scenes_json": "s3://...",
    "claims_json": "s3://..."
  },
  "entities": [],
  "relations": [],
  "metrics": {
    "duration_sec": 812,
    "gpu_sec": 96.2,
    "asr_avg_confidence": 0.91
  }
}
```

## Domain Policy Registry

サイト別に静的ポリシーを持つ。

- `platform`
- `collection_mode`
  - `api_only | metadata_only | captions_only | public_page_only | owned_media`
- `store_media_bytes`
  - `never | transient | allowed`
- `allowed_artifacts`
  - `metadata, captions, comments, thumbnails, transcript-derived-kg`
- `requires_manual_review`
- `terms_version`
- `last_reviewed_at`

初期値:

- `youtube`: `metadata_only + captions_only + public_page_only`, media bytes は `transient`
- `vimeo`: `public_page_only`
- `internet-archive`: `owned/open-license`, media bytes `allowed`
- `tiktok`: `requires_manual_review=true`
- `baidu-video`: `requires_manual_review=true`

## Storage Strategy

保存先の分離:

- hot artifacts
  - object storage
  - 7-30 日 TTL
- canonical evidence
  - collector tables / records
- promoted graph facts
  - entity/resource/resourceflow/intel

保存しないもの:

- フル動画の恒久保存
- 再取得可能な巨大バイナリの重複保持

## Reliability

必須要件:

- idempotency key = `platform + canonical_video_id + processing_profile + model_versions`
- 再実行時は artifact cache を優先
- scene / transcript chunk 単位で partial retry
- 失敗理由を `VideoProcessingJob` に残す

監視指標:

- fetch success rate
- caption availability rate
- ASR fallback rate
- GPU seconds / video minute
- entity promotion precision
- relation acceptance rate

## Recommended Delivery Plan

### Phase 0

- `YouTube` 公開動画だけに限定
- 本体動画保存なし
- metadata + captions + comments + thumbnail + channel graph
- KG は transcript/caption ベース

### Phase 1

- RunPod `audio-asr` endpoint 追加
- 字幕なし動画のみ ASR 実行
- scene / OCR を追加

### Phase 2

- Vimeo / Dailymotion / Internet Archive を追加
- site policy registry 導入
- evidence lineage を `resourceflow` に接続

### Phase 3

- review queue 付きで TikTok / Baidu 系を個別審査
- 法務・terms review を通った collection mode のみ解放

## Concrete Next Steps

1. `youtube.etzhayyim.com` の public crawl を Phase 0 対象として固定
2. `runpod-video-preprocess` と `runpod-audio-asr` を新 endpoint として分離
3. `video-kg-v1` JSON schema を定義
4. `collector_evidence_current` に動画 evidence ingest 口を追加
5. `resourceflow` に `video -> scene -> claim -> entity` lineage を追加
6. `tiktok` と `baidu` は actor を作る前に policy registry と manual review を実装

## Decision

結論として、今すぐ着手すべき順序は次。

1. `YouTube` 公開動画の metadata / captions / comments 中心の収集
2. 字幕不足分だけ RunPod で ASR
3. transcript/OCR/scene から evidence-first に KG 化
4. `TikTok / Baidu` は compliance policy が固まるまで actor 化しない
