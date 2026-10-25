# STT and OCR integration

CareFlow's web audio intake calls the in-repo Sarvam implementation
(`STT_translation.stt.speech_to_language`) directly in a background worker; no
separate STT server is required. The independent `/speech-to-language` endpoint
described above remains available for external integrations. Text from approved
upstream components can still be supplied through the structured ingestion
endpoints (`/sources/stt`, `/sources/ocr`, `/sources/message`) to preserve
provenance. Uploaded audio/PDF/images are stored encrypted as originals.

Endpoints:

- `POST /api/cases/{case_id}/sources/stt`
- `POST /api/cases/{case_id}/sources/ocr`
- `POST /api/cases/{case_id}/sources/message`

Consent is required. Ingesting the same payload twice is idempotent: stable source IDs are reported under `duplicates` instead of creating duplicate sources.

CareFlow's document intake runs `OCR1.model.extract_document()` and
`OCR1.organizer.organize_ocr_result()` in-process and persists the organizer's
structured output; no OCR HTTP service is called. External OCR callers can use
`/sources/ocr` with the structured source contract below. STT results are read
from `chunks`/`transcription` with `start_seconds`/`end_seconds`.

## STT payload

```json
{
  "transcript_id": "visit-123",
  "language": "hi",
  "language_status": "declared",
  "model": "your-stt-model",
  "confidence": 0.91,
  "captured_at": "2026-10-20T09:00:00Z",
  "segments": [
    {
      "text": "मुझे कल से बुखार है।",
      "start": 12.4,
      "end": 16.1,
      "confidence": 0.88,
      "language": "hi"
    }
  ]
}
```

Accepted aliases: `results` for `segments`; `transcript`/`text` for a single unsegmented transcript; `start_time`/`end_time` for `start`/`end`; `language_code`/`lang`; `recorded_at` for `captured_at`; `stt_model` for `model`; `id` for `transcript_id`.

Mapping:

- `transcript_id` -> `raw_metadata.transcript_id`, stable parent ID.
- `segments[i].text` -> original text (encrypted) and normalized text.
- `start`/`end` -> `SourceRef.audio_start`/`audio_end`.
- `confidence` -> source confidence; below 0.5 adds `low_source_confidence`.
- `language` -> declared language; script heuristics may add `language_metadata_mismatch` without overwriting metadata.
- Empty segments are skipped with notes; all-empty payloads are rejected with a 422-style `InputError`.

## OCR payload

Direct document intake no longer accepts vendor OCR layouts or calls an OCR
microservice. `OCR1.organizer.organize_ocr_result()` returns `file_name`,
`ocr_model`, `is_medical`, and `sources`. CareFlow adds `document_id` before
persistence. `/sources/ocr` also accepts optional `language`, `document_id`,
and `document_type` fields, as shown below:

```json
{
  "document_id": "lab-2026-10-20",
  "file_name": "report.pdf",
  "language": "en",
  "ocr_model": "glm-ocr",
  "is_medical": true,
  "document_type": "lab_report",
  "sources": [
    {
      "block_id": "p1_b1",
      "text": "Temperature 101.5 F.",
      "page": 1,
      "unreadable": false
    },
    {
      "block_id": "p1_b2",
      "text": "SpO2 ___ %",
      "page": 1,
      "unreadable": true
    }
  ]
}
```

GLM-OCR does not supply bounding boxes or confidence scores, so those fields
are absent from the in-process OCR output. External callers of `/sources/ocr`
may still include optional `bbox` and `confidence` fields for each source.

The organizer decides medical relevance and only emits medically relevant
sources. It uses a compact exclusion gate: the model returns ranges of
non-clinical blocks to drop, while all remaining OCR blocks are kept by
default. This reduces organizer latency without letting the model rewrite or
invent source text. If `is_medical` is false, the payload is rejected with
`non_medical_document` and nothing is persisted.

Mapping:

- Each selected source becomes a separate `SourceRecord` with `page`, optional `bbox`/confidence and stable ID.
- `unreadable: true` adds `unreadable_critical_field`, which forces `critical_failure` and prevents a routine outcome.
- Low confidence adds `low_source_confidence`; the extractor additionally caps confidence and marks `low_extraction_confidence`.
- `document_id`/`file_name` is preserved as a document reference. It is not the same as an encrypted upload ID.
- Prompt-injection indicators in the original text add `prompt_injection_suspected:*` warnings; they never alter rules or triage.

## Typed message payload

```json
{
  "text": "I have had a headache for two days.",
  "language": "en",
  "captured_at": "2026-10-20T10:00:00Z",
  "message_id": "msg-1",
  "confidence": 0.9
}
```

## Provenance expectations

Upstream components should send:

1. The **original** text, not just a cleaned version.
2. Stable transcript/document/segment IDs so retries deduplicate.
3. Segment timing for audio and page/bbox for documents, if available.
4. Per-segment confidence and declared language.
5. An explicit unreadable marker for critical fields the OCR could not read.

Do not send derived clinical conclusions from the STT/OCR component; send text and provenance only. Do not claim clinical accuracy for upstream models based on this integration.
