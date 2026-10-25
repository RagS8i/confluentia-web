"""Runtime configuration for CareFlow AI.

Everything that affects processing is configurable through the environment so
that the model provider, ruleset and storage can be swapped without code
changes.  No real patient data or secrets are baked into this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RULES_PATH = PROJECT_ROOT / "app" / "triage" / "rules_demo.json"

SUPPORTED_LANGUAGES = {
    "en",
    "hi",
    "hinglish",
    "ta",
    "te",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "pa",
    "or",
    "as",
    "ur",
    "ne",
    "kok",
    "ks",
    "sd",
    "sa",
    "sat",
    "mni",
    "brx",
    "mai",
    "doi",
    "unknown",
}

ALLOWED_UPLOAD_MIME = (
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "text/plain",
)


def _load_dotenv(path: Optional[Path] = None) -> None:
    """Load simple KEY=VALUE lines without overriding real environment values."""
    target = path or PROJECT_ROOT / ".env"
    if not target.exists():
        return
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    db_path: Path = DEFAULT_DATA_DIR / "careflow.db"
    rules_path: Path = DEFAULT_RULES_PATH

    # Model provider: "sarvam" | "ollama" | "mock".  "mock" is deterministic and
    # intended for tests, CI and offline demonstrations only.  "ollama" is kept
    # as a legacy local fallback; Sarvam is the default hosted provider.
    llm_provider: str = "sarvam"

    # Sarvam AI: GLM-5.3 for clinical extraction and Sarvam Vision for OCR.
    sarvam_api_key: Optional[str] = None
    sarvam_base_url: str = "https://api.sarvam.ai"
    sarvam_chat_model: str = "glm5.3"
    sarvam_chat_timeout_s: float = 120.0
    sarvam_chat_max_tokens: int = 4096
    sarvam_reasoning_effort: str = "low"
    sarvam_vision_model: str = "sarvam-vision"
    sarvam_ocr_timeout_s: float = 300.0
    sarvam_ocr_poll_s: float = 2.0
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_tts_timeout_s: float = 60.0
    patient_voice_max_chars: int = 12_000

    # Legacy local Ollama provider (kept for offline development/tests).
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout_s: float = 120.0
    ollama_num_ctx: int = 8192
    ollama_num_gpu: Optional[int] = None
    ollama_keep_alive: str = "30m"
    ollama_temperature: float = 0.0
    max_model_retries: int = 2
    model_backoff_s: float = 0.5

    pipeline_version: str = "careflow-pipeline/0.1.0"
    prompt_version: str = "careflow-extract/0.1.0"

    max_upload_bytes: int = 20 * 1024 * 1024
    allowed_upload_mime: tuple = field(default_factory=lambda: ALLOWED_UPLOAD_MIME)
    processing_mode: str = "background"  # "background" | "sync"
    max_workers: int = 2

    # Encryption: base64 urlsafe 32-byte Fernet key.  If omitted a key is
    # generated once under data/.careflow.key (dev convenience only).
    encryption_key: Optional[str] = None
    auth_mode: str = "dev"  # "dev" (trusted headers) | "api_key"
    api_keys: dict = field(default_factory=dict)
    retention_days: int = 365
    source_access_audit: bool = True

    # Outbound workflow. "outbox" keeps messages/escalations in the local
    # reviewable outbox; "webhook" posts JSON to the configured URL; "fail"
    # simulates a delivery failure for drills and tests.
    communication_delivery: str = "outbox"
    escalation_delivery: str = "outbox"
    outbound_webhook_url: Optional[str] = None
    outbound_timeout_s: float = 5.0
    wait_estimate_min_sample: int = 3

    # Direct intake settings (STT and OCR both run in-process).
    # Upper bound on how long CareFlow waits for the blocking Sarvam call.
    # Exceeding it fails the job, but cannot stop an in-flight network request.
    stt_timeout_s: float = 120.0

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def key_file(self) -> Path:
        return self.data_dir / ".careflow.key"

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


def load_settings() -> Settings:
    _load_dotenv()
    data_dir = Path(os.getenv("CAREFLOW_DATA_DIR", str(DEFAULT_DATA_DIR)))
    api_keys_raw = os.getenv("CAREFLOW_API_KEYS", "")
    api_keys = {}
    if api_keys_raw:
        # format: key1:role1:user1,key2:role2:user2
        for entry in api_keys_raw.split(","):
            parts = entry.split(":")
            if len(parts) == 3:
                api_keys[parts[0]] = {"role": parts[1], "user_id": parts[2]}

    settings = Settings(
        data_dir=data_dir,
        db_path=Path(os.getenv("CAREFLOW_DB_PATH", str(data_dir / "careflow.db"))),
        rules_path=Path(
            os.getenv("CAREFLOW_RULES_PATH", str(DEFAULT_RULES_PATH))
        ),
        llm_provider=os.getenv("CAREFLOW_LLM_PROVIDER", "sarvam").strip().lower(),
        sarvam_api_key=os.getenv("SARVAM_API_KEY") or None,
        sarvam_base_url=os.getenv(
            "CAREFLOW_SARVAM_BASE_URL", "https://api.sarvam.ai"
        ).rstrip("/"),
        sarvam_chat_model=os.getenv("CAREFLOW_SARVAM_CHAT_MODEL", "glm5.3"),
        sarvam_chat_timeout_s=_float("CAREFLOW_SARVAM_TIMEOUT_S", 120.0),
        sarvam_chat_max_tokens=_int("CAREFLOW_SARVAM_MAX_TOKENS", 4096),
        sarvam_reasoning_effort=os.getenv(
            "CAREFLOW_SARVAM_REASONING_EFFORT", "low"
        ).strip().lower(),
        sarvam_vision_model=os.getenv(
            "CAREFLOW_OCR_MODEL", "sarvam-vision"
        ).strip() or "sarvam-vision",
        sarvam_ocr_timeout_s=_float("CAREFLOW_OCR_TIMEOUT_S", 300.0),
        sarvam_ocr_poll_s=_float("CAREFLOW_OCR_POLL_S", 2.0),
        sarvam_tts_model=os.getenv("CAREFLOW_TTS_MODEL", "bulbul:v3"),
        sarvam_tts_speaker=os.getenv("CAREFLOW_TTS_SPEAKER", "shubh"),
        sarvam_tts_timeout_s=_float("CAREFLOW_TTS_TIMEOUT_S", 60.0),
        patient_voice_max_chars=_int("CAREFLOW_PATIENT_VOICE_MAX_CHARS", 12_000),
        ollama_base_url=os.getenv(
            "CAREFLOW_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        ).rstrip("/"),
        ollama_model=os.getenv("CAREFLOW_OLLAMA_MODEL", "qwen3:8b"),
        ollama_timeout_s=_float("CAREFLOW_OLLAMA_TIMEOUT_S", 120.0),
        ollama_num_ctx=_int("CAREFLOW_OLLAMA_NUM_CTX", 8192),
        ollama_num_gpu=(
            _int("CAREFLOW_OLLAMA_NUM_GPU", -1)
            if os.getenv("CAREFLOW_OLLAMA_NUM_GPU") not in (None, "")
            else None
        ),
        ollama_keep_alive=os.getenv("CAREFLOW_OLLAMA_KEEP_ALIVE", "30m"),
        ollama_temperature=_float("CAREFLOW_OLLAMA_TEMPERATURE", 0.0),
        max_model_retries=_int("CAREFLOW_MODEL_RETRIES", 2),
        model_backoff_s=_float("CAREFLOW_MODEL_BACKOFF_S", 0.5),
        pipeline_version=os.getenv(
            "CAREFLOW_PIPELINE_VERSION", "careflow-pipeline/0.1.0"
        ),
        prompt_version=os.getenv(
            "CAREFLOW_PROMPT_VERSION", "careflow-extract/0.1.0"
        ),
        max_upload_bytes=_int("CAREFLOW_MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
        processing_mode=os.getenv("CAREFLOW_PROCESSING_MODE", "background").lower(),
        max_workers=_int("CAREFLOW_MAX_WORKERS", 2),
        encryption_key=os.getenv("CAREFLOW_ENCRYPTION_KEY") or None,
        auth_mode=os.getenv("CAREFLOW_AUTH_MODE", "dev").lower(),
        api_keys=api_keys,
        retention_days=_int("CAREFLOW_RETENTION_DAYS", 365),
        source_access_audit=_bool("CAREFLOW_SOURCE_ACCESS_AUDIT", True),
        communication_delivery=os.getenv("CAREFLOW_COMMUNICATION_DELIVERY", "outbox").lower(),
        escalation_delivery=os.getenv("CAREFLOW_ESCALATION_DELIVERY", "outbox").lower(),
        outbound_webhook_url=os.getenv("CAREFLOW_OUTBOUND_WEBHOOK_URL") or None,
        outbound_timeout_s=_float("CAREFLOW_OUTBOUND_TIMEOUT_S", 5.0),
        wait_estimate_min_sample=_int("CAREFLOW_WAIT_ESTIMATE_MIN_SAMPLE", 3),
        stt_timeout_s=_float("CAREFLOW_STT_TIMEOUT_S", 120.0),
    )
    for directory in (settings.data_dir, settings.uploads_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
