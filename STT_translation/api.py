from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

try:  # Imported as the STT_translation package.
    from .schemas import Language, SpeechTranslationRequest
except ImportError:  # Run with --app-dir STT_translation.
    from schemas import Language, SpeechTranslationRequest  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load STT and translation models here if needed.
    yield


app = FastAPI(
    title="Careflow Speech Translation API",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/speech-to-language")
async def translate_audio(
    audio: UploadFile = File(...),
    target_language: Language = Form(Language.ENGLISH),
):
    try:
        request = SpeechTranslationRequest(
            target_language=target_language,
        )

        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        def _process():
            try:
                from .stt import speech_to_language
            except ImportError:
                from stt import speech_to_language  # type: ignore[no-redef]
            return speech_to_language(
                audio_bytes=audio_bytes,
                target_language=request.target_language.value,
                filename=audio.filename or "audio.wav",
            )

        return await run_in_threadpool(_process)

    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc
    except Exception as exc:
        logger.exception("Speech translation failed")
        raise HTTPException(
            status_code=500,
            detail="Speech transcription or translation failed.",
        ) from exc
