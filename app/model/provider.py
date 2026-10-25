"""Configurable language-model providers.

``SarvamProvider`` calls GLM-5.3 through Sarvam's OpenAI-compatible chat
completions API.  ``OllamaProvider`` targets Qwen3-8B (or any Ollama chat
model) and remains available as a legacy local fallback.  ``MockProvider`` is a
deterministic lexicon-based stand-in used for tests, CI and offline
demonstrations; it is *not* an SLM and must not be used to claim model quality.

Core application code depends only on this small interface, never on a vendor's
response format.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import httpx

from ..config import Settings
from ..schemas import ExtractionResult, FactStatus, ExtractedFact, FactType, Modality, SourceRecord
from .sarvam import SarvamChatError, chat_completion


class ModelError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _strip_code_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class BaseProvider:
    name = "base"
    model = "none"

    def extract_facts(self, sources: Sequence[SourceRecord], settings: Settings) -> ExtractionResult:
        raise NotImplementedError

    def draft_summary(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        raise NotImplementedError

    def assess_urgency(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "status": "unknown"}


class SarvamProvider(BaseProvider):
    """GLM-5.3 clinical extraction through Sarvam chat completions v2."""

    name = "sarvam"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.sarvam_chat_model
        self.base_url = settings.sarvam_base_url
        self._extraction_schema = ExtractionResult.model_json_schema()
        # ExtractedFact has defaults for these safety-critical fields, so
        # Pydantic omits them from ``required``. Structured-output models then
        # drop them and the server silently applies "unknown"/"patient_reported".
        fact_schema = self._extraction_schema.get("$defs", {}).get("ExtractedFact")
        if fact_schema is not None:
            required = fact_schema.setdefault("required", [])
            for field_name in ("status", "assertion"):
                if field_name not in required:
                    required.append(field_name)
        self._draft_schema = {
            "type": "object",
            "properties": {
                "summary_narrative": {"type": "string"},
                "draft_referral_note": {"type": "string"},
            },
            "required": ["summary_narrative", "draft_referral_note"],
        }
        self._urgency_schema = {
            "type": "object",
            "properties": {
                "urgency_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "rationale": {"type": "string"},
            },
            "required": ["urgency_score", "rationale"],
        }

    def _chat(
        self,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> str:
        try:
            return chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model=self.model,
                schema=schema,
                api_key=self.settings.sarvam_api_key,
                base_url=self.base_url,
                timeout_s=timeout or self.settings.sarvam_chat_timeout_s,
                temperature=0.0,
                max_tokens=self.settings.sarvam_chat_max_tokens,
                reasoning_effort=self.settings.sarvam_reasoning_effort or None,
            )
        except SarvamChatError as exc:
            raise ModelError(exc.code, str(exc), exc.retryable) from exc

    def extract_facts(self, sources: Sequence[SourceRecord], settings: Settings) -> ExtractionResult:
        from .prompts import EXTRACTION_SYSTEM, build_extraction_prompt

        content = self._chat(EXTRACTION_SYSTEM, build_extraction_prompt(sources), self._extraction_schema)
        try:
            parsed = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise ModelError("malformed_output", "model did not return valid JSON", True) from exc
        try:
            return ExtractionResult.model_validate(parsed)
        except Exception as exc:
            raise ModelError("schema_invalid", "model JSON failed schema validation", True) from exc

    def draft_summary(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        from .prompts import DRAFT_SYSTEM, build_draft_prompt

        content = self._chat(DRAFT_SYSTEM, build_draft_prompt(payload), self._draft_schema)
        try:
            parsed = json.loads(_strip_code_fences(content))
            if not isinstance(parsed, dict):
                raise ValueError("not an object")
            return {
                "summary_narrative": str(parsed.get("summary_narrative") or ""),
                "draft_referral_note": str(parsed.get("draft_referral_note") or ""),
            }
        except (ValueError, TypeError) as exc:
            raise ModelError("malformed_output", "draft JSON was invalid", True) from exc

    def assess_urgency(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        from .prompts import URGENCY_SYSTEM, build_urgency_prompt

        content = self._chat(URGENCY_SYSTEM, build_urgency_prompt(payload), self._urgency_schema)
        try:
            parsed = json.loads(_strip_code_fences(content))
            if not isinstance(parsed, dict):
                raise ValueError("not an object")
            return {
                "urgency_score": parsed.get("urgency_score"),
                "rationale": str(parsed.get("rationale") or ""),
            }
        except (ValueError, TypeError) as exc:
            raise ModelError("malformed_output", "urgency JSON was invalid", True) from exc

    def health(self) -> Dict[str, Any]:
        configured = bool(self.settings.sarvam_api_key)
        return {
            "provider": self.name,
            "model": self.model,
            "status": "ok" if configured else "unavailable",
            "api_key_configured": configured,
        }


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url
        self._extraction_schema = ExtractionResult.model_json_schema()
        # ExtractedFact has defaults for these safety-critical fields, so
        # Pydantic omits them from ``required``. Structured-output models then
        # drop them and the server silently applies "unknown"/"patient_reported".
        fact_schema = self._extraction_schema.get("$defs", {}).get("ExtractedFact")
        if fact_schema is not None:
            required = fact_schema.setdefault("required", [])
            for field_name in ("status", "assertion"):
                if field_name not in required:
                    required.append(field_name)
        self._draft_schema = {
            "type": "object",
            "properties": {
                "summary_narrative": {"type": "string"},
                "draft_referral_note": {"type": "string"},
            },
            "required": ["summary_narrative", "draft_referral_note"],
        }
        self._urgency_schema = {
            "type": "object",
            "properties": {
                "urgency_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "rationale": {"type": "string"},
            },
            "required": ["urgency_score", "rationale"],
        }

    def _chat(
        self,
        system: str,
        user: str,
        schema: Optional[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> str:
        options = {
            "temperature": self.settings.ollama_temperature,
            "num_ctx": self.settings.ollama_num_ctx,
        }
        if self.settings.ollama_num_gpu is not None:
            options["num_gpu"] = self.settings.ollama_num_gpu
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
            "keep_alive": self.settings.ollama_keep_alive,
        }
        if schema is not None:
            body["format"] = schema
        if self.model.lower().startswith("qwen3"):
            body["think"] = False

        try:
            with httpx.Client(timeout=timeout or self.settings.ollama_timeout_s) as client:
                response = client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise ModelError("timeout", "model request timed out", True) from exc
        except httpx.HTTPError as exc:
            raise ModelError("unavailable", "model server is unavailable", True) from exc

        if response.status_code >= 400:
            detail = response.text[:500].lower()
            if "out of memory" in detail or "cuda" in detail and "memory" in detail:
                raise ModelError("oom", "model server ran out of memory", True)
            if schema is not None and response.status_code == 400 and "format" in detail:
                body.pop("format", None)
                body["format"] = "json"
                try:
                    with httpx.Client(timeout=timeout or self.settings.ollama_timeout_s) as client:
                        response = client.post(f"{self.base_url}/api/chat", json=body)
                except httpx.HTTPError as exc:
                    raise ModelError("unavailable", "model server is unavailable", True) from exc
                if response.status_code >= 400:
                    raise ModelError("server_error", "model server returned an error", True)
            else:
                raise ModelError("server_error", "model server returned an error", True)

        try:
            data = response.json()
            content = (data.get("message") or {}).get("content") or ""
        except (ValueError, KeyError) as exc:
            raise ModelError("malformed_output", "model response envelope was invalid", True) from exc
        if not content.strip():
            raise ModelError("malformed_output", "model returned empty content", True)
        return content

    def extract_facts(self, sources: Sequence[SourceRecord], settings: Settings) -> ExtractionResult:
        from .prompts import EXTRACTION_SYSTEM, build_extraction_prompt

        user = build_extraction_prompt(sources)
        content = self._chat(EXTRACTION_SYSTEM, user, self._extraction_schema)
        try:
            parsed = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise ModelError("malformed_output", "model did not return valid JSON", True) from exc
        try:
            result = ExtractionResult.model_validate(parsed)
        except Exception as exc:
            raise ModelError("schema_invalid", "model JSON failed schema validation", True) from exc
        return result

    def draft_summary(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        from .prompts import DRAFT_SYSTEM, build_draft_prompt

        user = build_draft_prompt(payload)
        content = self._chat(DRAFT_SYSTEM, user, self._draft_schema)
        try:
            parsed = json.loads(_strip_code_fences(content))
            if not isinstance(parsed, dict):
                raise ValueError("not an object")
            return {
                "summary_narrative": str(parsed.get("summary_narrative") or ""),
                "draft_referral_note": str(parsed.get("draft_referral_note") or ""),
            }
        except (ValueError, TypeError) as exc:
            raise ModelError("malformed_output", "draft JSON was invalid", True) from exc

    def assess_urgency(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        from .prompts import URGENCY_SYSTEM, build_urgency_prompt

        user = build_urgency_prompt(payload)
        content = self._chat(URGENCY_SYSTEM, user, self._urgency_schema)
        try:
            parsed = json.loads(_strip_code_fences(content))
            if not isinstance(parsed, dict):
                raise ValueError("not an object")
            return {
                "urgency_score": parsed.get("urgency_score"),
                "rationale": str(parsed.get("rationale") or ""),
            }
        except (ValueError, TypeError) as exc:
            raise ModelError("malformed_output", "urgency JSON was invalid", True) from exc

    def health(self) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [m.get("name") for m in response.json().get("models", [])]
                available = self.model in models
                return {
                    "provider": self.name,
                    "model": self.model,
                    "status": "ok" if available else "degraded",
                    "model_available": available,
                    "available_models": models[:20],
                }
            return {"provider": self.name, "model": self.model, "status": "degraded"}
        except httpx.HTTPError:
            return {"provider": self.name, "model": self.model, "status": "unavailable"}


@dataclass
class _Term:
    canonical: str
    pattern: re.Pattern
    fact_type: FactType = FactType.symptom


def _term(canonical: str, variants: Sequence[str], fact_type=FactType.symptom) -> _Term:
    escaped = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return _Term(canonical, re.compile(f"(?i)({escaped})"), fact_type)


_SYMPTOMS = [
    _term("chest pain", ["chest pain", "pain in chest", "seene me dard", "chhati me dard", "सीने में दर्द", "छाती में दर्द"]),
    _term("shortness of breath", ["shortness of breath", "breathlessness", "difficulty breathing", "cannot breathe", "saans lene me takleef", "saans phool", "सांस लेने में तकलीफ", "सांस फूलना", "saans"]),
    _term("fever", ["fever", "bukhar", "बुखार"]),
    _term("cough", ["cough", "khansi", "खांसी"]),
    _term("headache", ["headache", "sir dard", "sirdard", "सिरदर्द", "सिर में दर्द"]),
    _term("dizziness", ["dizziness", "dizzy", "chakkar", "चक्कर"]),
    _term("vomiting", ["vomiting", "vomited", "ulti", "उल्टी"]),
    _term("nausea", ["nausea", "जी मिचलाना"]),
    _term("abdominal pain", ["abdominal pain", "stomach pain", "pet dard", "pet me dard", "पेट दर्द"]),
    _term("diarrhea", ["diarrhea", "loose motion", "loose motions", "dast", "दस्त"]),
    _term("rash", ["rash", "चकत्ते"]),
    _term("bleeding", ["bleeding", "khoon", "खून"]),
    _term("weakness", ["weakness", "fatigue", "kamzori", "कमजोरी", "थकान"]),
    _term("sore throat", ["sore throat", "gale me dard", "गले में दर्द"]),
    _term("seizure", ["seizure", "fits", "दौरा", "मिर्गी"]),
    _term("palpitations", ["palpitations", "heart racing", "धड़कन"]),
    _term("swelling", ["swelling", "सूजन"]),
    _term("burning urination", ["burning urination", "पेशाब में जलन"]),
    _term("blurred vision", ["blurred vision", "धुंधला दिखना"]),
    _term("loss of consciousness", ["unconscious", "fainted", "बेहोश"]),
]

_CONDITIONS = [
    ("diabetes", ["diabetes", "diabetic", "sugar disease", "मधुमेह"]),
    ("hypertension", ["hypertension", "high blood pressure", "high bp", "उच्च रक्तचाप"]),
    ("asthma", ["asthma", "दमा"]),
    ("heart disease", ["heart disease", "heart problem", "दिल की बीमारी"]),
    ("kidney disease", ["kidney disease", "renal disease", "गुर्दे"]),
    ("tuberculosis", ["tuberculosis", "tb"]),
    ("pregnancy", ["pregnant", "pregnancy", "गर्भवती"]),
    ("cancer", ["cancer", "कैंसर"]),
    ("stroke", ["stroke", "paralysis", "लकवा"]),
]

_MEDICATIONS = [
    "metformin", "amlodipine", "insulin", "paracetamol", "acetaminophen",
    "ibuprofen", "aspirin", "atorvastatin", "warfarin", "losartan",
    "telmisartan", "levothyroxine", "salbutamol", "inhaler", "prednisolone",
    "azithromycin", "amoxicillin", "omeprazole", "pantoprazole",
]

_MED_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(m) for m in sorted(_MEDICATIONS, key=len, reverse=True)) + r")\b"
)

_NEGATION = re.compile(
    r"(?i)\b(no|not|without|denies|denied|never|n't|nahi|nahin|नहीं|बिना|बिल्कुल नहीं)\b"
)
_UNCERTAIN = re.compile(
    r"(?i)\b(maybe|perhaps|possibly|not sure|i think|unsure|shayad|pata nahi|lagta hai|अंदाजा|शायद|पता नहीं)\b"
)
_PROGRESSION_WORSE = re.compile(r"(?i)\b(worse|worsening|increasing|badh raha|बढ़|बिगड़)\b")
_PROGRESSION_BETTER = re.compile(r"(?i)\b(better|improving|kam ho raha|ठीक हो|सुधार)\b")
_PROGRESSION_CONSTANT = re.compile(r"(?i)\b(constant|continuous|same|lagatar|लगातार)\b")
_PROGRESSION_INTERMITTENT = re.compile(r"(?i)\b(intermittent|sometimes|kabhi kabhi|कभी कभी)\b")
_HISTORICAL = re.compile(
    r"(?i)\b(was|were|previously|earlier|last\s+week|last\s+month|history\s+of|पहले|पिछले)\b"
)

_TEMPORAL_PATTERNS = [
    (r"(?i)\bsince\s+yesterday\b", 24.0, "day"),
    (r"(?i)\b(yesterday|kal|कल)\b", 24.0, "day"),
    (r"(?i)\b(started\s+)?this\s+morning\b", 8.0, "hour"),
    (r"(?i)\b(aaj\s+subah|आज सुबह)\b", 8.0, "hour"),
    (r"(?i)\b(last\s+night|raat\s+se|रात से)\b", 12.0, "hour"),
    (r"(?i)\bfor\s+(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*(hour|hours|hrs|day|days|week|weeks|month|months|saal|sal|din)\b", None, "duration"),
    (r"(?i)\b(\d+(?:\.\d+)?)\s*(hour|hours|day|days|week|weeks|month|months|din|saal)\s+(?:ago|pehle|पहले)\b", None, "ago"),
    (r"(?i)\b(since\s+(?:last\s+)?(?:week|month)|\d+\s+(?:din|दिन)\s+se)\b", None, "since"),
    (r"(?i)\b(for\s+\d+\s+(?:din|दिन))\b", None, "duration"),
    (r"(?i)\b(since\s+(?:childhood|birth))\b", None, "long_term"),
]

_UNIT_FACTORS = {
    "hour": 1.0,
    "hours": 1.0,
    "hrs": 1.0,
    "day": 24.0,
    "days": 24.0,
    "din": 24.0,
    "दिन": 24.0,
    "week": 168.0,
    "weeks": 168.0,
    "month": 720.0,
    "months": 720.0,
    "saal": 8760.0,
    "sal": 8760.0,
}


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind(";", 0, start),
        text.rfind("\n", 0, start),
    )
    rights = [pos for pos in (text.find(".", end), text.find("!", end), text.find("?", end), text.find(";", end), text.find("\n", end)) if pos != -1]
    right = min(rights) if rights else len(text)
    return left + 1, right


def find_temporal(text: str) -> Optional[str]:
    for pattern, _hours, _kind in _TEMPORAL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def _negated(text: str, start: int, end: int) -> bool:
    left, _ = _clause_bounds(text, start, end)
    window = text[left: min(len(text), end + 20)]
    # A negation before a contrast marker ("no fever BUT chest pain") applies
    # only to the clause that contains the matched term.
    rel = start - left
    spans = []
    cursor = 0
    for marker in re.finditer(r"(?i)\b(?:but|however|lekin|magar|लेकिन|मगर)\b", window):
        spans.append((cursor, marker.start()))
        cursor = marker.end()
    spans.append((cursor, len(window)))
    segment = next((window[a:b] for a, b in spans if a <= rel <= b), window)
    return bool(_NEGATION.search(segment))


class MockProvider(BaseProvider):
    """Deterministic lexicon extractor for tests and synthetic demos."""

    name = "mock"

    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.model = "careflow-mock-lexicon/0.1.0"

    def _maybe_fail(self, sources: Sequence[SourceRecord]) -> None:
        if self.mode in {"timeout", "unavailable", "oom", "malformed", "schema_invalid"}:
            if self.mode == "timeout":
                raise ModelError("timeout", "simulated model timeout", True)
            if self.mode == "unavailable":
                raise ModelError("unavailable", "simulated model server unavailability", True)
            if self.mode == "oom":
                raise ModelError("oom", "simulated out-of-memory", True)
            if self.mode == "schema_invalid":
                raise ModelError("schema_invalid", "simulated schema failure", True)
            raise ModelError("malformed_output", "simulated malformed JSON", True)
        if self.mode == "fail_ocr" and any(s.modality == Modality.ocr for s in sources):
            raise ModelError("unavailable", "simulated OCR extraction failure", True)

    def extract_facts(self, sources: Sequence[SourceRecord], settings: Settings) -> ExtractionResult:
        self._maybe_fail(sources)
        facts: List[ExtractedFact] = []
        questions: List[str] = []
        seen: set[tuple[str, str, str]] = set()

        for source in sources:
            if not source.usable:
                continue
            text = source.original_text
            for term in _SYMPTOMS:
                for match in term.pattern.finditer(text):
                    left, right = _clause_bounds(text, match.start(), match.end())
                    quote = text[left:right].strip()
                    if not quote:
                        quote = match.group(0)
                    status = FactStatus.absent if _negated(text, match.start(), match.end()) else FactStatus.present
                    if _UNCERTAIN.search(quote) and status == FactStatus.present:
                        status = FactStatus.uncertain
                    key = (term.canonical, quote, status.value)
                    if key in seen:
                        continue
                    seen.add(key)
                    temporal = find_temporal(quote)
                    progression = None
                    if _PROGRESSION_WORSE.search(quote):
                        progression = "worsening"
                    elif _PROGRESSION_BETTER.search(quote):
                        progression = "improving"
                    elif _PROGRESSION_CONSTANT.search(quote):
                        progression = "constant"
                    elif _PROGRESSION_INTERMITTENT.search(quote):
                        progression = "intermittent"
                    facts.append(
                        ExtractedFact(
                            fact_type=term.fact_type,
                            label=term.canonical,
                            value=term.canonical,
                            status=status,
                            original_text=quote,
                            normalized_text=term.canonical,
                            symptom_onset=temporal,
                            progression=progression,
                            confidence=source.confidence or 0.85,
                            notes="mock_lexicon",
                        )
                    )

            for label, variants in _CONDITIONS:
                pattern = re.compile(r"(?i)\b(" + "|".join(re.escape(v) for v in variants) + r")\b")
                match = pattern.search(text)
                if not match:
                    continue
                left, right = _clause_bounds(text, match.start(), match.end())
                quote = text[left:right].strip()
                key = (f"history:{label}", quote, "historical")
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    ExtractedFact(
                        fact_type=FactType.history,
                        label=label,
                        value=label,
                        status=FactStatus.historical,
                        original_text=quote,
                        normalized_text=label,
                        confidence=source.confidence or 0.8,
                        notes="mock_lexicon",
                    )
                )

            for match in _MED_RE.finditer(text):
                drug = match.group(1).lower()
                left, right = _clause_bounds(text, match.start(), match.end())
                quote = text[left:right].strip()
                status = FactStatus.present
                window = text[max(0, match.start() - 40): match.end() + 20].lower()
                if re.search(r"\b(stopped|discontinued|band kar|बंद)\b", window):
                    status = FactStatus.historical
                elif _NEGATION.search(window):
                    status = FactStatus.absent
                dose = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|iu|units?)", quote, re.I)
                key = (f"med:{drug}", quote, status.value)
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    ExtractedFact(
                        fact_type=FactType.medication,
                        label=drug,
                        value=drug,
                        status=status,
                        unit=dose.group(2).lower() if dose else None,
                        original_text=quote,
                        normalized_text=drug,
                        confidence=source.confidence or 0.8,
                        notes="mock_lexicon",
                    )
                )

            for match in re.finditer(r"(?i)\ballergic\s+to\s+([A-Za-z\u0900-\u097F ]{2,30})", text):
                substance = match.group(1).strip().split(" ")[0]
                left, right = _clause_bounds(text, match.start(), match.end())
                quote = text[left:right].strip()
                key = (f"allergy:{substance}", quote, "present")
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    ExtractedFact(
                        fact_type=FactType.history,
                        label=f"allergy: {substance}",
                        value=substance,
                        status=FactStatus.present,
                        original_text=quote,
                        normalized_text=f"allergy to {substance}",
                        confidence=0.8,
                        notes="mock_lexicon",
                    )
                )

            for match in re.finditer(
                r"(?i)\b(?:bp|blood pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b",
                text,
            ):
                left, right = _clause_bounds(text, match.start(), match.end())
                quote = text[left:right].strip()
                key = ("blood pressure", quote, "present")
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    ExtractedFact(
                        fact_type=FactType.vital_sign,
                        label="blood pressure",
                        value=f"{match.group(1)}/{match.group(2)}",
                        unit="mmHg",
                        status=FactStatus.present,
                        original_text=quote,
                        normalized_text=f"blood pressure {match.group(1)}/{match.group(2)} mmHg",
                        confidence=source.confidence or 0.85,
                        notes="mock_lexicon",
                    )
                )

            for label, pattern, default_unit, ftype in [
                ("SpO2", r"(?i)\b(?:spo2|oxygen saturation|o2 sat|saturation)\D{0,12}(\d{2,3})\s*%?", "%", FactType.vital_sign),
                ("temperature", r"(?i)\b(?:temperature|temp)\D{0,12}(\d{2,3}(?:\.\d+)?)\s*(?:°?\s*([cf]))?", "C", FactType.vital_sign),
                ("blood glucose", r"(?i)\b(?:blood sugar|sugar|glucose)\D{0,12}(\d{2,3}(?:\.\d+)?)\s*(mg/dl|mmol/l)?", "mg/dL", FactType.measurement),
                ("pulse", r"(?i)\b(?:pulse|heart rate|hr)\D{0,12}(\d{2,3})\s*(bpm|/min)?", "bpm", FactType.vital_sign),
            ]:
                for match in re.finditer(pattern, text):
                    left, right = _clause_bounds(text, match.start(), match.end())
                    quote = text[left:right].strip()
                    unit = default_unit
                    if match.lastindex and match.lastindex >= 2:
                        maybe_unit = match.group(2)
                        if maybe_unit:
                            unit = maybe_unit.upper() if maybe_unit.lower() in {"c", "f"} else maybe_unit
                    status = (
                        FactStatus.historical
                        if _HISTORICAL.search(quote)
                        and not re.search(r"(?i)\b(now|current|today|aaj|अभी)\b", quote)
                        else FactStatus.present
                    )
                    key = (label, quote, status.value)
                    if key in seen:
                        continue
                    seen.add(key)
                    facts.append(
                        ExtractedFact(
                            fact_type=ftype,
                            label=label,
                            value=match.group(1),
                            unit=unit,
                            status=status,
                            original_text=quote,
                            normalized_text=f"{label} {match.group(1)} {unit}",
                            confidence=source.confidence or 0.85,
                            notes="mock_lexicon",
                        )
                    )

            if not any(f.fact_type == FactType.symptom for f in facts):
                questions.append("Can you describe your main symptom and when it started?")

        return ExtractionResult(
            facts=facts,
            follow_up_questions=questions,
            interpretation_notes=["deterministic mock extraction; not a model evaluation"],
        )

    def assess_urgency(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        """Deterministic stand-in for tests only; not a clinical model."""
        self._maybe_fail([])
        outcome = str(payload.get("workflow_status") or "UNASSESSED")
        score = {
            "EMERGENCY": 90,
            "PRIORITY_REVIEW": 65,
            "UNASSESSED": 45,
            "ROUTINE_REVIEW": 15,
        }.get(outcome, 45)
        facts = payload.get("facts") or []
        labels = {
            str(item.get("label", "")).strip().lower()
            for item in facts
            if item.get("status") == "present" and item.get("fact_type") == "symptom"
        }
        weights = {
            "chest pain": 10,
            "shortness of breath": 10,
            "breathing difficulty": 10,
            "bleeding": 8,
            "seizure": 10,
            "loss of consciousness": 12,
            "vomiting": 4,
        }
        for label, weight in weights.items():
            if label in labels:
                score += weight
        for item in facts:
            if item.get("fact_type") not in {"measurement", "vital_sign"}:
                continue
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            unit = str(item.get("unit") or "").strip().lower()
            label = str(item.get("label", "")).strip().lower()
            if ("spo2" in label or "oxygen saturation" in label) and value < 92:
                score += 8
            if ("temperature" in label or "temp" in label) and (
                (unit == "c" and value >= 39) or (unit == "f" and value >= 102.2)
            ):
                score += 5
        score = max(0, min(100, score))
        return {
            "urgency_score": score,
            "rationale": (
                "Mock advisory urgency derived from the supplied rule outcome, "
                "present symptoms and measurements. Not a clinical model and not "
                "clinically validated."
            ),
        }

    def draft_summary(self, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
        self._maybe_fail([])
        facts = payload.get("facts", [])
        symptoms = [f["label"] for f in facts if f.get("fact_type") == "symptom" and f.get("status") == "present"]
        negated = [f["label"] for f in facts if f.get("status") == "absent"]
        complaint = symptoms[0] if symptoms else "not clearly stated"
        narrative = (
            f"Patient reports {', '.join(dict.fromkeys(symptoms))}."
            if symptoms
            else "No clearly stated symptom was extracted from the available sources."
        )
        if negated:
            narrative += " Explicitly denied: " + ", ".join(dict.fromkeys(negated)) + "."
        narrative += " This is a documentation aid, not a diagnosis."
        note = (
            "AI DRAFT - FOR CLINICIAN REVIEW\n"
            f"Presenting complaint: {complaint}.\n"
            f"Reported symptoms: {', '.join(dict.fromkeys(symptoms)) or 'none extracted'}.\n"
            "Clinician must verify evidence, contradictions and missing information before use."
        )
        return {"summary_narrative": narrative, "draft_referral_note": note}


def build_provider(settings: Settings, provider: Optional[BaseProvider] = None) -> BaseProvider:
    if provider is not None:
        return provider
    if settings.llm_provider == "mock":
        return MockProvider()
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings)
    if settings.llm_provider == "sarvam":
        return SarvamProvider(settings)
    raise ValueError(
        f"Unsupported CAREFLOW_LLM_PROVIDER={settings.llm_provider!r}; "
        "use 'sarvam', 'ollama' or 'mock'."
    )
