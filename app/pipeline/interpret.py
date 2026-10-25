"""Stage B - multilingual interpretation readiness.

Original language and wording are preserved.  This stage records whether the
language is declared/detected/unsupported and flags ambiguous wording for
human verification; it does not silently translate or overwrite source text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from ..schemas import LanguageStatus, SourceRecord

_AMBIGUITY = re.compile(
    r"(?i)\b(maybe|perhaps|possibly|not sure|i think|unsure|sometimes|"
    r"shayad|pata nahi|lagta hai|kabhi kabhi|अंदाजा|शायद|पता नहीं|कभी कभी)\b"
)


@dataclass
class InterpretationReport:
    notes: List[str] = field(default_factory=list)
    ambiguous_source_ids: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    unsupported_languages: List[str] = field(default_factory=list)


def interpret_sources(sources: Sequence[SourceRecord]) -> InterpretationReport:
    report = InterpretationReport()
    for source in sources:
        if source.language and source.language not in report.languages:
            report.languages.append(source.language)
        if source.language_status == LanguageStatus.unsupported:
            if source.language not in report.unsupported_languages:
                report.unsupported_languages.append(source.language)
                report.notes.append(
                    f"Language '{source.language}' is outside the documented evaluation set; "
                    "interpretation is marked uncertain."
                )
        if source.language_status == LanguageStatus.unknown:
            report.notes.append(
                f"Source {source.source_id} has unknown language metadata; interpretation may be unreliable."
            )
        if _AMBIGUITY.search(source.original_text or ""):
            report.ambiguous_source_ids.append(source.source_id)
            report.notes.append(
                f"Source {source.source_id} contains ambiguous/uncertain wording; claims from it require verification."
            )
    return report
