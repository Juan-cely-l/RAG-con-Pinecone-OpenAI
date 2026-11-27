"""Input validation for user questions."""

from __future__ import annotations

import re
import unicodedata

from errors import InvalidInputError

_WHITESPACE_RE = re.compile(r"\s+")


def _strip_control_chars(text: str) -> str:
    """Remove control/format characters, turning line breaks and tabs into spaces.

    Normal unicode (accents, ``¿``, ``ñ``, emoji...) is preserved.
    """
    cleaned: list[str] = []
    for char in text:
        if char in "\t\n\r\x0b\x0c":
            cleaned.append(" ")
            continue
        category = unicodedata.category(char)
        # Cc = control, Cf = format (zero-width, bidi overrides), Cs = surrogates,
        # Co = private use, Cn = unassigned.
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            continue
        cleaned.append(char)
    return "".join(cleaned)


def validate_question(q: str | None, max_chars: int) -> str:
    """Normalize and validate a user question.

    Strips control characters, collapses whitespace and checks length/content.
    Returns the normalized question or raises ``InvalidInputError``.
    """
    if q is None:
        raise InvalidInputError("La pregunta no puede estar vacía. / The question cannot be empty.")
    if not isinstance(q, str):
        raise InvalidInputError("La pregunta debe ser texto. / The question must be a string.")

    normalized = unicodedata.normalize("NFC", q)
    normalized = _strip_control_chars(normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if not normalized:
        raise InvalidInputError(
            "La pregunta no puede estar vacía ni contener solo espacios. "
            "/ The question cannot be empty or whitespace only."
        )
    if len(normalized) > max_chars:
        raise InvalidInputError(
            f"La pregunta es demasiado larga ({len(normalized)} caracteres; "
            f"máximo {max_chars}). / The question is too long "
            f"({len(normalized)} chars; limit {max_chars})."
        )
    if not any(char.isalnum() for char in normalized):
        raise InvalidInputError(
            "La pregunta debe contener al menos una letra o un número. "
            "/ The question must contain at least one letter or digit."
        )
    return normalized
