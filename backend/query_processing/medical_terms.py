"""Small, deterministic aliases for terminology found in the prepared corpus.

These variants are retrieval hints only. They never replace or alter the user's
question and are not used to infer diagnoses or treatment.
"""

import re

from query_processing.preprocess import normalize

_TERM_PAIRS = (
    ("influenza", "flu"),
    ("high blood pressure", "hypertension"),
    ("high blood sugar", "hyperglycemia"),
    ("gastroesophageal reflux disease", "gerd"),
    ("hemoglobin a1c", "hba1c"),
    ("hemoglobin a1c", "a1c"),
)


def expand_medical_terms(question: str) -> list[str]:
    """Return normalized original wording plus only direct corpus-backed aliases."""
    normalized = normalize(question)
    variants = [normalized]
    additions: list[str] = []
    for left, right in _TERM_PAIRS:
        for source, alias in ((left, right), (right, left)):
            if re.search(r"(?<![a-z0-9])" + re.escape(source) + r"(?![a-z0-9])", normalized):
                if not re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", normalized):
                    additions.append(alias)
    if additions:
        variants.append(normalized + " " + " ".join(dict.fromkeys(additions)))
    return variants
