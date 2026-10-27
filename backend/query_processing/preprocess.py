"""Rule-based query normalisation that runs before BERT.

Nothing here encodes medical knowledge: it cleans text, extracts keywords,
guesses the *kind* of question being asked, and flags phrasings that warrant a
safety notice (emergencies, requests for personal diagnosis or dosing).
"""

import re
import unicodedata
from dataclasses import dataclass, field

_CONTROL_CHARS = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

_CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "how's": "how is", "who's": "who is",
    "where's": "where is", "when's": "when is", "why's": "why is", "it's": "it is",
    "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
    "isn't": "is not", "aren't": "are not", "i'm": "i am", "i've": "i have",
    "shouldn't": "should not", "couldn't": "could not", "wouldn't": "would not",
}
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down during
    each few for from further had has have having he her here hers herself him himself his
    how i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves tell explain please know get also may might
    must shall""".split()
)

# Ordered: the first matching pattern wins.
_QUESTION_TYPES: list[tuple[str, re.Pattern]] = [
    ("comparison", re.compile(r"\b(difference between|differ|compare|comparison|versus|vs\.?)\b")),
    ("symptoms", re.compile(r"\b(symptoms?|signs?|feel like|presentation)\b")),
    ("causes", re.compile(r"\b(causes?|caused|why do|why does|risk factors?|etiology)\b")),
    ("treatment", re.compile(r"\b(treat|treatment|treated|therapy|cure|manage|management|medication|medicine|drugs?)\b")),
    ("prevention", re.compile(r"\b(prevent\w*|avoid\w*|reduce the risk)\b")),
    ("diagnosis", re.compile(r"\b(diagnos\w*|tests?|detect\w*|screen\w*)\b")),
    ("prognosis", re.compile(r"\b(prognosis|outlook|life expectancy|recover\w*)\b")),
    ("definition", re.compile(r"^(what is|what are|define|meaning of|what does)\b|\bdefinition\b")),
]

_EMERGENCY = re.compile(
    r"\b(chest pain|can ?not breathe|cannot breathe|difficulty breathing|unconscious|"
    r"not breathing|seizure|overdose|severe bleeding|suicid\w*|kill myself|end my life|"
    r"self[- ]harm|stroke symptoms|heart attack)\b"
)
_PERSONAL_ADVICE = re.compile(
    r"\b(do i have|should i (take|stop|start|use)|how much .* should i|"
    r"what dose .* (me|i)|my (doctor|prescription|dose|dosage)|am i (sick|dying|pregnant))\b"
)


@dataclass
class PreprocessedQuery:
    original_text: str
    clean_text: str          # whitespace/unicode-normalised, original casing
    normalized_text: str     # lower-cased, contractions expanded
    tokens: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    question_type: str = "general"
    safety_flags: list[str] = field(default_factory=list)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = _CONTROL_CHARS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize(text: str) -> str:
    text = clean(text).lower()
    return _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(1)], text)


def tokenize(normalized_text: str) -> list[str]:
    return _WORD.findall(normalized_text)


def extract_keywords(tokens: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for tok in tokens:
        if tok not in STOPWORDS and (len(tok) > 1 or tok.isdigit()):
            seen.setdefault(tok, None)
    return list(seen)


def classify_question(normalized_text: str) -> str:
    for label, pattern in _QUESTION_TYPES:
        if pattern.search(normalized_text):
            return label
    return "general"


def detect_safety_flags(normalized_text: str) -> list[str]:
    flags = []
    if _EMERGENCY.search(normalized_text):
        flags.append("possible_emergency")
    if _PERSONAL_ADVICE.search(normalized_text):
        flags.append("personal_medical_advice")
    return flags


def preprocess(text: str) -> PreprocessedQuery:
    clean_text = clean(text)
    normalized = normalize(clean_text)
    tokens = tokenize(normalized)
    return PreprocessedQuery(
        original_text=text,
        clean_text=clean_text,
        normalized_text=normalized,
        tokens=tokens,
        keywords=extract_keywords(tokens),
        question_type=classify_question(normalized),
        safety_flags=detect_safety_flags(normalized),
    )
