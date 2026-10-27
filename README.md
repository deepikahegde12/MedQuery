# MedQuery

**An Intelligent Medical Question Answering System Using BERT and Knowledge Retrieval**

MedQuery takes a medical question in plain language, processes it with a pretrained
BERT model, retrieves evidence from a connected medical knowledge base, and extracts
an answer only when that evidence supports one. Every answer is shown next to the
passage it came from and that passage's source.

> **Educational prototype.** MedQuery does not diagnose, treat or prescribe, and its
> output is not verified medical advice.

> **The knowledge base is not included.** The app ships with a retrieval *interface* and
> an adapter that reports `NOT_CONFIGURED`. Until you connect your own knowledge base,
> every query says that evidence retrieval is unavailable. It never shows a made-up answer
> or citation. See [Connecting your knowledge base](#connecting-your-knowledge-base).

---

## Contents

1. [Architecture](#architecture)
2. [Technology stack](#technology-stack)
3. [Setup and running](#setup-and-running)
4. [What BERT does in MedQuery](#what-bert-does-in-medquery)
5. [API reference](#api-reference)
6. [Answer statuses](#answer-statuses)
7. [Configuration](#configuration)
8. [Connecting your knowledge base](#connecting-your-knowledge-base)
9. [Testing](#testing)
10. [Known limitations](#known-limitations)

---

## Architecture

```
            ┌──────────────────────────── backend/ (FastAPI) ─────────────────────────────┐
 Browser    │                                                                              │
 frontend/ ─┼─► POST /api/query ─► answer_processing/pipeline.py                           │
            │                        │                                                     │
            │                        ├─1─► query_processing/                               │
            │                        │       preprocess.py   clean, normalise, keywords,   │
            │                        │                       question type, safety flags   │
            │                        │       bert_encoder.py BERT WordPiece tokens +       │
            │                        │                       768-d contextual embedding    │
            │                        │                                                     │
            │                        ├─2─► retrieval/service.py  timeout, errors, filtering│
            │                        │       └─► YOUR ADAPTER (retrieval.base.Retriever)   │
            │                        │           default: NotConfiguredRetriever           │
            │                        │                                                     │
            │                        ├─3─► EVIDENCE GATE: no usable passages → no answer   │
            │                        │                                                     │
            │                        └─4─► answer_processing/generators.py                 │
            │                                extractive BERT-QA span │ evidence-only       │
            │                                                                              │
            │   app/storage.py  SQLite query history  (GET /api/query/{id}, /api/history)  │
            └──────────────────────────────────────────────────────────────────────────────┘
```

```
medquery/
├── .env.example                 configuration template (no secrets)
├── README.md
├── frontend/                    static UI, no build step, served by the backend
│   ├── index.html
│   ├── css/styles.css           light/dark theme, responsive
│   ├── js/app.js                UI behaviour
│   ├── js/api.js                REST client
│   ├── js/format.js             pure helpers (escaping, validation, status labels)
│   ├── package.json             test script + jsdom dev dependency
│   └── tests/                   node --test unit + DOM interaction tests
└── backend/
    ├── requirements.txt         web app + test dependencies
    ├── requirements-ml.txt      torch + transformers (optional)
    ├── pytest.ini
    ├── app/                     FastAPI app, config, routes, storage, model loading
    ├── query_processing/        preprocessing + BERT encoder
    ├── retrieval/               interface, NOT_CONFIGURED adapter, adapter template
    │   └── adapters/            ← put your knowledge-base adapter here
    ├── answer_processing/       evidence gate, answer generators, response models
    └── tests/                   pytest suite (mocks only)
```

`hello.py` and the nested `Medquery/` clone in the workspace root were already there and are untouched.

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| API | **FastAPI 0.115** + Uvicorn | Async framework with input validation and auto-generated docs at `/api/docs` |
| Validation / config | **Pydantic 2**, pydantic-settings | Typed request/response models; settings come from env vars or `.env` |
| NLP | **Hugging Face Transformers 4.46** + **PyTorch 2.5 (CPU)** | Standard way to load pretrained BERT; CPU wheels avoid a GPU requirement |
| Query encoder | `bert-base-uncased` (configurable) | General BERT; can swap to BioBERT / BiomedBERT |
| Answer extraction | `deepset/bert-base-cased-squad2` (configurable) | BERT fine-tuned on SQuAD 2.0, which can also say "no answer here" |
| Storage | SQLite (Python standard library) | Query history with no extra service |
| Frontend | HTML + CSS + JavaScript ES modules | No build tooling; served directly by FastAPI |
| Tests | pytest, Node's built-in test runner, jsdom | Run offline, no knowledge base or paid API needed |

## Setup and running

Requirements: **Python 3.10+** (tested on 3.12), and optionally **Node 18+** for the frontend tests.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate                     # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
# Optional but recommended: BERT models (CPU-only torch is ~200 MB)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.46.3

copy ..\.env.example ..\.env               # optional; defaults work without it

uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**. The backend serves both the UI and the API, so there is
nothing separate to start for the frontend. Interactive API docs are at **http://localhost:8000/api/docs**.

**First query:** model weights download from the Hugging Face Hub on first use and are
cached under `~/.cache/huggingface`: `bert-base-uncased` is about 440 MB and the QA model
about 430 MB. With the knowledge base unconfigured, only the query encoder is loaded; the
QA model loads the first time retrieval returns evidence. Set `MEDQUERY_PRELOAD_MODELS=true`
to load both in the background at startup, or `MEDQUERY_HF_OFFLINE=true` to use cached
weights only.

**Without torch/transformers** the app still runs. The encoder reports `unavailable`, queries
use keyword-only processing, and when evidence is present the app shows it without
extracting an answer (`EVIDENCE_ONLY`).

## What BERT does in MedQuery

BERT (Bidirectional Encoder Representations from Transformers) is used in two separate places:

1. **Query understanding** (`query_processing/bert_encoder.py`). The question is split into
   WordPiece tokens (`[CLS] what are the common symptoms of ... [SEP]`) and passed through the
   pretrained encoder. Each token gets a 768-dimensional vector that depends on the words
   around it. MedQuery mean-pools those vectors (using the attention mask) into one
   normalised **question embedding**. That embedding goes to your knowledge-base adapter in
   `RetrievalQuery.embedding`, where it can drive semantic (dense-vector) search.
2. **Answer extraction** (`answer_processing/generators.py`). A BERT model fine-tuned for
   reading comprehension (SQuAD 2.0) reads the question with each retrieved passage and
   predicts the start and end of an answer span, with a confidence score. The answer is
   copied word for word from the evidence, so it cannot add claims that aren't in your
   knowledge base. The SQuAD 2.0 model can also predict "no answer", which MedQuery reports
   as `NO_ANSWER_IN_EVIDENCE`.

**What BERT does not do:** it is not a source of medical knowledge and it does not check
whether anything is medically correct. Without a knowledge base, MedQuery gives no answers at all.

## API reference

Every response uses one envelope:

```jsonc
{ "success": true,  "data": { ... } }
{ "success": false, "error": { "code": "validation_error", "message": "...", "details": [...] } }
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | App status, model load states, knowledge-base configuration, input limits |
| `POST` | `/api/query` | Run the full pipeline. Body: `{"question": "..."}` |
| `GET` | `/api/query/{id}` | Fetch a stored result (`404` if unknown, `400` if the id is malformed) |
| `GET` | `/api/history` | Recent queries (`id`, `question`, `status`, `created_at`) |
| `DELETE` | `/api/history` | Delete all stored queries |
| `GET` | `/api/docs` | Swagger UI |

**Validation:** the question must be 3–500 characters after whitespace normalisation and
must contain letters. The request body is capped at 8 KB (`413 payload_too_large`). Invalid
input returns `422 validation_error`. Unexpected failures return `500 internal_error` with no
internal details exposed.

**`POST /api/query` response (`data`):**

```jsonc
{
  "id": "f0b9564c012749bcb2e5610115fa0e4d",
  "question": "What are the common symptoms of [condition]?",
  "status": "KB_NOT_CONFIGURED",          // see "Answer statuses"
  "answer": null,                          // or { text, confidence, method, evidence_index, verified:false }
  "evidence": [],                          // passages + used_for_answer, highlight_start/end
  "sources": [],                           // de-duplicated { source_id, title, url }
  "retrieval": { "status": "NOT_CONFIGURED", "adapter": "not_configured",
                 "message": "No medical knowledge base is connected...", "passages_returned": 0 },
  "analysis": {
    "normalized_text": "what are the common symptoms of [condition]?",
    "keywords": ["common", "symptoms", "condition"],
    "question_type": "symptoms",
    "safety_flags": [],
    "bert": { "available": true, "token_count": 12, "wordpieces": ["[CLS]", "what", "..."],
              "truncated": false, "embedding_dim": 768 }
  },
  "notices": [{ "level": "warning", "code": "kb_not_configured", "message": "..." }],
  "timings_ms": { "understanding": 94.1, "retrieval": 0.4, "total": 94.5 },
  "created_at": "2026-10-27T17:00:00Z",
  "disclaimer": "MedQuery is an educational prototype..."
}
```

## Answer statuses

| Status | Meaning | Answer shown? |
|---|---|---|
| `ANSWERED` | Span extracted from evidence with confidence ≥ `QA_MIN_CONFIDENCE` | Yes, with evidence |
| `LOW_CONFIDENCE` | Span extracted, but confidence is below the threshold | Yes, with a warning |
| `NO_ANSWER_IN_EVIDENCE` | Evidence was found, but the QA model found no answer span in it | No; evidence shown |
| `EVIDENCE_ONLY` | Evidence was found; `evidence_only` mode, or the QA model is unavailable | No; evidence shown |
| `NO_EVIDENCE` | The knowledge base returned nothing relevant | No |
| `KB_NOT_CONFIGURED` | No knowledge base is connected | No |
| `RETRIEVAL_FAILED` | The adapter timed out or raised an error | No; UI offers retry |

Safety notices are added on top of the status: `possible_emergency` (urgent phrasing, so the user
is told to contact emergency services) and `personal_medical_advice` (requests for a personal
diagnosis or dose).

## Configuration

All settings are environment variables prefixed `MEDQUERY_`, read from the process environment
or from `.env` in the project root. See [.env.example](.env.example) for the full list with
comments. The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `MEDQUERY_QUERY_MODEL_NAME` | `bert-base-uncased` | Query encoder (any BERT-family model or local path) |
| `MEDQUERY_KB_ADAPTER` | *(empty)* | `module.path:ClassName` of your knowledge-base adapter |
| `MEDQUERY_KB_CONNECTION` | *(empty)* | Connection string passed to your adapter |
| `MEDQUERY_KB_TOP_K` | `5` | Passages requested and kept |
| `MEDQUERY_KB_TIMEOUT_SECONDS` | `10` | Retrieval timeout |
| `MEDQUERY_KB_MIN_RELEVANCE` | `0.0` | Discard passages scoring below this |
| `MEDQUERY_ANSWER_MODE` | `extractive` | `extractive` or `evidence_only` |
| `MEDQUERY_QA_MODEL_NAME` | `deepset/bert-base-cased-squad2` | Extractive QA model |
| `MEDQUERY_QA_MIN_CONFIDENCE` | `0.15` | Below this, the status is `LOW_CONFIDENCE` |

No API keys are required. If you add an adapter that needs a secret, read it from an
environment variable inside the adapter and keep it in `.env`, which is git-ignored.

## Connecting your knowledge base

MedQuery only needs one class that implements `retrieval.base.Retriever`.

### 1. The contract (`backend/retrieval/base.py`)

**Input: `RetrievalQuery`**

| Field | Type | Notes |
|---|---|---|
| `original_text` | `str` | Cleaned question, original casing |
| `normalized_text` | `str` | Lower-cased, contractions expanded |
| `keywords` | `list[str]` | Stop-words removed; useful for BM25 / keyword search |
| `question_type` | `str` | `definition`, `symptoms`, `causes`, `treatment`, `prevention`, `diagnosis`, `prognosis`, `comparison`, `general` |
| `embedding` | `list[float] \| None` | L2-normalised BERT embedding (768-d for base models); `None` if BERT is unavailable |
| `top_k` | `int` | How many passages to return |

**Output: `RetrievalResult`**

| Field | Type | Notes |
|---|---|---|
| `status` | `RetrievalStatus` | `OK`, `NO_RESULTS` or `ERROR` (`TIMEOUT` and `NOT_CONFIGURED` are set by MedQuery) |
| `passages` | `list[Passage]` | Must be empty unless status is `OK` |
| `message` | `str \| None` | Optional explanation shown to the user |

**`Passage`:** `text` (required, used for answer extraction), `source_id` (required),
`source_title`, `source_url` (http/https only are rendered as links), `relevance_score`
(higher is better; 0–1 recommended), `metadata` (free-form dict).

**Rules:** return only real knowledge-base content; return `NO_RESULTS` when nothing matches;
raise an exception on failure (MedQuery turns it into `RETRIEVAL_FAILED`). `retrieve()` runs in
a worker thread with a timeout, so blocking I/O is fine.

### 2. Implement the adapter

Edit [backend/retrieval/adapters/custom_adapter_template.py](backend/retrieval/adapters/custom_adapter_template.py)
(or copy it to a new module in the same folder) and fill in `__init__` (open your index or
database from `self.connection`), `retrieve`, and optionally `close`.

### 3. Point MedQuery at it (`.env`)

```ini
MEDQUERY_KB_ADAPTER=retrieval.adapters.custom_adapter_template:MyKnowledgeBaseAdapter
MEDQUERY_KB_CONNECTION=path/or/url/to/your/knowledge/base
```

Restart the server. `GET /api/health` should now show
`"knowledge_base": {"configured": true, "adapter": "my_knowledge_base", ...}` and the status pill
in the UI turns green. If the adapter fails to import, the app still starts, logs the error,
and reports the knowledge base as not configured, with the reason.

### 4. Tune

- If your scores are on a 0–1 scale, set `MEDQUERY_KB_MIN_RELEVANCE` to filter weak matches.
- If your passages are long, split them into chunks of roughly 100–300 words. The QA model reads
  up to 384 tokens at a time, so shorter passages give better answer spans.
- If you use dense retrieval, embed your documents with the **same** model as
  `MEDQUERY_QUERY_MODEL_NAME` so the vectors are comparable. Mean-pool over the attention mask
  and L2-normalise, as `bert_encoder.py` does.
- If your knowledge base is biomedical text, try `MEDQUERY_QUERY_MODEL_NAME=dmis-lab/biobert-base-cased-v1.1`.
- To show evidence without extracting answers, set `MEDQUERY_ANSWER_MODE=evidence_only`.

### Files you may need to edit

| File | Why |
|---|---|
| `backend/retrieval/adapters/custom_adapter_template.py` (or a new file next to it) | **Your adapter. Usually the only code change.** |
| `.env` (copied from `.env.example`) | `MEDQUERY_KB_ADAPTER`, `MEDQUERY_KB_CONNECTION`, tuning values |
| `backend/requirements.txt` | Add any client library your knowledge base needs (e.g. `faiss-cpu`, `psycopg`, `elasticsearch`) |
| `backend/tests/` | Add tests for your adapter |
| `backend/app/config.py` | Only if your adapter needs extra typed settings |

## Testing

```powershell
# Backend: 60 tests, all mocked, no model download or knowledge base needed
cd backend
.venv\Scripts\python -m pytest

# Frontend: 17 tests (pure helpers + DOM interaction in jsdom)
cd frontend
npm install
npm test
```

What the tests cover: query preprocessing; request validation and error envelopes (422, 413,
404, 405, 500); the BERT encoder with a mocked tokenizer and model (lazy loading, pooling,
truncation, graceful failure); the `NOT_CONFIGURED` adapter; adapter loading from a dotted
path; timeouts and adapter exceptions; empty and filtered retrieval results; the pipeline with
and without evidence (and a check that the answer generator is never called without evidence);
low-confidence and no-answer cases; the extractive generator's span selection; persistence and
history; and UI behaviour (status pill, templates, validation, result rendering, HTML escaping,
`javascript:` URL blocking, error/retry, and history replay).

The test doubles live in `backend/tests/doubles.py` and the frontend DOM test. They use neutral
placeholder strings, not medical content, and are never loaded by the running app.

## Known limitations

- **No knowledge base is included.** Until you connect one, every query returns `KB_NOT_CONFIGURED`.
- **Extractive answers are short spans.** The QA model copies phrases from the evidence; it
  doesn't write summaries. Questions that need several facts combined may get
  `NO_ANSWER_IN_EVIDENCE` or a partial span.
- **Confidence is the model's own score,** not a measure of medical correctness. `verified` is always `false`.
- **Question-type and safety detection are keyword rules**, not a trained classifier. They can
  miss unusual phrasing or flag harmless questions.
- **CPU inference:** about 100 ms per query for the encoder on a laptop CPU. Extractive QA adds
  roughly 100–300 ms per passage. The first query is slow while weights download and load.
- **No login and no rate limiting.** History is shared by everyone using the server. This is
  fine for a local demo but not for public deployment.
- **No generative LLM.** Answer generation sits behind the `AnswerGenerator` interface in
  `answer_processing/generators.py`, so an optional LLM generator could be added later
  without changing the rest of the pipeline.
