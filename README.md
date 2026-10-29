# MedQuery

**An Intelligent Medical Question Answering System Using BERT and Knowledge Retrieval**

MedQuery takes a medical question in plain language, processes it with a pretrained
BERT model, retrieves evidence from a connected medical knowledge base, and extracts
an answer only when that evidence supports one. Every answer is shown next to the
passage it came from and that passage's source.

> **Educational prototype.** MedQuery does not diagnose, treat or prescribe, and its
> output is not verified medical advice.

> **Knowledge base status:** 25 attributed source summaries, 63 processed chunks, and the FAISS
> index files are present under `backend/data/`. The processed chunks and manifests have matching
> corpus fingerprints. Loading and validating the vector file requires runtime dependencies.

---

## Contents

1. [Architecture](#architecture)
2. [Technology stack](#technology-stack)
3. [Setup and running](#setup-and-running)
4. [What BERT does in MedQuery](#what-bert-does-in-medquery)
5. [API reference](#api-reference)
6. [Answer statuses](#answer-statuses)
7. [Configuration](#configuration)
8. [Building the medical knowledge base](#building-the-medical-knowledge-base)
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
            │                        │       └─► FaissRetriever                            │
            │                        │           sentence embeddings → persisted FAISS    │
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
    ├── requirements-ml.txt      Torch, Transformers, sentence-transformers, FAISS (optional)
    ├── pytest.ini
    ├── app/                     FastAPI app, config, routes, storage, model loading
    ├── query_processing/        preprocessing + BERT encoder
    ├── retrieval/               interfaces, ingestion, FAISS adapter
    │   └── adapters/            built-in FAISS adapter + custom template
    ├── sources/                 attributed source files + source manifest
    ├── data/processed_medical_corpus/ generated chunks and ingestion report (git-ignored)
    ├── data/medical_kb/          generated FAISS index + matching metadata (git-ignored)
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

Requirements: **Python 3.10–3.12** for the pinned dependencies (the existing project was
tested on 3.12), and optionally **Node 18+** for the frontend tests. Newer Python versions
may require updating the pinned Pydantic and ML dependencies.

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

**First query:** models use the local Hugging Face cache by default. With built-in FAISS, the
separate BERT query vector is skipped because FAISS uses its sentence-embedding model; the QA
model loads the first time retrieval returns evidence. Set `MEDQUERY_PRELOAD_MODELS=true`
to load the QA model at startup (and the query encoder only for adapters that require it), or
`MEDQUERY_HF_OFFLINE=true` to use cached weights only.

**Without torch/transformers** the app still runs. BERT-based query vectors are optional for
adapters; when the extractive QA model is unavailable and evidence is present, the app shows
the evidence without extracting an answer (`EVIDENCE_ONLY`).

## What BERT does in MedQuery

BERT (Bidirectional Encoder Representations from Transformers) is used in two separate places:

1. **Query understanding** (`query_processing/bert_encoder.py`). The question is split into
   WordPiece tokens (`[CLS] what are the common symptoms of ... [SEP]`) and passed through the
   pretrained encoder. Each token gets a 768-dimensional vector that depends on the words
   around it. MedQuery mean-pools those vectors (using the attention mask) into one
   normalised **question embedding**. Custom adapters may use it; built-in FAISS retrieval
   uses the separate sentence embedding model recorded with the index. These vector spaces
   are not interchangeable.
2. **Answer extraction** (`answer_processing/generators.py`). A BERT model fine-tuned for
   reading comprehension (SQuAD 2.0) reads the question with each retrieved passage and
   predicts the start and end of an answer span, with a model score that is not calibrated
   as probability or medical confidence. The answer is
   copied from the evidence rather than composed, though passage selection can still be wrong.
   The SQuAD 2.0 model can also predict "no answer", which MedQuery reports
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
  "retrieval": { "status": "NOT_CONFIGURED", "adapter": "faiss",
                 "message": "No valid FAISS knowledge index is available.", "passages_returned": 0 },
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
| `EMERGENCY_GUIDANCE` | A possible emergency was detected; retrieval and answer extraction were skipped | No |

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
| `MEDQUERY_KB_ADAPTER` | FAISS adapter | `module.path:ClassName` of your knowledge-base adapter |
| `MEDQUERY_KB_CONNECTION` | `backend/data/medical_kb` | Index directory passed to the adapter |
| `MEDQUERY_KB_ALLOW_MODEL_DOWNLOAD` | `false` | Allow a query-time download of the index's embedding model |
| `MEDQUERY_KB_HYBRID_RETRIEVAL` | `true` | Enable built-in BM25 lexical ranking fused with FAISS dense ranking |
| `MEDQUERY_KB_TOP_K` | `5` | Passages requested and kept |
| `MEDQUERY_KB_TIMEOUT_SECONDS` | `10` | Retrieval timeout |
| `MEDQUERY_KB_MIN_RELEVANCE` | `0.65` | Discard passages scoring below this cosine-derived ranking floor |
| `MEDQUERY_ANSWER_MODE` | `extractive` | `extractive` or `evidence_only` |
| `MEDQUERY_QA_MODEL_NAME` | `deepset/bert-base-cased-squad2` | Extractive QA model |
| `MEDQUERY_QA_MIN_CONFIDENCE` | `0.15` | Below this, the status is `LOW_CONFIDENCE` |

No API keys are required. If you add an adapter that needs a secret, read it from an
environment variable inside the adapter and keep it in `.env`, which is git-ignored.

The source/processed corpus paths are ingestion CLI arguments (`--input-dir`, `--manifest`, and
`--output-dir`), not API settings. `MEDQUERY_KB_CONNECTION` points to the built index directory;
FAISS metadata lives beside the vector index as `chunks.json`, so there is no separate metadata
path setting. Choose the embedding model and optional immutable revision with the index builder's
`--model` and `--revision` flags. The selected model and revision are saved in `index_manifest.json`
and loaded by the adapter; BERT's `MEDQUERY_QUERY_MODEL_NAME` does not select the retrieval model.
The currently selected default is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
a general-purpose English sentence-similarity model with 384-dimensional output vectors. It is
used to rank text passages, not to supply medical knowledge.

## Building the medical knowledge base

The tracked source collection currently has **25 ready source summaries** in
`backend/sources/` (plus a folder README explicitly excluded by the manifest), according to
`backend/sources/sources.json`.
Topics include influenza and common cold; asthma and pneumonia; diabetes, blood pressure,
A1C, anemia, kidney and thyroid disease; GERD, diarrhea and constipation; allergy, migraine,
arthritis and common skin conditions; vaccines, screening and infection prevention; medication
safety; emergency warning signs; anxiety, depression, stress and sleep. The source organizations
are the U.S. National Library of Medicine's MedlinePlus, the U.S. Food and Drug Administration for
medication-safety material, and WHO/UNICEF for community hand hygiene. Date fields are included
only where a source date was verified.

The summaries are concise paraphrases with source links and attribution. MedlinePlus identifies
health-topic summaries and medical-test information as public-domain content in its
[content-use guidance](https://www.medlineplus.gov/about/using/usingcontent/); medical-encyclopedia
reuse terms are called out separately in the manifest. The WHO/UNICEF hand-hygiene guidance uses
a CC BY-NC-SA 3.0 IGO license, recorded with attribution and reuse notes. Review each source's
terms before reuse outside this project. Manifest fields distinguish verified source pages and
prepared summaries from ingestion and indexing. **The current working tree also contains 63
processed chunks and matching persisted FAISS files**; the source manifest's per-document operation
status fields still say `not_ingested`/`not_indexed` and should be treated as stale labels. The
processed corpus and index metadata manifests have matching corpus fingerprints; the vector file
has not been loaded in this environment. There are no pending source documents. The manifest retains a pending
topic note for locally verified crisis-support directory information and broader mental-wellbeing
coverage. Emergency guidance is a limited general warning-sign summary, not a comprehensive
triage protocol. This is a starter corpus, not an evaluated medical reference collection.

To add a document, place a UTF-8 Markdown, text, or HTML source under the matching topic
directory. Add its relative path to `backend/sources/sources.json` under `documents`, with a
stable lowercase `document_id`, `status: "ready"`, title, organization, HTTP(S) URL, topic,
provenance, license/usage notes, and a `retrieved_at` date. Add `updated_date` or
`publication_date` only when verified. Use `status: "pending"` and a reason until provenance
and reuse terms are checked. Use `status: "excluded"` for explanatory files that must not enter
the corpus. Ingestion excludes both; only `status: "ready"` documents become chunks. The current
per-document ingestion/index status fields are stale; use the processed and index manifests to
confirm generated artifacts and their corpus fingerprint.

The ingestion command reads UTF-8 files beneath `backend/sources` and joins them to
`sources/sources.json` by relative path. It extracts and normalizes text, preserves Markdown
headings, groups sentences within paragraphs, and keeps sentences intact when splitting at the
configured character target is possible. Paragraphs and sentences are packed into chunks without
crossing section headings. The configured overlap is formed from complete text units where they
fit, so it may be smaller than the target when a sentence is longer. Retrieved passage text keeps
its section heading. A separate `embedding_text` adds the document title and section context for
semantic indexing; URLs and other provenance remain structured metadata, not medical passage
prose. Exact normalized whole-document duplicates are skipped and reported.
Stable chunk IDs derive from document ID, section heading, and chunk text. Each chunk retains source
path, document and chunk IDs, title, organization, primary URL, additional URLs, section, hashes,
dates, and the source-manifest metadata snapshot. Pending and excluded entries do not enter the
chunk list. The attribution-only `Source and limits` section is not embedded as medical evidence;
its URLs, provenance, license, and usage notes remain attached as metadata. Generated manifests
record ingestion and indexing operations.

From the repository's `backend` directory, process all 25 ready summaries with:

```powershell
python -m retrieval.ingest --input-dir sources --manifest sources\sources.json `
  --output-dir data\processed_medical_corpus --chunk-size 800 --overlap 120
```

This writes `chunks.json` (section-aware text and source metadata), `processed_manifest.json`
(count and deterministic corpus fingerprint), and `ingestion_report.json` (skipped, pending,
and excluded files) under `backend/data/processed_medical_corpus/`. These are **ingested chunks**;
they contain no generated embeddings yet. No source text is copied into a separate manual dataset.

### Retrieval and evidence flow

The index builder embeds each chunk's `embedding_text` (title, section heading, and source text)
with the selected Sentence Transformers model. The answer extractor receives the section-aware
`text` field; source metadata stays separate. It encodes the query's cleaned original text with that same model
and its tokenizer/pooling path; the BERT query-understanding vector is a separate representation
and is not used by built-in FAISS search. Both document and query vectors are L2-normalized.
`IndexFlatIP` then ranks inner products, equivalent to cosine similarity for normalized vectors.
For the current adapter, raw cosine scores in `[-1, 1]` are mapped monotonically to
`[0, 1]` as `(cosine + 1) / 2`; this is a ranking scale, not a probability, medical correctness
score, or calibrated confidence.

`MEDQUERY_KB_MIN_RELEVANCE` is applied by `RetrievalService` after search and before the answer
extractor. Its default `0.65` corresponds to raw cosine `0.30` under that mapping. This is a
provisional retrieval floor intended to let MedQuery abstain on weak matches; it has not been
calibrated against this corpus. Relevance scores must be tuned using the evaluation set before
being treated as useful operating thresholds. A score never establishes that a passage is
medically correct.

The adapter builds a lightweight BM25 lexical index once from persisted chunk metadata and text.
It ranks dense FAISS and lexical BM25 results independently, then combines ranks with Reciprocal
Rank Fusion (RRF); it does not add incompatible raw scores. Set
`MEDQUERY_KB_HYBRID_RETRIEVAL=false` to use FAISS-only retrieval. Lexical initialization or search
errors fall back to FAISS ranking. Query expansion is deterministic and limited to direct corpus
aliases such as flu/influenza, hypertension/high blood pressure, and hyperglycemia/high blood
sugar. The original question is preserved and used for dense retrieval and answer extraction;
aliases only help lexical matching and never imply a diagnosis or treatment.

For built-in FAISS, BERT query-vector generation is skipped because the adapter creates its own
compatible Sentence Transformers query embedding. Custom adapters retain BERT encoding by default
and can declare `requires_query_embedding = False` if they do not consume it. This avoids computing
and discarding a second query vector.

The retrieval service removes repeated overlap passages within one source section and retains the
longer context when it contains a shorter duplicate; separate source citations remain separate.
For a single intent question, extractive QA reads matching sections and abstains if none match.
Multi-intent and comparison questions abstain because the API returns one span, not a composed
multi-part answer. Personal medication and other individualized advice requests show retrieved
general information, if any, without running answer extraction. Every generated span must match
its passage exactly, and its original evidence index is preserved. For distinct sources, a narrow
near-tie check only flags likely direct polarity or numeric conflicts when the sentence context
substantially overlaps. This heuristic cannot detect general medical contradictions. Source
metadata stays structured and is never added as though it were medical passage text. The UI labels
the returned QA value as an uncalibrated model score.

### Retrieval evaluation set

`backend/retrieval/evaluation/medical_retrieval_questions.json` is a manually reviewable,
**prepared but unrun** set of representative queries. Each entry has a query, topic, expected
document IDs and section headings, a corpus-sufficiency label, and whether abstention is expected.
The annotations refer only to passages in the prepared source documents; they do not include
answer strings or claim retrieval success. Review the labels before using the file as a gold set.

Each record follows this shape:

```json
{
  "id": "a1c-period",
  "query": "What time period does an A1C test reflect?",
  "topic": "diagnostic_tests.diabetes",
  "relevant_passages": [
    {"document_id": "medlineplus-a1c-summary", "section_heading": "What is an A1C test?"}
  ],
  "sufficient_evidence": true,
  "abstain_expected": false
}
```

After an index is built, suitable retrieval metrics include:

- **Precision@k:** relevant retrieved documents among the first `k`, divided by `k`.
- **Recall@k:** relevant expected documents found among the first `k`, divided by the number of
  expected relevant documents.
- **Mean Reciprocal Rank (MRR):** average reciprocal rank of the first relevant result for each
  answerable query.
- **Abstention checks:** rate of correct abstentions on queries labeled insufficient and rate of
  missed evidence on answerable queries. Report these separately from ranking metrics.

For queries with no expected relevant document, ranking metrics are not defined; evaluate abstention
instead. These measures assess retrieval and coverage only, not clinical correctness or the QA
model's answer accuracy. No evaluation has been run.

The [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) says inputs beyond
256 word pieces are truncated. The ingestion default is 800 characters
with a 120-character overlap target to leave room for headings under that limit; character counts
only approximate tokenizer length. The builder records the model's maximum sequence length and
source chunk settings in the generated index manifest. Check token-length distributions after the
model files are available and before treating the chunk defaults as validated.

The manifest format looks like this:

```json
{
  "documents": {
    "cardiovascular/high_blood_pressure.md": {
      "document_id": "my-stable-document-id",
      "status": "ready",
      "title": "Verified title from the source page",
      "organization": "Source organization",
      "source_url": "https://example.org/original-page",
      "topic": "cardiovascular.health",
      "retrieved_at": "2026-10-28",
      "content_provenance": "Where the text came from and how it was prepared",
      "license": "Verified reuse terms",
      "usage_notes": "Attribution and limits"
    }
  }
}
```

The URL and values above are schema examples only, not medical sources. Files missing required
attribution are skipped and listed in the ingestion report.

The optional ML/index dependencies are declared in `requirements-ml.txt` in the backend
directory (`sentence-transformers`, PyTorch, and `faiss-cpu`). Install them separately when
ready; neither command below installs packages.

```powershell
cd backend
pip install -r requirements-ml.txt
```

After ingestion, build or rebuild the separate FAISS index from the backend directory:

```powershell
python -m retrieval.build_index --processed-chunks data\processed_medical_corpus\chunks.json `
  --output-dir data\medical_kb --model sentence-transformers/all-MiniLM-L6-v2
```

The embedding model is configurable with `--model MODEL_ID_OR_LOCAL_PATH`; optionally pass
`--revision COMMIT_OR_TAG` to pin it. It is independent of BERT query classification and the
extractive answer model. The build is cache-only by default; if weights are not cached, explicitly
add `--allow-model-download`. The API also uses its local cache by default; API-time downloads
require `MEDQUERY_KB_ALLOW_MODEL_DOWNLOAD=true`.

Indexing writes `index.faiss` (normalized vectors with inner-product/cosine ranking), `chunks.json`
(matching chunk/source metadata), and `index_manifest.json` (model, revision, dimension, count, and
corpus fingerprint) under `backend/data/medical_kb/`. It checks that processed chunks match their
manifest and refuses an empty corpus without deleting an existing index. Re-run ingestion after
source changes, then rebuild the index. Generated artifacts live under ignored `backend/data/`.

During FastAPI lifespan startup, `FaissRetriever` reads the persisted vectors and metadata once;
it does not re-embed the corpus. The sentence-embedding model loads lazily on the first
non-emergency query and encodes that query only, checking its dimension against the saved index.
Startup does not download the embedding model. If the index is built after the application has
started, restart the backend to load it. BERT query classification remains separate. Retrieved
passages keep document ID, title, source URL, text, section, organization, hashes, and
source-manifest metadata. The existing `/api/query` response exposes passage text and source
title/URL. `/api/health` reports whether the index is configured, why it is unavailable, and
whether the lazy embedding model has loaded. If the index, dependencies, or model are unavailable,
retrieval returns an error or not-configured status and the evidence gate produces no unsupported
answer.

Stages are distinct: the 25 source files are **prepared**; ingestion creates **chunks**; indexing
generates **embeddings** and a **persisted FAISS index**. The endpoint is wired to the retrieval
interface, but runtime integration with the persisted index has **not been verified** in this
workspace. The processed chunks and manifests' corpus fingerprints match; FAISS vector loading was
not verified.
No PostgreSQL service is needed; SQLite remains the query-history store.

Start the API from `backend` as described above. The default adapter loads the persisted index
from `backend/data/medical_kb` once when the application starts. `/api/query` sends retrieved
passage text and source metadata through the existing evidence gate and extractive answer
pipeline. If the index is missing or retrieval fails, MedQuery abstains. Retrieved evidence
includes the document title, organization, URL, section, chunk ID, and ranking score. Scores
rank passages; they do not establish medical correctness. Set
`MEDQUERY_ANSWER_MODE=evidence_only` to display evidence without answer extraction.

Custom backends can still implement `retrieval.base.Retriever`; see
`backend/retrieval/adapters/custom_adapter_template.py` and configure
`MEDQUERY_KB_ADAPTER=package.module:ClassName`.

## Runtime checks

Run the backend from `backend/` after dependencies and local model weights are available:

```powershell
uvicorn app.main:app --reload --port 8000
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/query `
  -ContentType 'application/json' -Body '{"question":"What are common symptoms of influenza?"}'
```

The manual review set is `backend/retrieval/evaluation/medical_retrieval_questions.json`.
Later evaluation can calculate Precision@k and Recall@k against its relevant document/section
labels, and Mean Reciprocal Rank (MRR) from the first relevant passage. Also record abstention
behavior and source attribution for insufficient-evidence questions. These labels have not been
run against retrieval and should be reviewed before treating them as a gold set.

The current environment has no FAISS, Sentence Transformers, PyTorch, Transformers, FastAPI, or
pytest installation, so no cold/warm latency or influenza answer result is claimed. The 73-second
handover sample includes cold starts; measure warm requests separately after dependencies and
cached models are available.

### Measuring cold and warm query latency

Start the server in a fresh process using the normal local setup commands above. In a second
PowerShell window, measure the first request (cold model/index use) and then repeated requests
(warm process) against the same question. The API response also includes `timings_ms` for query
understanding, retrieval, answer generation, and total processing; those timings exclude client
network and server startup time.

```powershell
$uri = 'http://127.0.0.1:8000/api/query'
$body = @{ question = 'What are common symptoms of influenza?' } | ConvertTo-Json
$cold = Measure-Command {
  $coldResult = Invoke-RestMethod -Method Post -Uri $uri -ContentType 'application/json' -Body $body
}
$cold.TotalMilliseconds
$coldResult.data.timings_ms

1..5 | ForEach-Object {
  $warm = Measure-Command {
    $warmResult = Invoke-RestMethod -Method Post -Uri $uri -ContentType 'application/json' -Body $body
  }
  [pscustomobject]@{ Request = $_; ClientMs = [math]::Round($warm.TotalMilliseconds, 1); ServerMs = $warmResult.data.timings_ms.total }
}
```

Record cold and warm values separately and repeat with a fresh server process for another cold
sample. Do not interpret latency alone as an accuracy improvement; review the cited answer and
evidence against the prepared evaluation set as well.

## Testing

```powershell
# Backend: mocked tests; no model download or medical corpus needed
cd backend
.venv\Scripts\python -m pytest

# Frontend: 17 tests (pure helpers + DOM interaction in jsdom)
cd frontend
npm install
npm test
```

What the tests cover: query preprocessing; request validation and error envelopes (422, 413,
404, 405, 500); the BERT encoder with a mocked tokenizer and model (lazy loading, pooling,
truncation, graceful failure); ingestion/chunking/metadata; FAISS source attribution and
missing-index behavior; the `NOT_CONFIGURED` adapter; adapter loading from a dotted path;
timeouts and adapter exceptions; empty and filtered retrieval results; the pipeline with
and without evidence (and a check that the answer generator is never called without evidence);
low-confidence and no-answer cases; the extractive generator's span selection; persistence and
history; and UI behaviour (status pill, templates, validation, result rendering, HTML escaping,
`javascript:` URL blocking, error/retry, and history replay).

The test doubles live in `backend/tests/doubles.py` and the frontend DOM test. They use neutral
placeholder strings, not medical content, and are never loaded by the running app.

## Known limitations

- **The starter corpus is small and unevaluated.** It is largely English and U.S.-source-centered; broader wellbeing coverage and a locally verified crisis-support directory remain pending.
- **The evaluation dataset is prepared but has not been run.** The default relevance floor and disagreement heuristic are provisional; neither is calibrated on this corpus.
- **The embedding model truncates long text.** The configured character chunk limit only approximates its tokenizer limit, so chunk token lengths still need review with the model available.
- **Runtime retrieval is unverified.** Processed chunks and persisted FAISS files are present and manifest fingerprints match, but dependencies are absent in the current environment, so loading and querying have not been exercised here.
- **Pinned dependencies were not installable in the current Python 3.14 environment.** The pinned Pydantic Core source build failed with the available Windows linker; use Python 3.10–3.12 or update the dependency pins and validate them.
- **Extractive answers are short spans.** Comparison and multi-intent questions abstain because
  the response contract cannot provide a complete composed answer.
- **Confidence is the model's own score,** not a measure of medical correctness. `verified` is always `false`.
- **Question-type and safety detection are keyword rules**, not a trained classifier. They can
  miss unusual phrasing or flag harmless questions.
- **Latency is not measured here.** Skipping the unused BERT query vector and limiting QA inference
  to intent-matching passages are intended efficiency improvements, but their effect needs
  warm-request measurements with dependencies and model weights available.
- **No login and no rate limiting.** History is shared by everyone using the server. This is
  fine for a local demo but not for public deployment.
- **No generative LLM.** Answer generation sits behind the `AnswerGenerator` interface in
  `answer_processing/generators.py`, so an optional LLM generator could be added later
  without changing the rest of the pipeline.
