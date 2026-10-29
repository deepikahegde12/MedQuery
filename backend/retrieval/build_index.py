"""Embed a processed corpus and persist its cosine-similarity FAISS index.

The model is loaded from the local cache by default. Pass --allow-model-download
explicitly to permit downloading a Hugging Face model during an index build.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval.ingest import DEFAULT_MODEL, corpus_fingerprint

logger = logging.getLogger("medquery.index")


def build_index(processed_chunks: Path, output_dir: Path, model_name: str = DEFAULT_MODEL,
                revision: str | None = None, allow_model_download: bool = False) -> dict[str, Any]:
    chunks = json.loads(processed_chunks.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("processed corpus must be a JSON array of chunks")
    corpus_manifest_path = processed_chunks.with_name("processed_manifest.json")
    if not corpus_manifest_path.is_file():
        raise FileNotFoundError(f"Processed manifest not found: {corpus_manifest_path}; run retrieval.ingest first")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not chunks:
        raise ValueError("Processed corpus is empty; existing index files were left unchanged")
    required = ("document_id", "chunk_id", "text", "embedding_text", "title", "organization", "source_url", "content_hash")
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not all(chunk.get(key) for key in required):
            raise ValueError(f"Processed chunk {position} is missing required text or source metadata")
    expected_fingerprint = corpus_manifest.get("corpus_fingerprint")
    actual_fingerprint = corpus_fingerprint(chunks)
    if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
        raise ValueError("Processed chunks do not match processed_manifest.json; rerun retrieval.ingest")
    if corpus_manifest.get("chunk_count") != len(chunks):
        raise ValueError("Processed chunk count does not match processed_manifest.json; rerun retrieval.ingest")

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install backend/requirements-ml.txt before building a FAISS index") from exc

    try:
        encoder = SentenceTransformer(
            model_name, revision=revision, local_files_only=not allow_model_download,
        )
    except Exception as exc:
        mode = "downloads are disabled; use a cached model or pass --allow-model-download" if not allow_model_download else "check model name, revision, and network access"
        raise RuntimeError(f"Could not load embedding model {model_name!r} ({mode}): {exc}") from exc
    try:
        model_dimension = encoder.get_sentence_embedding_dimension()
        vectors = encoder.encode(
            [str(chunk["embedding_text"]) for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not generate embeddings with {model_name!r}: {exc}") from exc
    vectors = np.asarray(vectors, dtype="float32")
    if (vectors.ndim != 2 or vectors.shape[0] != len(chunks) or
            (model_dimension is not None and vectors.shape[1] != int(model_dimension))):
        raise ValueError("embedding model returned an invalid vector matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding model returned non-finite values")

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    metadata = {
        "schema_version": 1,
        "embedding_model": model_name,
        "embedding_model_revision": revision,
        "dimension": int(vectors.shape[1]),
        "embedding_max_sequence_length": getattr(encoder, "max_seq_length", None),
        "metric": "cosine_via_normalized_inner_product",
        "chunk_count": len(chunks),
        "corpus_fingerprint": expected_fingerprint,
        "source_chunk_size_chars": corpus_manifest.get("chunk_size_chars"),
        "source_overlap_chars": corpus_manifest.get("overlap_chars"),
        "source_processed_manifest": str(corpus_manifest_path.name),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    # Stage the vector file and matching chunk metadata before replacing the
    # current index files. A failed embedding run leaves the previous index intact.
    with tempfile.TemporaryDirectory(dir=output_dir) as staged:
        staged_dir = Path(staged)
        faiss.write_index(index, str(staged_dir / "index.faiss"))
        (staged_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        (staged_dir / "index_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for name in ("index.faiss", "chunks.json", "index_manifest.json"):
            (staged_dir / name).replace(output_dir / name)
    return {"status": "built", "chunk_count": len(chunks), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-chunks", type=Path, default=Path("data/processed_medical_corpus/chunks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/medical_kb"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Optional immutable Hugging Face commit or tag for reproducible builds")
    parser.add_argument("--allow-model-download", action="store_true", help="Permit downloading model files from Hugging Face")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build_index(args.processed_chunks, args.output_dir, args.model, args.revision,
                                 args.allow_model_download), indent=2))


if __name__ == "__main__":
    main()
