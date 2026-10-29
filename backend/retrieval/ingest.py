"""Create a processed chunk corpus from attributed Markdown, text, or HTML.

Source metadata belongs in a JSON manifest keyed by input filename. The tool
never guesses a URL, organization, or update date.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger("medquery.ingest")
SUPPORTED = {".md", ".txt", ".html", ".htm"}
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_fingerprint(chunks: list[dict[str, Any]]) -> str:
    """Fingerprint the ordered document/chunk content, independent of run time."""
    return _hash("\n".join(
        f"{chunk['document_id']}:{chunk['chunk_id']}:{chunk['content_hash']}"
        for chunk in chunks
    ))


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "noscript"}:
            self.skip_depth += 1
        if tag in {"h1", "h2", "h3", "h4"}:
            self.parts.append("\n\n# ")
        elif tag in {"p", "div", "li", "section", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "section"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def extract_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _HTMLText()
        parser.feed(raw)
        raw = "".join(parser.parts)
    raw = html.unescape(raw).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in raw.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized or (lines and lines[-1]):
            lines.append(normalized)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _sections(text: str) -> list[tuple[str, str]]:
    result: list[tuple[str, list[str]]] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        found = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if found:
            if body:
                result.append((heading, body))
            heading, body = found.group(1).strip(), []
        else:
            body.append(line)
    if body:
        result.append((heading, body))
    return [(h, "\n".join(b).strip()) for h, b in result if "\n".join(b).strip()]


def _split_oversized_sentence(sentence: str, size: int) -> list[str]:
    """Split only sentences that cannot fit, preferring word boundaries."""
    pieces: list[str] = []
    remaining = sentence.strip()
    while len(remaining) > size:
        cut = remaining.rfind(" ", 0, size + 1)
        if cut < size // 2:
            cut = size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _section_units(section: str, size: int) -> list[str]:
    """Keep paragraphs whole and split only oversized paragraphs at sentences."""
    units: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", section) if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            units.append(paragraph)
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        for sentence in sentences or [paragraph]:
            units.extend(_split_oversized_sentence(sentence, size))
    return units


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[tuple[str, str]]:
    """Return section-aware chunks, keeping paragraph/sentence boundaries where possible.

    Overlap is assembled from complete preceding text units up to the configured
    target. A long indivisible sentence may have less overlap to avoid duplicating
    a fragment that would lose its qualifications.
    """
    if size < 100 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be >=100 and overlap must be in [0, size)")
    output: list[tuple[str, str]] = []
    for heading, section in _sections(text):
        units = _section_units(section, size)
        start = 0
        while start < len(units):
            end = start
            used = 0
            while end < len(units):
                added = len(units[end]) + (2 if end > start else 0)
                if used + added > size and end > start:
                    break
                used += added
                end += 1
            piece = "\n\n".join(units[start:end]).strip()
            if piece:
                output.append((heading, piece))
            next_start = end
            overlap_used = 0
            cursor = end - 1
            while cursor > start:
                added = len(units[cursor]) + (2 if overlap_used else 0)
                if overlap_used + added > overlap:
                    break
                overlap_used += added
                next_start = cursor
                cursor -= 1
            start = next_start if next_start > start else end
    return output


def load_documents(input_dir: Path, manifest_path: Path | None, size: int, overlap: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {input_dir}")
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {manifest_path}")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else {}
    if not isinstance(raw_manifest, dict):
        raise ValueError("source manifest must be a JSON object")
    manifest = raw_manifest.get("documents", raw_manifest)
    if not isinstance(manifest, dict):
        raise ValueError("source manifest 'documents' must be a JSON object")
    chunks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_content: set[str] = set()
    seen_document_ids: set[str] = set()
    found_files: set[str] = set()
    paths = [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED]

    def source_order(path: Path) -> tuple[int, str]:
        rel = path.relative_to(input_dir).as_posix()
        meta = manifest.get(rel)
        has_provenance = (
            isinstance(meta, dict)
            and meta.get("status", "ready") == "ready"
            and all(str(meta.get(key, "")).strip() for key in ("source_url", "title", "organization"))
        )
        return (0 if has_provenance else 1, rel)

    for path in sorted(paths, key=source_order):
        rel = path.relative_to(input_dir).as_posix()
        found_files.add(rel)
        try:
            meta = manifest.get(rel, {})
            if not isinstance(meta, dict):
                raise ValueError("manifest entry must be a JSON object")
            status = meta.get("status", "ready")
            if status == "excluded":
                errors.append({"file": rel, "status": "excluded", "reason": str(meta.get("reason", "Excluded by source manifest."))})
                continue
            if status != "ready":
                errors.append({"file": rel, "status": "pending", "reason": str(meta.get("reason", "Source is not approved for ingestion."))})
                continue
            text = extract_text(path)
            if len(text) < 40:
                raise ValueError("document contains too little readable text")
            content_hash = _hash(text)
            if content_hash in seen_content:
                errors.append({"file": rel, "status": "skipped_duplicate", "reason": "duplicate content skipped"})
                continue
            if not all(str(meta.get(key, "")).strip() for key in ("source_url", "title", "organization")):
                raise ValueError("manifest must provide title, organization, and source_url")
            url = str(meta["source_url"])
            parsed_url = urlsplit(url)
            if parsed_url.scheme not in {"https", "http"} or not parsed_url.hostname or any(ch.isspace() for ch in url):
                raise ValueError("source_url must use http or https")
            doc_id = str(meta.get("document_id") or _hash(url + "\n" + rel)[:24])
            if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,79}", doc_id):
                raise ValueError("document_id must be a stable lowercase identifier (3-80 letters, digits, . _ -)")
            if doc_id in seen_document_ids:
                raise ValueError(f"duplicate document_id: {doc_id}")
            seen_document_ids.add(doc_id)
            seen_content.add(content_hash)
            used_chunk_ids: set[str] = set()
            for section, chunk in chunk_text(text, size, overlap):
                if section.casefold() in {"source and limits", "source and attribution"}:
                    # Attribution and reuse notes are retained in structured
                    # metadata, not indexed as if they were medical evidence.
                    continue
                chunk_text_with_heading = f"## {section}\n\n{chunk}" if section else chunk
                embedding_text = f"{meta['title']}\n{section}\n{chunk}" if section else f"{meta['title']}\n{chunk}"
                chunk_id = f"{doc_id}-{_hash((section or '') + chr(10) + chunk)[:16]}"
                if chunk_id in used_chunk_ids:
                    occurrence = 2
                    while f"{chunk_id}-{occurrence}" in used_chunk_ids:
                        occurrence += 1
                    chunk_id = f"{chunk_id}-{occurrence}"
                used_chunk_ids.add(chunk_id)
                chunks.append({
                    "document_id": doc_id, "chunk_id": chunk_id, "text": chunk_text_with_heading,
                    "embedding_text": embedding_text,
                    "title": str(meta["title"]), "organization": str(meta["organization"]),
                    "source_url": url, "topic": meta.get("topic"), "section_heading": section or None,
                    "publication_date": meta.get("publication_date"), "updated_date": meta.get("updated_date"),
                    "retrieved_at": meta.get("retrieved_at"),
                    "content_provenance": meta.get("content_provenance"), "license": meta.get("license"),
                    "usage_notes": meta.get("usage_notes"),
                    "source_path": rel,
                    # Operational stages are recorded in generated manifests, not
                    # copied as stale status claims from the prepared-source manifest.
                    "source_metadata": {key: value for key, value in meta.items()
                                        if key not in {"ingestion_status", "index_status"}},
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "document_content_hash": content_hash, "content_hash": _hash(embedding_text),
                })
        except Exception as exc:  # retain per-file failure details for reproducibility
            errors.append({"file": rel, "status": "skipped_error", "reason": str(exc)})
    for rel in sorted(set(manifest) - found_files):
        errors.append({"file": rel, "status": "missing_file", "reason": "Manifest entry has no matching source file."})
    return chunks, errors


def ingest_corpus(input_dir: Path, output_dir: Path, manifest_path: Path | None = None,
                  chunk_size: int = 800, overlap: int = 120) -> dict[str, Any]:
    chunks, errors = load_documents(input_dir, manifest_path, chunk_size, overlap)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = corpus_fingerprint(chunks)
    report = {
        "schema_version": 1, "chunk_count": len(chunks),
        "skipped_count": sum(e["status"] != "excluded" for e in errors),
        "excluded_count": sum(e["status"] == "excluded" for e in errors),
        "errors_and_pending": errors, "corpus_fingerprint": fingerprint,
    }
    manifest = {
        "schema_version": 1, "chunk_count": len(chunks), "corpus_fingerprint": fingerprint,
        "source_directory": input_dir.as_posix(), "chunk_size_chars": chunk_size,
        "overlap_chars": overlap, "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    # Processed text stays separate from vector/index artifacts.
    with tempfile.TemporaryDirectory(dir=output_dir) as staged:
        staged_dir = Path(staged)
        (staged_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        (staged_dir / "processed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staged_dir / "ingestion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for name in ("chunks.json", "processed_manifest.json", "ingestion_report.json"):
            (staged_dir / name).replace(output_dir / name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract approved source documents and write a processed chunk corpus.")
    parser.add_argument("--input-dir", type=Path, default=Path("sources"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_medical_corpus"))
    parser.add_argument("--manifest", type=Path, default=Path("sources/sources.json"))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(ingest_corpus(args.input_dir, args.output_dir, args.manifest, args.chunk_size, args.overlap), indent=2))


if __name__ == "__main__":
    main()
