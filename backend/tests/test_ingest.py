import json

from retrieval.ingest import chunk_text, extract_text, load_documents


def test_chunking_preserves_heading_and_overlap():
    text = "# Symptoms\n\n" + ("Common signs include fever and cough. " * 20)
    chunks = chunk_text(text, size=180, overlap=30)
    assert len(chunks) > 1
    assert all(heading == "Symptoms" for heading, _ in chunks)
    assert chunks[0][1][-30:] in chunks[1][1]


def test_html_extraction_discards_navigation_and_scripts(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<nav>menu noise</nav><h1>Article</h1><p>Useful medical information.</p><script>secret()</script>")
    body = extract_text(path)
    assert "Article" in body and "Useful medical information" in body
    assert "menu noise" not in body and "secret" not in body


def test_load_documents_keeps_manifest_metadata_and_skips_duplicate(tmp_path):
    docs = tmp_path / "sources"
    docs.mkdir()
    text = "# Overview\n\n" + ("This source passage contains enough attributable information. " * 2)
    (docs / "one.md").write_text(text)
    (docs / "copy.txt").write_text(text)
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"one.md": {
        "title": "Verified title", "organization": "Example Health Agency",
        "source_url": "https://example.org/health", "topic": "general",
        "publication_date": "2025-01-02",
    }}))
    chunks, errors = load_documents(docs, manifest, 500, 50)
    assert len(chunks) == 1
    assert chunks[0]["title"] == "Verified title"
    assert chunks[0]["organization"] == "Example Health Agency"
    assert chunks[0]["source_url"] == "https://example.org/health"
    assert chunks[0]["section_heading"] == "Overview"
    assert chunks[0]["publication_date"] == "2025-01-02"
    assert any("duplicate" in row["reason"] for row in errors)


def test_missing_provenance_is_reported_and_skipped(tmp_path):
    (tmp_path / "article.txt").write_text("A sufficiently long local passage should be skipped without source metadata.")
    chunks, errors = load_documents(tmp_path, None, 500, 50)
    assert chunks == []
    assert "manifest" in errors[0]["reason"]
