from pathlib import Path

from ingest.chunk import ementa_of, load_corpus_file, split_articles
from ingest.secret_scan import assert_clean, findings


def test_split_articles_keeps_preamble_and_arts():
    body = "TÍTULO I\n\nArt. 1º Um.\n\nArt. 5º Cinco."
    parts = split_articles(body)
    labels = [p[0] for p in parts]
    assert "preambulo" in labels
    assert any(p.startswith("art-1") for p in labels)
    assert any(p.startswith("art-5") for p in labels)


def test_load_vendored_lgpd():
    path = Path("corpus/texts/lgpd.md")
    docs = load_corpus_file(path)
    assert docs
    assert docs[0]["metadata"]["doc_id"] == "lgpd"
    assert any("dado pessoal" in d["page_content"].lower() for d in docs)
    em = ementa_of(docs)
    assert em["metadata"]["source_type"] == "ementa"


def test_secret_scan_catches_key():
    assert findings("token sk-" + "a" * 24)
    assert_clean("A saúde é direito de todos.")


def test_ingest_dry_run():
    from ingest.ingest import main
    assert main(["--dry-run"]) == 0


def test_secret_scan_refuses(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        assert_clean("api_key = 'supersecretvalue'", "x")
