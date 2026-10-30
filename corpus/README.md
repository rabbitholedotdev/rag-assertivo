# Public PT-BR corpus

Vendored excerpts of **official Brazilian legislation** (texts published by the
Presidency of the Republic at planalto.gov.br). Official legislative texts are
public. This tree is the only source the ingest script reads.

**Not in this corpus (and never will be):** paid courses, member-area scrapes,
private transcripts, YouTube dumps, or anything behind a login.

To refresh from the official HTML (best-effort, still review before ingest):

```bash
python -m ingest.fetch_planalto --dry-run
```

The vendored files are the source of truth for eval. The fetcher is a
convenience, not a license to scrape private sites.
