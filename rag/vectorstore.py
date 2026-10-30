"""Qdrant collection in HYBRID mode: dense (semantic) + sparse (BM25).

- Dense: fastembed + multilingual-e5-large (strong on PT-BR), unnamed vector `''`.
- Sparse: fastembed + Qdrant/bm25 (keyword), vector `langchain-sparse`.

Both run locally. Vector names follow langchain-qdrant so hybrid search works
without extra wiring.

`HF_HUB_DISABLE_SHARED_BLOBS=1` is required: sharded ONNX (e5-large, Jina reranker)
fails at runtime with "External data path escapes model directory" when the Hub
cache stores `.onnx` and `.onnx_data` as symlinks under different blob prefixes.
"""
from __future__ import annotations

import os

from rag.config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
    RAG_RERANK,
    RERANK_MODEL,
)
from rag.documents import Chunk
from rag.retrieve import SearchBackend

SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = ""
SPARSE_VECTOR_NAME = "langchain-sparse"

_dense = None
_sparse = None
_reranker = None  # None=not tried, False=unavailable, obj=loaded
_client = None


def get_reranker():
    """Cross-encoder for candidate reordering. Off with RAG_RERANK=0. None if missing."""
    global _reranker
    if _reranker is None:
        if not RAG_RERANK:
            _reranker = False
        else:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
            except Exception:  # noqa: BLE001 — hybrid search still runs without rerank
                _reranker = False
    return _reranker or None


def get_dense_embeddings():
    global _dense
    if _dense is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        providers = ["CUDAExecutionProvider"] if os.getenv("FASTEMBED_CUDA") == "1" else None
        kw = {"providers": providers} if providers else {}
        _dense = FastEmbedEmbeddings(model_name=EMBED_MODEL, **kw)
    return _dense


def get_sparse_embeddings():
    global _sparse
    if _sparse is None:
        from langchain_qdrant import FastEmbedSparse
        _sparse = FastEmbedSparse(model_name=SPARSE_MODEL)
    return _sparse


def get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    return _client


def _dense_size(embeddings) -> int:
    return len(embeddings.embed_query("dimensão"))


def ensure_collection(name: str | None = None) -> None:
    """Create the hybrid collection + payload indexes if missing."""
    from qdrant_client.http.models import (
        Distance,
        PayloadSchemaType,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
        SparseVectorParams,
        TextIndexParams,
        TokenizerType,
        VectorParams,
    )

    name = name or COLLECTION_NAME
    client = get_client()
    if client.collection_exists(name):
        return
    size = _dense_size(get_dense_embeddings())
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE, on_disk=True),
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
        ),
    )
    for field in ("metadata.area", "metadata.doc_id", "metadata.source_type", "metadata.article"):
        try:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=PayloadSchemaType.KEYWORD)
        except Exception:  # noqa: BLE001 — already exists
            pass
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="metadata.title",
            field_schema=TextIndexParams(
                type="text", tokenizer=TokenizerType.MULTILINGUAL,
                min_token_len=2, max_token_len=30, lowercase=True,
            ),
        )
    except Exception:  # noqa: BLE001
        pass
    print(f"Collection '{name}' created HYBRID ({size}d + BM25) + int8 quantization.")


def get_vectorstore(create: bool = True, collection: str | None = None, retrieval_mode=None):
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    collection = collection or COLLECTION_NAME
    if retrieval_mode is None:
        retrieval_mode = RetrievalMode.HYBRID
    dense = get_dense_embeddings()
    sparse = get_sparse_embeddings()
    client = get_client()
    if not client.collection_exists(collection):
        if not create:
            raise RuntimeError(f"Collection '{collection}' does not exist. Run ingest first.")
        ensure_collection(collection)
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=retrieval_mode,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def _filter(source_types, exclude_types):
    from qdrant_client.http import models as qmodels

    must, must_not = [], []
    if source_types:
        must.append(qmodels.FieldCondition(
            key="metadata.source_type",
            match=qmodels.MatchAny(any=[s.lower() for s in source_types]),
        ))
    if exclude_types:
        must_not.append(qmodels.FieldCondition(
            key="metadata.source_type",
            match=qmodels.MatchAny(any=[s.lower() for s in exclude_types]),
        ))
    if not must and not must_not:
        return None
    return qmodels.Filter(must=must or None, must_not=must_not or None)


class QdrantBackend:
    """SearchBackend over langchain-qdrant hybrid / dense / sparse modes."""

    def search(
        self,
        query: str,
        k: int,
        *,
        source_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
        mode: str = "hybrid",
        collection: str | None = None,
        exact: bool = False,
    ) -> list[tuple[Chunk, float]]:
        from langchain_qdrant import RetrievalMode
        from qdrant_client.http import models as qmodels

        modo = {
            "hybrid": RetrievalMode.HYBRID,
            "dense": RetrievalMode.DENSE,
            "sparse": RetrievalMode.SPARSE,
        }[mode]
        vs = get_vectorstore(create=False, collection=collection, retrieval_mode=modo)
        sp = qmodels.SearchParams(exact=True) if exact else None
        flt = _filter(source_types, exclude_types)
        pairs = vs.similarity_search_with_score(query, k=k, filter=flt, search_params=sp)
        out = []
        for doc, score in pairs:
            chunk = Chunk(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
            out.append((chunk, float(score)))
        return out
