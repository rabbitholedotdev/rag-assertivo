from eval.metrics import (
    abstained,
    content_recall_at_k,
    numbers_hallucination,
    primary_recall_at_k,
    synthetic_share,
)
from rag.documents import Chunk


def test_primary_recall_ignores_ementa():
    docs = [
        Chunk("ementa do lgpd", metadata={"doc_id": "ementa-lgpd", "source_type": "ementa"}),
        Chunk("dado pessoal: informação", metadata={"doc_id": "lgpd", "source_type": "lei"}),
    ]
    assert primary_recall_at_k(docs, "lgpd") == 1.0
    assert primary_recall_at_k(docs[:1], "lgpd") == 0.0
    assert synthetic_share(docs) == 0.5


def test_content_recall_shingles():
    gold = ["A saúde é direito de todos e dever do Estado, garantido mediante políticas sociais e econômicas."]
    hit = Chunk(gold[0][:80], metadata={"source_type": "lei"})
    miss = Chunk("selic juros mercado financeiro hoje", metadata={"source_type": "lei"})
    assert content_recall_at_k([hit], gold) == 1.0
    assert content_recall_at_k([miss], gold) == 0.0


def test_numbers_and_abstain():
    h = numbers_hallucination("prazo de 45 dias", "o prazo é de 15 dias")
    assert h["fora"] == ["45"]
    assert abstained("Não encontrei isso nos trechos recuperados.")
