import json

from rag.answer import FAIL_CLOSED, NOT_FOUND, answer, format_text, parse_generation
from rag.documents import Chunk
from rag.generate import extractive_gen


def _doc(text, **meta):
    meta.setdefault("title", "Lei")
    meta.setdefault("source_type", "lei")
    meta.setdefault("url", "https://www.planalto.gov.br/")
    return Chunk(page_content=text, metadata=meta)


def test_parse_generation_json():
    raw = json.dumps({"frases": [{"texto": "O prazo é de 5 anos.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}]}]})
    fr = parse_generation(raw)
    assert fr[0]["texto"].startswith("O prazo")


def test_parse_generation_fenced_and_garbage():
    raw = 'blah ```json\n{"frases": [{"texto": "x", "provas": [{"trecho": 1, "citacao": "x"}]}]}\n```'
    assert parse_generation(raw)[0]["texto"] == "x"


def test_answer_cuts_invented_number():
    docs = [_doc("O prazo é de 5 anos, salvo fraude.")]

    def gen(_prompt, timeout=None):
        return json.dumps({
            "frases": [{
                "texto": "O prazo é de 15 anos.",
                "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}],
            }]
        })

    r = answer("qual o prazo?", docs, gen, None)
    assert r["status"] == "nao_encontrado"
    assert r["cortadas"][0]["motivo"] == "numero_sem_prova"
    assert NOT_FOUND in format_text(r)


def test_answer_keeps_proved_sentence():
    docs = [_doc("O prazo é de 5 anos, salvo fraude.")]

    def gen(_prompt, timeout=None):
        return json.dumps({
            "frases": [{
                "texto": "O prazo é de 5 anos.",
                "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}],
            }]
        })

    r = answer("qual o prazo?", docs, gen, None)
    assert r["status"] == "ok"
    assert "5 anos" in format_text(r)


def test_verifier_cuts_unsupported():
    docs = [_doc("O prazo é de 5 anos, salvo fraude.")]

    def gen(_prompt, timeout=None):
        return json.dumps({
            "frases": [{
                "texto": "O prazo é de 5 anos.",
                "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}],
            }]
        })

    def ver(_prompt, timeout=None):
        return json.dumps({"itens": [{"sustenta": False, "ressalva_omitida": False}]})

    r = answer("qual o prazo?", docs, gen, ver)
    assert r["status"] == "nao_encontrado"
    assert r["cortadas"][0]["motivo"] == "nao_sustenta"


def test_fail_closed_on_bad_json():
    docs = [_doc("qualquer")]

    def gen(_prompt, timeout=None):
        return "isto não é json"

    r = answer("x", docs, gen, None)
    assert r["status"] == "falha_verificacao"
    assert FAIL_CLOSED in format_text(r)


def test_extractive_stub_is_proved():
    docs = [_doc("A saúde é direito de todos e dever do Estado, garantido mediante políticas sociais.")]
    r = answer("a saúde é direito de quem?", docs, extractive_gen, None)
    assert r["status"] == "ok"
    assert r["frases"]
