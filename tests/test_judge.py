from eval.judge import faithfulness, judge_phrases, lexical_supported


def test_lexical_supported_high_on_copy():
    ctx = "A saúde é direito de todos e dever do Estado."
    assert lexical_supported("A saúde é direito de todos.", ctx) >= 0.5


def test_lexical_supported_low_on_unrelated():
    assert lexical_supported("A selic está em 13 por cento hoje.", "A saúde é direito de todos.") < 0.3


def test_judge_phrases_uses_proof_first():
    chunks = ["O prazo é de 5 anos, salvo fraude."]
    out = judge_phrases(
        "prazo",
        [("O prazo é de 15 anos.", [{"trecho": 1, "citacao": "prazo é de 5 anos"}])],
        chunks,
    )
    assert out[0]["veredito"] == "numero_sem_prova"
    assert out[0]["backend"] == "proof"


def test_faithfulness_dict():
    r = faithfulness("A saúde é direito de todos.", "A saúde é direito de todos e dever do Estado.")
    assert 0 <= r["score"] <= 1
    assert r["backend"] == "lexical"
