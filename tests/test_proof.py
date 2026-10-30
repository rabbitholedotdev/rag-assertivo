from rag.proof import check_phrase, citation_found, number_tokens


def test_citacao_literal_com_espaco_caixa_e_aspas():
    assert citation_found('Prazo de “dois anos”', 'o  prazo de\n"dois anos" após')


def test_citacao_com_palavra_trocada_nao_casa():
    assert not citation_found("prazo de três anos", "o prazo de dois anos")


def test_citacao_vazia_nao_casa():
    assert not citation_found("   ", "qualquer coisa")


def test_numero_por_token():
    assert number_tokens("5 anos") == {"5"} and "5" not in number_tokens("15 anos")


def test_ordinal_barra_ponto_e_data():
    assert number_tokens("art. 2º da Lei 7.783/89 de 16.06.2015.") == {"2", "7.783/89", "16.06.2015"}


def test_extenso_ate_cem():
    assert number_tokens("cinco anos") == number_tokens("5 anos")
    assert number_tokens("cem dias") == {"100"}


def test_dinheiro_e_percentual():
    assert "1.500,00" in number_tokens("R$ 1.500,00") and "30" in number_tokens("30%")


def test_dispositivo_legal():
    assert number_tokens("art. 7º, XXIX") == {"7"}
    assert "1.500,00" not in number_tokens("R$ 1.500")


def test_check_phrase_motivos():
    chunks = ["O prazo é de 5 anos, salvo fraude."]
    ok = {"texto": "O prazo é de 5 anos.", "provas": [{"trecho": 1, "citacao": "prazo é de 5 anos"}]}
    assert check_phrase(ok, chunks) is None
    assert check_phrase({**ok, "texto": "O prazo é de 15 anos."}, chunks) == "numero_sem_prova"
    assert check_phrase({**ok, "provas": [{"trecho": 1, "citacao": "prazo de 5"}]}, chunks) == "citacao_ausente"
    assert check_phrase({**ok, "provas": [{"trecho": 9, "citacao": "prazo é de 5 anos"}]}, chunks) == "citacao_ausente"
    assert check_phrase({**ok, "provas": []}, chunks) == "citacao_ausente"


def test_citacao_nao_string_corta_so_a_frase():
    chunks = ["O prazo é de 5 anos."]
    for c in (None, 2024, ["x"]):
        assert check_phrase({"texto": "x", "provas": [{"trecho": 1, "citacao": c}]}, chunks) == "citacao_ausente"


def test_citacao_respeita_fronteira_de_palavra():
    assert not citation_found("prazo", "o contraprazo é de 5 dias")
    assert not citation_found("5 anos", "prazo de 15 anos")
    assert citation_found("prazo", "o prazo é de 5 dias")
    assert citation_found("5 anos.", "prazo de 5 anos.")


def test_numero_nao_casa_prefixo_de_numero_maior():
    assert not citation_found("1.500", "valor R$ 1.500.000")
    assert not citation_found("30", "prazo de 30,5 dias")
    assert not citation_found("500,00", "total 1.500,00")
    assert citation_found("1.500,00", "total 1.500,00 reais")
    assert citation_found("30 dias", "prazo de 30 dias.")
