from rag.numeros import numeros_nao_encontrados as nne


def test_pontuacao_colada_nao_e_numero_novo():
    assert nne("veja o art. 312, e a lei de 19.12.2003.", "art. 312 do CPP e 19.12.2003") == []


def test_separador_de_milhar_diferente():
    assert nne("Lei 9.868", "Lei 9868") == []


def test_ano_de_quatro_digitos_contra_forma_abreviada():
    assert nne("art. 27 da Lei 9.868/1999", "Lei 9.868/99, art. 27") == []
    assert nne("Lei de 2005", "Lei 9.868/99") == ["2005"]


def test_numero_inventado_continua_acusado():
    assert nne("prazo de 45 dias e 30%", "o prazo é de 15 dias") == ["45", "30%"]
