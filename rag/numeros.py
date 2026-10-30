"""Numbers in the answer that never appear in the source (anti-invention guard).

Format-tolerant: trailing punctuation, thousand-separator variants, and a 4-digit
year written abbreviated in the source (`1999` vs `9.868/99`) are not counted as
missing. Invented figures still are.
"""
import re

_NUM = re.compile(r"\d[\d.,]*\s*%?")


def _sem_sep(s: str) -> str:
    return re.sub(r"\s+", "", s).replace(".", "").replace(",", "")


def numeros_nao_encontrados(answer: str, fonte: str) -> list[str]:
    """Numbers from `answer` that are absent in `fonte`. Does not count as missing:

    - punctuation glued to the token (`art. 312,`, `19.12.2003.`);
    - a different thousand/date separator (`9.868` vs `9868`);
    - a 4-digit year the source writes abbreviated (`1999` vs `9.868/99`).
    """
    alvo = _sem_sep(fonte)
    bruto = re.sub(r"\s+", "", fonte)
    achados: dict[str, None] = {}
    for n in _NUM.findall(answer):
        tok = re.sub(r"[.,]+$", "", n.strip())
        digitos = _sem_sep(tok)
        if not digitos:
            continue
        if digitos in alvo:
            continue
        if re.fullmatch(r"(19|20)\d\d", digitos) and f"/{digitos[2:]}" in bruto:
            continue
        achados[tok] = None
    return list(achados)
