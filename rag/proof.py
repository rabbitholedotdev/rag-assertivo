"""Phrase-level proof, checked by code: the citation exists literally in the chunk
and every number in the sentence appears in its citations.

Fail-closed: a sentence without a passing citation never reaches the user.
"""
import re
import unicodedata

_ASPAS = str.maketrans({
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'", "´": "'", "`": "'",
})
_NUM = re.compile(r"(?<![\d.,/])\d+(?:[.,/]\d+)*(?!\d)")
_EXTENSO = {
    w: i
    for i, w in enumerate(
        "zero um dois três quatro cinco seis sete oito nove dez onze doze treze "
        "quatorze quinze dezesseis dezessete dezoito dezenove vinte".split()
    )
}
_EXTENSO.update(duas=2, quatorze=14, catorze=14)
# Drop "um": it is an article in most PT-BR sentences and would cut a correct phrase.
_EXTENSO.pop("um")
_EXTENSO.update({
    "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
    "setenta": 70, "oitenta": 80, "noventa": 90, "cem": 100,
})
_DEZENA = {k: v for k, v in _EXTENSO.items() if v in range(20, 100, 10)}
_UNIDADE = {k: v for k, v in _EXTENSO.items() if 1 <= v <= 9}


def normalize(s: str) -> str:
    """NFC, lowercase, straight quotes, any whitespace collapsed. No word is rewritten."""
    s = unicodedata.normalize("NFC", s).translate(_ASPAS).casefold()
    return re.sub(r"\s+", " ", s).strip()


def citation_found(citacao: str, chunk: str) -> bool:
    """Literal and whole: the citation must not start or end mid-word in the chunk."""
    c = normalize(citacao)
    if not c:
        return False
    ini = r"(?<!\w)" if c[0].isalnum() else ""
    fim = r"(?!\w)" if c[-1].isalnum() else ""
    if c[0].isdigit():
        ini += r"(?<!\d[.,/])"  # "500,00" must not match inside "1.500,00"
    if c[-1].isdigit():
        fim += r"(?![.,/]\d)"  # "30" must not match inside "30,5"
    return re.search(ini + re.escape(c) + fim, normalize(chunk)) is not None


def number_tokens(text: str) -> set[str]:
    t = normalize(text)
    out = {m.group().rstrip(".,/") for m in _NUM.finditer(t)}
    palavras = re.findall(r"[a-zçãéêíóôú]+", t)
    i = 0
    while i < len(palavras):
        w = palavras[i]
        if w in _DEZENA and i + 2 < len(palavras) and palavras[i + 1] == "e" and palavras[i + 2] in _UNIDADE:
            out.add(str(_DEZENA[w] + _UNIDADE[palavras[i + 2]]))
            i += 3
            continue
        if w in _EXTENSO and w != "zero":
            out.add(str(_EXTENSO[w]))
        i += 1
    return out


def check_phrase(frase: dict, chunks: list[str]) -> str | None:
    """None if the sentence is backed by code; otherwise the cut reason."""
    provas = frase.get("provas") or []
    if not provas:
        return "citacao_ausente"
    citacoes = []
    for p in provas:
        n = p.get("trecho")
        if (
            not isinstance(p.get("citacao"), str)
            or not isinstance(n, int)
            or not 1 <= n <= len(chunks)
            or not citation_found(p["citacao"], chunks[n - 1])
        ):
            return "citacao_ausente"
        citacoes.append(p["citacao"])
    if not number_tokens(frase.get("texto", "")) <= number_tokens(" ".join(citacoes)):
        return "numero_sem_prova"
    return None
