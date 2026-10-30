"""Retrieval funnel: at which step does the right chunk die?

Steps:
  0 gold document has no evidence
  1 evidence is split across chunks
  2 outside the candidate pool
  3 dropped by rerank
  4 MCP k-cut (CLI found it, MCP k is smaller)
  5 generation ignored the evidence
  viva
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import comb

from eval.metrics import facet_coverage

L_PADRAO: float = 0.6
DEGRAUS = ("0", "1", "2", "3", "4", "5", "viva")
MARCAS_PRIMEIRA_BUSCA = ("denso", "esparso", "nenhum", "sintetico_na_vaga")

LAT_OK, LAT_DONO, GANHO_MIN, GANHO_DONO, PERDA_MAX = 1.10, 1.25, 3, 6, 1


def cobertura_chunk(doc, keywords) -> float:
    return facet_coverage([doc], keywords) or 0.0


def melhor_cobertura(docs, keywords) -> float:
    return max((cobertura_chunk(d, keywords) for d in docs), default=0.0)


def evidencia(docs, keywords, L) -> bool:
    return melhor_cobertura(docs, keywords) >= L


def diagnostico_aula(chunks, keywords, L) -> dict:
    ordem = sorted(chunks, key=lambda d: d.metadata.get("chunk") or 0)
    pares = zip(ordem, ordem[1:])
    return {
        "n_chunks": len(chunks),
        "uniao": facet_coverage(chunks, keywords) or 0.0,
        "melhor_chunk": melhor_cobertura(chunks, keywords),
        "melhor_janela2": max((facet_coverage([a, b], keywords) or 0.0 for a, b in pares), default=0.0),
    }


def classificar(
    vivo: dict,
    tem_ouro: bool,
    caminho: str,
    cli_final_vivo: bool | None = None,
    resposta_cobre: bool | None = None,
) -> str:
    if caminho == "mcp" and cli_final_vivo and not vivo["final"]:
        return "4"
    if not vivo["pool"]:
        if tem_ouro and not vivo["aula"]:
            return "0"
        if tem_ouro and not vivo["chunk"]:
            return "1"
        return "2"
    if not vivo["final"]:
        return "3"
    if resposta_cobre is False:
        return "5"
    return "viva"


def marcas(degrau, vivo, tem_ouro, trace_vivo: dict, n_sintetico_fusao: int, n_chunks_ouro) -> list[str]:
    m = []
    if not tem_ouro:
        m.append("sem_ouro")
    if n_chunks_ouro == 0:
        m.append("ouro_vazio")
    if tem_ouro and vivo["pool"] and not vivo["chunk"]:
        m.append("outra_aula")
    if degrau == "2":
        achou = [modo for modo in ("denso", "esparso") if trace_vivo.get(modo)]
        m += achou or ([] if trace_vivo.get("fusao") else ["nenhum"])
        if n_sintetico_fusao >= 1:
            m.append("sintetico_na_vaga")
    return m


def resumo(itens: list[dict]) -> dict:
    grupos = defaultdict(list)
    for it in itens:
        grupos[f'{it["colecao"]}|{it["golden"]}|{it["caminho"]}'].append(it)
    out = {}
    for chave, its in sorted(grupos.items()):
        degraus = Counter(it["degrau"] for it in its if it["degrau"] is not None)
        mc = Counter(m for it in its for m in it["marcas"])
        assert sum(degraus.values()) == len(its) - mc["sem_facetas"], chave
        out[chave] = {"n": len(its), "degraus": {d: degraus.get(d, 0) for d in DEGRAUS}, "marcas": dict(mc)}
    return out


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def _vivas(itens, caminho) -> dict[str, bool]:
    return {i["id"]: i["degrau"] == "viva" for i in itens if i["caminho"] == caminho and i["degrau"] is not None}


def _bc(base_itens, novo_itens, caminho) -> tuple[int, int]:
    vb, vn = _vivas(base_itens, caminho), _vivas(novo_itens, caminho)
    comuns = vb.keys() & vn.keys()
    return sum(1 for q in comuns if vn[q] and not vb[q]), sum(1 for q in comuns if vb[q] and not vn[q])


def _taxa_sem_base(gate) -> float:
    fr = [f for it in gate["items"] for f in it.get("frases", [])]
    return sum(1 for f in fr if f["veredito"] == "nao_sustenta") / len(fr) if fr else 0.0


def aceite(base: dict, novo: dict, visado: str, gate_base: dict | None, gate_novo: dict | None) -> dict:
    """`base`/`novo` = {collection: (items, summary)}. Each path measured separately."""
    caminhos = ("cli", "mcp")
    crit = []

    def add(nome, ok, detalhe):
        crit.append({"nome": nome, "ok": bool(ok), "detalhe": detalhe})

    ganhos, det = {}, {}
    for c in caminhos:
        b, cc = _bc(base[visado][0], novo[visado][0], c)
        ganhos[c] = b - cc
        det[c] = {"b": b, "c": cc, "p": round(mcnemar(b, cc), 4)}
    add("ganho_visado", all(g >= GANHO_MIN for g in ganhos.values()), det)

    perdas = {}
    for col in (k for k in novo if k != visado):
        for c in caminhos:
            b, cc = _bc(base[col][0], novo[col][0], c)
            perdas[f"{col}|{c}"] = cc - b
    add("perda_outro", all(p <= PERDA_MAX for p in perdas.values()), perdas)

    razoes = {}
    for col in novo:
        lb, ln = base[col][1].get("latencia") or {}, novo[col][1].get("latencia") or {}
        for c in caminhos:
            for q in ("p50", "p95"):
                if lb.get(c) and ln.get(c) and lb[c][q]:
                    razoes[f"{col}|{c}|{q}"] = round(ln[c][q] / lb[c][q], 3)
    pior = max(razoes.values(), default=1.0)
    pede_dono = LAT_OK < pior <= LAT_DONO and all(g >= GANHO_DONO for g in ganhos.values())
    add("latencia", pior <= LAT_OK or pede_dono, {"pior": pior, "razoes": razoes})

    difs = []
    for col, (itens, _) in novo.items():
        por = {c: {i["id"]: i for i in itens if i["caminho"] == c and i["degrau"] is not None} for c in caminhos}
        for q in por["cli"].keys() & por["mcp"].keys():
            a, m = por["cli"][q], por["mcp"][q]
            if any(a["config"][x] != m["config"][x] for x in ("k", "cand", "pools_separados")) or a["ids_final"] != m["ids_final"]:
                difs.append(f"{col}:{q}")
    add("mcp_cli_iguais", not difs, {"n_diferentes": len(difs), "exemplos": difs[:5]})

    if gate_base is not None and gate_novo is not None:
        tb, tn = _taxa_sem_base(gate_base), _taxa_sem_base(gate_novo)
        add("frase_sem_base", tn <= tb, {"base": round(tb, 4), "novo": round(tn, 4)})

    ok = all(c["ok"] for c in crit)
    return {"ok": ok, "pede_dono": ok and pede_dono, "criterios": crit}
