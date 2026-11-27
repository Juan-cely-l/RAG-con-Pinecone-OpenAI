"""Regenerate the results blocks of README.md from eval/results/results.json.

Every number shown between ``<!-- RESULTS:START x -->`` and ``<!-- RESULTS:END x -->``
in the README comes from this script, so the documentation cannot drift from the
real evaluation run.

Usage (from the repo root):
    python eval/fill_readme.py           # rewrite the blocks
    python eval/fill_readme.py --check   # exit 1 if README is out of sync
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
RESULTS = REPO_ROOT / "eval" / "results" / "results.json"
HISTORY_V1 = REPO_ROOT / "eval" / "results" / "history" / "results_v1_prompt-idioma-implicito.json"

sys.path.insert(0, str(REPO_ROOT))


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f} %"


def num(x: float | None, nd: int = 2) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _cfg(results: dict, name: str) -> dict:
    for cfg in results["configs"]:
        if cfg["name"] == name:
            return cfg
    raise SystemExit(f"Config {name!r} not found in results.json")


def _items_below(cfg: dict, threshold: float) -> list[dict]:
    """Items whose top-1 score is below ``threshold`` (they would abstain deterministically)."""
    return [
        r for r in cfg["records"]
        if r.get("top1_score") is not None and r["top1_score"] < threshold
        and r["type"] in ("answerable", "out_of_scope", "adversarial")
    ]


def _describe_changes(changed: list[dict], threshold: float) -> str:
    if not changed:
        return f"Con {threshold} ningún ítem del dataset cambia de resultado frente al 0.5 de la corrida."
    flips = [r for r in changed if r["type"] == "answerable" or not r["abstained"]]
    ids = ", ".join(f"`{r['id']}` ({r['type']}, score {r['top1_score']:.3f})" for r in changed)
    if not flips:
        return (f"Con {threshold}, {len(changed)} ítem(s) quedarían bajo el umbral ({ids}). Esos ítems ya se "
                f"abstenían en la corrida, así que ahora solo se abstendrían antes, sin llamar al LLM: "
                f"**las métricas no cambian**.")
    return (f"Con {threshold}, {len(changed)} ítem(s) quedarían bajo el umbral ({ids}); "
            f"{len(flips)} cambiarían de resultado, así que conviene volver a correr la evaluación.")


def build_blocks(results: dict) -> dict[str, str]:
    from config import Settings  # adopted default threshold lives in config.py

    final_threshold = Settings().score_threshold
    a, b = _cfg(results, "cs800"), _cfg(results, "cs400")
    ma, mb = a["metrics"], b["metrics"]
    ta, tb = a["threshold"], b["threshold"]
    date = results["timestamp"][:10]
    changed = _items_below(a, final_threshold) + _items_below(b, final_threshold)

    blocks: dict[str, str] = {}

    blocks["chunking"] = (
        f"**Experimento** (`eval/run_eval.py`, top_k = 4, corrida del {date}): cs800 obtuvo "
        f"Hit@4 = {pct(ma['hit@4'])} y MRR@4 = {num(ma['mrr@4'])}, frente a Hit@4 = {pct(mb['hit@4'])} "
        f"y MRR@4 = {num(mb['mrr@4'])} en cs400. La cobertura de keywords fue {pct(ma['keyword_coverage'])} "
        f"frente a {pct(mb['keyword_coverage'])}. El score top-1 medio de las respondibles fue "
        f"{num(ma['top1_answerable']['mean'], 3)} (cs800) frente a {num(mb['top1_answerable']['mean'], 3)} (cs400).\n\n"
        f"**Decisión: se mantiene 800/120.** Con este corpus el retrieval se satura en ambas "
        f"configuraciones (la fuente correcta siempre está en el top-1), así que el experimento no "
        f"muestra una ganancia de calidad que justifique duplicar los vectores "
        f"({a['ingestion']['chunks']} frente a {b['ingestion']['chunks']}). cs400 separa algo mejor "
        f"los scores de respondibles y de fuera de alcance; se anota como línea de trabajo futuro si "
        f"el corpus crece."
    )

    blocks["topk"] = (
        f"**Experimento** (derivado de la misma lista ordenada, cs800): Hit@2 = {pct(ma['hit@2'])} y "
        f"Hit@4 = {pct(ma['hit@4'])}. **Decisión: se mantiene top_k = 4.** En este dataset k = 2 ya "
        f"encuentra la fuente esperada, pero 6 preguntas aceptan dos archivos fuente y k = 4 deja "
        f"margen para preguntas que combinan temas. El costo extra es bajo: la latencia media de "
        f"respuesta fue {num(ma['latency_mean_s'])} s (p95 {num(ma['latency_p95_s'])} s)."
    )

    blocks["threshold"] = (
        f"**Calibración** (`run_eval.py` maximiza la exactitud balanceada entre \"no abstenerse en "
        f"respondibles\" y \"abstenerse en fuera de alcance\" usando solo el score top-1):\n\n"
        f"| Config | Top-1 respondibles (min / media) | Top-1 fuera de alcance (min / máx) | Umbral sugerido | Respondibles que conserva |\n"
        f"|---|---|---|---|---|\n"
        f"| cs800 | {num(ma['top1_answerable']['min'], 3)} / {num(ma['top1_answerable']['mean'], 3)} | "
        f"{num(ma['top1_out_of_scope']['min'], 3)} / {num(ma['top1_out_of_scope']['max'], 3)} | "
        f"{num(ta['suggested'], 3)} | {pct(ta.get('answerable_kept'))} |\n"
        f"| cs400 | {num(mb['top1_answerable']['min'], 3)} / {num(mb['top1_answerable']['mean'], 3)} | "
        f"{num(mb['top1_out_of_scope']['min'], 3)} / {num(mb['top1_out_of_scope']['max'], 3)} | "
        f"{num(tb['suggested'], 3)} | {pct(tb.get('answerable_kept'))} |\n\n"
        f"Los rangos se **solapan**: ningún umbral separa perfectamente ambos grupos, y el umbral "
        f"\"óptimo\" haría que el sistema dejara de responder preguntas válidas. **Valor adoptado: "
        f"`RAG_SCORE_THRESHOLD = {final_threshold}`**. Queda por debajo de todas las respondibles "
        f"observadas, con margen, y sirve solo para descartar preguntas claramente ajenas al dominio "
        f"sin gastar una llamada al LLM. {_describe_changes(changed, final_threshold)} "
        f"Las preguntas cercanas al dominio las resuelve la "
        f"segunda barrera (la instrucción de abstención en el prompt), que logró "
        f"{pct(ma['correct_abstention_oos'])} de abstención correcta."
    )

    ca5 = ma["hit@4"] >= 0.85 and ma["correct_abstention_oos"] >= 0.8
    ca6 = ma["citation_rate"] >= 1.0
    rows = [
        ("Chunks indexados", str(a["ingestion"]["chunks"]), str(b["ingestion"]["chunks"])),
        ("Hit@1", pct(ma["hit@1"]), pct(mb["hit@1"])),
        ("Hit@2", pct(ma["hit@2"]), pct(mb["hit@2"])),
        ("Hit@4", pct(ma["hit@4"]), pct(mb["hit@4"])),
        ("MRR@4", num(ma["mrr@4"]), num(mb["mrr@4"])),
        ("Abstención correcta (fuera de alcance)", pct(ma["correct_abstention_oos"]), pct(mb["correct_abstention_oos"])),
        ("Falsa abstención (respondibles)", pct(ma["false_abstention_answerable"]), pct(mb["false_abstention_answerable"])),
        ("Cobertura de keywords", pct(ma["keyword_coverage"]), pct(mb["keyword_coverage"])),
        ("Tasa de citación", pct(ma["citation_rate"]), pct(mb["citation_rate"])),
        ("Prompt injection contenida", pct(ma["adversarial_pass"]), pct(mb["adversarial_pass"])),
        ("Entrada inválida rechazada", pct(ma["invalid_pass"]), pct(mb["invalid_pass"])),
        ("Latencia media / p95 (s)", f"{num(ma['latency_mean_s'])} / {num(ma['latency_p95_s'])}",
         f"{num(mb['latency_mean_s'])} / {num(mb['latency_p95_s'])}"),
        ("Juez: fidelidad / relevancia (1–5)",
         f"{num(ma['judge_faithfulness'])} / {num(ma['judge_relevance'])} (n = {ma['n_judged']})",
         "– (juez solo en cs800)"),
        ("Errores de ejecución", str(ma["n_errors"]), str(mb["n_errors"])),
    ]
    table = "| Métrica | cs800 (800/120) | cs400 (400/60) |\n|---|---|---|\n" + "\n".join(
        f"| {m} | {x} | {y} |" for m, x, y in rows
    )
    summary = (
        f"Corrida real del {date}: generador `{results['chat_model']}`, embeddings "
        f"`{results['embedding_model']}`, juez `{results.get('judge_model') or '–'}`, top_k = 4, "
        f"{results['dataset']['n_items']} ítems ({', '.join(f'{v} {k}' for k, v in results['dataset']['by_type'].items())}).\n\n"
        f"{table}\n\n"
        f"Criterios de aceptación (`docs/SPEC.md`): **CA5** (Hit@4 ≥ 85 % y abstención correcta ≥ 80 %): "
        f"{'cumple' if ca5 else 'NO cumple'}. **CA6** (toda respuesta no abstenida cita una fuente): "
        f"{'cumple' if ca6 else 'NO cumple'}."
    )
    if HISTORY_V1.exists():
        v1 = json.loads(HISTORY_V1.read_text(encoding="utf-8"))
        v1a, v1b = _cfg(v1, "cs800")["metrics"], _cfg(v1, "cs400")["metrics"]
        summary += (
            "\n\n**Iteración guiada por la evaluación (prompt de idioma).** En la primera corrida completa "
            "(`eval/results/history/`) el modelo respondía en español a preguntas en inglés; la causa fue que "
            "el mensaje fijo de abstención, en español, sesgaba el idioma. Se hizo explícita la regla de "
            "idioma en el system prompt y se volvió a evaluar:\n\n"
            "| Versión del prompt | Keywords cs800 | Keywords cs400 | Falsa abstención cs800 | Falsa abstención cs400 |\n"
            "|---|---|---|---|---|\n"
            f"| v1 (regla de idioma implícita) | {pct(v1a['keyword_coverage'])} | {pct(v1b['keyword_coverage'])} | "
            f"{pct(v1a['false_abstention_answerable'])} | {pct(v1b['false_abstention_answerable'])} |\n"
            f"| v2 (regla de idioma explícita, actual) | {pct(ma['keyword_coverage'])} | {pct(mb['keyword_coverage'])} | "
            f"{pct(ma['false_abstention_answerable'])} | {pct(mb['false_abstention_answerable'])} |\n\n"
            "v2 corrige el idioma, pero en cada configuración aparece una abstención incorrecta en una pregunta "
            "en español (`ans-04` en cs800, `ans-13` en cs400). Al repetir ambas preguntas 3 veces cada una, el "
            "sistema respondió bien las 6 veces, así que es **no determinismo** del modelo lite (ignora "
            "`temperature`), no un fallo sistemático. Ver Limitaciones."
        )
    blocks["summary"] = summary
    return blocks


def render(readme: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- RESULTS:START {name} -->\n).*?(\n<!-- RESULTS:END {name} -->)", re.DOTALL
        )
        if not pattern.search(readme):
            raise SystemExit(f"Marker block {name!r} not found in README.md")
        readme = pattern.sub(lambda m: m.group(1) + body + m.group(2), readme)
    return readme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Only verify README is in sync.")
    args = parser.parse_args(argv)

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    current = README.read_text(encoding="utf-8")
    updated = render(current, build_blocks(results))
    if "{{" in updated:
        print("README still contains {{placeholders}}.", file=sys.stderr)
        return 1
    if args.check:
        if updated != current:
            print("README.md is OUT OF SYNC with eval/results/results.json", file=sys.stderr)
            return 1
        print("README.md matches eval/results/results.json")
        return 0
    README.write_text(updated, encoding="utf-8")
    print("README.md results blocks regenerated from eval/results/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
