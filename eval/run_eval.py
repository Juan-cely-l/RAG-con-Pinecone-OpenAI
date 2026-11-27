"""Offline evaluation harness for the MMA rules RAG assistant.

Usage (from the repo root):

    python eval/run_eval.py                       # cs800 + cs400, judge on cs800
    python eval/run_eval.py --configs cs800 --no-judge --limit 5
    python eval/run_eval.py --skip-ingest         # reuse exp-* namespaces
    python eval/run_eval.py --report-only         # rebuild REPORT.md from results.json

Every external dependency (ingestion, retrieval, generation, judge, sleep) is
injected through ``Deps`` so the whole flow can be exercised with fakes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from errors import IndexNotReadyError, InvalidInputError, RAGError  # noqa: E402

logger = logging.getLogger("eval")

DEFAULT_DATASET = REPO_ROOT / "eval" / "dataset.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "results"
DATA_DIR = REPO_ROOT / "data"

PRESETS: dict[str, dict[str, int]] = {
    "cs800": {"chunk_size": 800, "chunk_overlap": 120},  # production default
    "cs400": {"chunk_size": 400, "chunk_overlap": 60},
}
EVAL_TOP_K = 4
ABSTENTION_PHRASES = (
    "no tengo informacion suficiente",
    "no tengo suficiente informacion",
    "i don't have enough information",
    "i do not have enough information",
    "not enough information",
)
LOW_KEYWORD_COVERAGE = 0.5
JUDGE_PASS = 4


# --------------------------------------------------------------------------- deps


@dataclass
class Deps:
    load_settings: Callable[..., Any]
    run_ingestion: Callable[..., dict]
    build_rag_pipeline: Callable[..., Any]
    retrieve: Callable[..., list]
    answer_question: Callable[..., dict]
    judge: Callable[..., dict] | None = None
    sleep: Callable[[float], None] = time.sleep
    no_info_message: str = ""
    index_poll_attempts: int = 6
    index_poll_delay: float = 10.0



JUDGE_MODEL: str | None = None  # set from --judge-model in main()

def load_real_deps() -> Deps:
    """Import the real pipeline lazily (only when actually running an evaluation)."""
    from config import load_settings
    from ingest import run_ingestion
    from rag_chain import NO_INFO_MESSAGE, answer_question, build_rag_pipeline, get_chat_model, retrieve
    from retry import call_with_retry

    def judge(settings: Any, question: str, answer: str, contexts: list[str]) -> dict:
        # A different model than the generator (own free-tier quota, less self-preference bias).
        llm = get_chat_model(replace(settings, chat_model=JUDGE_MODEL or settings.chat_model, temperature=0.0))
        prompt = build_judge_prompt(question, answer, contexts)
        response = call_with_retry(
            lambda: llm.invoke(prompt),
            max_retries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            description="LLM judge",
        )
        return parse_judge_output(_message_text(response))

    return Deps(
        load_settings=load_settings,
        run_ingestion=run_ingestion,
        build_rag_pipeline=build_rag_pipeline,
        retrieve=retrieve,
        answer_question=answer_question,
        judge=judge,
        no_info_message=NO_INFO_MESSAGE,
    )


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):  # Gemini may return a list of parts
        parts = []
        for part in content:
            parts.append(part.get("text", "") if isinstance(part, dict) else str(part))
        return "".join(parts)
    return str(content)


# --------------------------------------------------------------------------- judge


def build_judge_prompt(question: str, answer: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "(sin contexto)"
    return (
        "You are a strict evaluator of a retrieval-augmented QA system about MMA rules.\n"
        "Score the ANSWER using ONLY the CONTEXT.\n"
        "- faithfulness (1-5): 5 = every claim is supported by the context; 1 = mostly unsupported/hallucinated.\n"
        "- relevance (1-5): 5 = directly and completely answers the question; 1 = off-topic.\n"
        "Citations like [data/file.md] are expected and are not claims.\n"
        'Reply with ONLY a JSON object: {"faithfulness": <int>, "relevance": <int>, "reason": "<short>"}\n\n'
        f"CONTEXT:\n{ctx}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}\n"
    )


def parse_judge_output(text: str) -> dict:
    """Extract {"faithfulness", "relevance", "reason"} robustly from model text."""
    result: dict[str, Any] = {"faithfulness": None, "relevance": None, "reason": "", "raw": text[:500]}
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    data: dict[str, Any] = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}
    for key in ("faithfulness", "relevance"):
        value = data.get(key)
        if value is None:
            m = re.search(rf'"?{key}"?\s*[:=]\s*([1-5])', text or "", flags=re.IGNORECASE)
            value = m.group(1) if m else None
        try:
            iv = int(float(value)) if value is not None else None
            result[key] = iv if iv is not None and 1 <= iv <= 5 else None
        except (TypeError, ValueError):
            result[key] = None
    result["reason"] = str(data.get("reason", ""))[:300]
    return result


# --------------------------------------------------------------------------- helpers


def load_dataset(path: Path | str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            for key in ("id", "question", "type"):
                if key not in item:
                    raise ValueError(f"{path}:{lineno}: missing field {key!r}")
            items.append(item)
    return items


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        text = text.replace(dash, "-")
    return text.lower()


def source_key(source: Any) -> str:
    """Normalize a source path to its basename ("data\\x.md", "./data/x.md" -> "x.md")."""
    return str(source or "").replace("\\", "/").rstrip("/").split("/")[-1].lower()


def expected_sources(item: dict) -> list[str]:
    exp = item.get("expected_source")
    if exp is None:
        return []
    return [exp] if isinstance(exp, str) else list(exp)


def keyword_hits(answer: str, keywords: list) -> dict[str, bool]:
    """Each keyword is a string or a list of alternative strings (any matches)."""
    norm = normalize_text(answer)
    hits = {}
    for kw in keywords:
        alts = kw if isinstance(kw, list) else [kw]
        hits["|".join(alts)] = any(normalize_text(a) in norm for a in alts)
    return hits


def is_abstention(result: dict, no_info_message: str) -> bool:
    if result.get("abstained"):
        return True
    norm = normalize_text(result.get("answer", ""))
    phrases = list(ABSTENTION_PHRASES)
    if no_info_message:
        phrases.append(normalize_text(no_info_message)[:60])
    return any(p and p in norm for p in phrases)


def corpus_filenames() -> list[str]:
    try:
        return sorted(p.name.lower() for p in DATA_DIR.iterdir() if p.suffix in (".md", ".txt"))
    except OSError:
        return []


def has_citation(answer: str, filenames: list[str]) -> bool:
    low = (answer or "").lower()
    return "[data/" in low or any(name in low for name in filenames)


def _doc_source(doc: Any) -> str:
    meta = getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {}) or {}
    return str(meta.get("source", ""))


def _doc_text(doc: Any) -> str:
    if isinstance(doc, (tuple, list)) and doc:
        doc = doc[0]
    if isinstance(doc, dict):
        return str(doc.get("page_content", doc.get("text", "")))
    return str(getattr(doc, "page_content", doc))


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    rank = max(1, math.ceil(pct / 100 * len(vals)))
    return vals[rank - 1]


def rate(flags: list[bool]) -> float | None:
    return sum(1 for f in flags if f) / len(flags) if flags else None


# --------------------------------------------------------------------------- evaluation


def _wait_for_pipeline(settings: Any, deps: Deps, expected_chunks: int | None) -> Any:
    """Build the pipeline, retrying while Pinecone (eventually consistent) is not ready."""
    last_exc: Exception | None = None
    for attempt in range(1, deps.index_poll_attempts + 1):
        try:
            pipeline = deps.build_rag_pipeline(settings, top_k=settings.top_k, namespace=settings.namespace)
        except IndexNotReadyError as exc:
            last_exc = exc
            logger.info("Index not ready (attempt %d/%d): %s", attempt, deps.index_poll_attempts, exc)
            deps.sleep(deps.index_poll_delay)
            continue
        count = _namespace_vector_count(pipeline, settings.namespace)
        if expected_chunks and count is not None and count < expected_chunks:
            logger.info("Namespace %r has %d/%d vectors; waiting...", settings.namespace, count, expected_chunks)
            last_exc = IndexNotReadyError(f"only {count}/{expected_chunks} vectors visible")
            deps.sleep(deps.index_poll_delay)
            continue
        return pipeline
    raise IndexNotReadyError(f"Index not ready after {deps.index_poll_attempts} attempts: {last_exc}")


def _namespace_vector_count(pipeline: Any, namespace: str) -> int | None:
    """Best effort: read vector count for the namespace from describe_index_stats."""
    store = getattr(pipeline, "vector_store", None)
    index = getattr(store, "index", None) or getattr(store, "_index", None)
    if index is None or not hasattr(index, "describe_index_stats"):
        return None
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces") if isinstance(stats, dict) else getattr(stats, "namespaces", None)
        ns = (namespaces or {}).get(namespace or "")
        if ns is None:
            return 0
        return int(ns.get("vector_count") if isinstance(ns, dict) else getattr(ns, "vector_count", 0))
    except Exception:  # noqa: BLE001 - purely informative
        logger.debug("describe_index_stats failed", exc_info=True)
        return None


def evaluate_item(
    item: dict,
    pipeline: Any,
    settings: Any,
    deps: Deps,
    *,
    use_judge: bool,
    sleep_s: float,
    filenames: list[str],
) -> dict:
    itype = item["type"]
    record: dict[str, Any] = {
        "id": item["id"],
        "type": itype,
        "lang": item.get("lang"),
        "question": item["question"] if len(item["question"]) <= 200 else item["question"][:80] + f"... ({len(item['question'])} chars)",
        "expected_source": item.get("expected_source"),
        "error": None,
    }

    if itype == "invalid":
        try:
            deps.answer_question(item["question"], pipeline=pipeline)
            record.update(passed=False, raised=None)
        except InvalidInputError as exc:
            record.update(passed=True, raised=type(exc).__name__, message=str(exc)[:200])
        except Exception as exc:  # noqa: BLE001
            record.update(passed=False, raised=type(exc).__name__, message=str(exc)[:200])
        return record

    try:
        ranked = deps.retrieve(pipeline, item["question"])
        if sleep_s:
            deps.sleep(sleep_s)
        retrieved = []
        for doc, score in ranked:
            meta = getattr(doc, "metadata", {}) or {}
            retrieved.append({"source": _doc_source(doc), "chunk_index": meta.get("chunk_index"), "score": float(score)})
        record["retrieved"] = retrieved
        record["top1_score"] = retrieved[0]["score"] if retrieved else None

        exp = {source_key(s) for s in expected_sources(item)}
        rank = None
        if exp:
            for i, r in enumerate(retrieved, 1):
                if source_key(r["source"]) in exp:
                    rank = i
                    break
        record["rank"] = rank

        t0 = time.perf_counter()
        result = deps.answer_question(item["question"], pipeline=pipeline)
        wall = time.perf_counter() - t0
        record["latency_s"] = float(result.get("latency_s") or wall)
        answer = str(result.get("answer", ""))
        record["answer"] = answer
        record["abstained_flag"] = bool(result.get("abstained"))
        record["abstain_reason"] = result.get("abstain_reason")
        record["abstained"] = is_abstention(result, deps.no_info_message)
        record["answer_sources"] = result.get("sources", [])
        record["has_citation"] = (not record["abstained"]) and has_citation(answer, filenames)
        if sleep_s:  # answer_question embeds the query (+ LLM call unless it abstained by score)
            deps.sleep(sleep_s)

        if itype == "answerable":
            hits = keyword_hits(answer, item.get("expected_keywords", []))
            record["keyword_hits"] = hits
            record["keyword_coverage"] = (sum(hits.values()) / len(hits)) if hits else None
            if use_judge and deps.judge and not record["abstained"]:
                docs = result.get("documents") or [d for d, _ in ranked]
                contexts = [f"({_doc_source(d[0] if isinstance(d, (tuple, list)) else d)}) {_doc_text(d)}" for d in docs]
                try:
                    record["judge"] = deps.judge(settings, item["question"], answer, contexts)
                except Exception as exc:  # noqa: BLE001
                    record["judge"] = {"faithfulness": None, "relevance": None, "reason": f"judge error: {exc}"[:300]}
                if sleep_s:
                    deps.sleep(sleep_s)
            record["passed"] = bool(rank) and not record["abstained"]
        elif itype == "out_of_scope":
            record["passed"] = record["abstained"]
        elif itype == "adversarial":
            forbidden = [k for k in item.get("forbidden_keywords", []) if normalize_text(k) in normalize_text(answer)]
            record["forbidden_hits"] = forbidden
            record["passed"] = record["abstained"] or not forbidden
    except Exception as exc:  # noqa: BLE001 - record and continue
        logger.warning("Item %s failed: %s: %s", item["id"], type(exc).__name__, exc)
        logger.debug("traceback", exc_info=True)
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        record["passed"] = False
    return record


def evaluate_config(
    name: str,
    items: list[dict],
    deps: Deps,
    *,
    use_judge: bool,
    sleep_s: float,
    skip_ingest: bool,
    on_progress: Callable[[dict], None] | None = None,
    max_consecutive_errors: int = 5,
) -> dict:
    preset = PRESETS[name]
    base = deps.load_settings()
    settings = replace(base, top_k=EVAL_TOP_K, namespace=f"exp-{name}", **preset)
    cfg: dict[str, Any] = {
        "name": name,
        "params": {**preset, "top_k": settings.top_k, "namespace": settings.namespace,
                   "score_threshold": settings.score_threshold},
        "judge": use_judge,
        "ingestion": None,
        "records": [],
        "aborted": None,
    }
    if not skip_ingest:
        logger.info("[%s] Ingesting into namespace %r (chunk %d / overlap %d)...",
                    name, settings.namespace, settings.chunk_size, settings.chunk_overlap)
        cfg["ingestion"] = deps.run_ingestion(settings, namespace=settings.namespace, reset=True)
        logger.info("[%s] Ingestion: %s", name, cfg["ingestion"])
        deps.sleep(deps.index_poll_delay)
    expected_chunks = (cfg["ingestion"] or {}).get("chunks")
    pipeline = _wait_for_pipeline(settings, deps, expected_chunks)

    filenames = corpus_filenames()
    consecutive = 0
    for n, item in enumerate(items, 1):
        logger.info("[%s] (%d/%d) %s", name, n, len(items), item["id"])
        rec = evaluate_item(item, pipeline, settings, deps, use_judge=use_judge, sleep_s=sleep_s, filenames=filenames)
        cfg["records"].append(rec)
        consecutive = consecutive + 1 if rec.get("error") else 0
        if on_progress:
            on_progress(cfg)
        if max_consecutive_errors and consecutive >= max_consecutive_errors:
            cfg["aborted"] = f"{consecutive} consecutive errors (quota?); last: {rec['error']}"
            logger.error("[%s] Aborting config: %s", name, cfg["aborted"])
            break
    cfg["metrics"] = compute_metrics(cfg["records"], settings.score_threshold)
    cfg["threshold"] = suggest_threshold(cfg["records"], settings.score_threshold)
    return cfg


# --------------------------------------------------------------------------- metrics


def compute_metrics(records: list[dict], current_threshold: float | None = None) -> dict:
    ok = [r for r in records if not r.get("error")]
    ans = [r for r in ok if r["type"] == "answerable"]
    oos = [r for r in ok if r["type"] == "out_of_scope"]
    adv = [r for r in ok if r["type"] == "adversarial"]
    inv = [r for r in records if r["type"] == "invalid"]
    answered = [r for r in ans if not r.get("abstained")]
    non_abstained_all = [r for r in ok if r["type"] != "invalid" and not r.get("abstained")]

    def hit(k: int) -> float | None:
        return rate([bool(r.get("rank")) and r["rank"] <= k for r in ans])

    judged = [r["judge"] for r in ans if r.get("judge")]
    m = {
        "n_items": len(records),
        "n_errors": sum(1 for r in records if r.get("error")),
        "n_answerable": len(ans),
        "n_out_of_scope": len(oos),
        "hit@1": hit(1),
        "hit@2": hit(2),
        "hit@4": hit(4),
        "mrr@4": mean([1.0 / r["rank"] if r.get("rank") else 0.0 for r in ans]) if ans else None,
        "correct_abstention_oos": rate([r["abstained"] for r in oos]),
        "false_abstention_answerable": rate([r["abstained"] for r in ans]),
        "keyword_coverage": mean([r.get("keyword_coverage") for r in answered]),
        "citation_rate": rate([r.get("has_citation", False) for r in non_abstained_all]),
        "adversarial_pass": rate([r["passed"] for r in adv]),
        "invalid_pass": rate([r["passed"] for r in inv]),
        "latency_mean_s": mean([r.get("latency_s") for r in ok if "latency_s" in r]),
        "latency_p95_s": percentile([r.get("latency_s") for r in ok if "latency_s" in r], 95),
        "judge_faithfulness": mean([j.get("faithfulness") for j in judged]),
        "judge_relevance": mean([j.get("relevance") for j in judged]),
        "n_judged": sum(1 for j in judged if j.get("faithfulness") is not None),
        "top1_answerable": _stats([r.get("top1_score") for r in ans]),
        "top1_out_of_scope": _stats([r.get("top1_score") for r in oos]),
        "top1_answerable_en": _stats([r.get("top1_score") for r in ans if r.get("lang") == "en"]),
        "top1_answerable_es": _stats([r.get("top1_score") for r in ans if r.get("lang") == "es"]),
        "hit@4_en": rate([bool(r.get("rank")) for r in ans if r.get("lang") == "en"]),
        "hit@4_es": rate([bool(r.get("rank")) for r in ans if r.get("lang") == "es"]),
    }
    if current_threshold is not None:
        m["current_threshold"] = current_threshold
    return m


def _stats(values: list[float | None]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "mean": None, "max": None}
    return {"n": len(vals), "min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}


def _threshold_accuracy(t: float, ans_scores: list[float], oos_scores: list[float]) -> tuple[float, float, float]:
    keep = sum(1 for s in ans_scores if s >= t) / len(ans_scores) if ans_scores else 0.0
    abst = sum(1 for s in oos_scores if s < t) / len(oos_scores) if oos_scores else 0.0
    return (keep + abst) / 2, keep, abst


def suggest_threshold(records: list[dict], current: float | None = None) -> dict:
    """Pick the retrieval-score threshold that best separates answerable from out-of-scope.

    Objective: balanced accuracy = mean(non-abstention on answerable, abstention on OOS),
    using only the top-1 retrieval score (i.e. what the deterministic abstention sees).
    Ties are broken by the largest margin to the nearest observed score.
    """
    ans = [r["top1_score"] for r in records if r["type"] == "answerable" and r.get("top1_score") is not None]
    oos = [r["top1_score"] for r in records if r["type"] == "out_of_scope" and r.get("top1_score") is not None]
    out: dict[str, Any] = {"current": current, "suggested": None}
    if not ans or not oos:
        out["note"] = "No hay suficientes puntajes de ambos tipos para calibrar."
        return out
    scores = sorted(set(ans + oos))
    candidates = [scores[0] - 0.01] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 0.01]
    best = None
    for t in candidates:
        bacc, keep, abst = _threshold_accuracy(t, ans, oos)
        margin = min(abs(t - s) for s in scores)
        key = (round(bacc, 9), margin)
        if best is None or key > best[0]:
            best = (key, t, bacc, keep, abst)
    _, t, bacc, keep, abst = best
    out.update(suggested=round(t, 4), balanced_accuracy=bacc, answerable_kept=keep, oos_abstained=abst)
    if current is not None:
        cb, ck, ca = _threshold_accuracy(current, ans, oos)
        out.update(current_balanced_accuracy=cb, current_answerable_kept=ck, current_oos_abstained=ca)
    out["oos_above_suggested"] = [
        {"id": r["id"], "score": r["top1_score"]}
        for r in records if r["type"] == "out_of_scope" and r.get("top1_score") is not None and r["top1_score"] >= t
    ]
    out["separable"] = min(ans) > max(oos)
    return out


# --------------------------------------------------------------------------- failures


def failure_reasons(rec: dict) -> list[str]:
    reasons = []
    if rec.get("error"):
        return [f"error: {rec['error']}"]
    t = rec["type"]
    if t == "answerable":
        if not rec.get("rank"):
            reasons.append("recuperación falló (fuente esperada fuera del top-4)")
        if rec.get("abstained"):
            reasons.append("abstención incorrecta (pregunta respondible)")
        else:
            cov = rec.get("keyword_coverage")
            if cov is not None and cov < LOW_KEYWORD_COVERAGE:
                missing = [k for k, v in rec.get("keyword_hits", {}).items() if not v]
                reasons.append(f"cobertura de keywords baja ({cov:.0%}; faltan: {', '.join(missing)})")
            if not rec.get("has_citation"):
                reasons.append("respuesta sin cita de fuente")
            j = rec.get("judge") or {}
            for key in ("faithfulness", "relevance"):
                if j.get(key) is not None and j[key] < JUDGE_PASS:
                    reasons.append(f"juez: {key}={j[key]} ({j.get('reason', '')})")
    elif t == "out_of_scope" and not rec.get("abstained"):
        reasons.append("no se abstuvo en pregunta fuera de alcance")
    elif t == "adversarial" and not rec.get("passed"):
        reasons.append(f"inyección de prompt no contenida (palabras prohibidas: {rec.get('forbidden_hits')})")
    elif t == "invalid" and not rec.get("passed"):
        reasons.append(f"entrada inválida no rechazada con InvalidInputError (lanzó: {rec.get('raised')})")
    return reasons


def is_success(rec: dict) -> bool:
    if rec.get("type") != "answerable" or rec.get("error") or rec.get("abstained") or rec.get("rank") != 1:
        return False
    if rec.get("keyword_coverage") != 1.0 or not rec.get("has_citation"):
        return False
    j = rec.get("judge") or {}
    return all(j.get(k) is None or j[k] >= JUDGE_PASS for k in ("faithfulness", "relevance"))


# --------------------------------------------------------------------------- report


def _f(v: Any, pct: bool = False, nd: int = 3) -> str:
    if v is None:
        return "–"
    if pct:
        return f"{v:.0%}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _excerpt(text: str | None, n: int = 300) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return (text[:n] + "…") if len(text) > n else text


SUMMARY_ROWS = [
    ("Hit@1", "hit@1", True), ("Hit@2", "hit@2", True), ("Hit@4", "hit@4", True), ("MRR@4", "mrr@4", False),
    ("Abstención correcta (OOS)", "correct_abstention_oos", True),
    ("Falsa abstención (respondibles)", "false_abstention_answerable", True),
    ("Cobertura de keywords", "keyword_coverage", True), ("Tasa de citación", "citation_rate", True),
    ("Adversarial pasa", "adversarial_pass", True), ("Entrada inválida rechazada", "invalid_pass", True),
    ("Latencia media (s)", "latency_mean_s", False), ("Latencia p95 (s)", "latency_p95_s", False),
    ("Juez: fidelidad (1-5)", "judge_faithfulness", False), ("Juez: relevancia (1-5)", "judge_relevance", False),
    ("Ítems con error", "n_errors", False),
]


def render_report(results: dict) -> str:
    cfgs = results.get("configs", [])
    L: list[str] = []
    L.append("# Reporte de evaluación — Asistente RAG de reglamento MMA\n")
    L.append(f"- Fecha (UTC): {results.get('timestamp')}")
    L.append(f"- Modelo de chat: `{results.get('chat_model')}` · Embeddings: `{results.get('embedding_model')}`"
             + (f" · Juez: `{results.get('judge_model')}`" if results.get("judge_model") else ""))
    ds = results.get("dataset", {})
    L.append(f"- Dataset: `{ds.get('path')}` — {ds.get('n_items')} ítems ({ds.get('by_type')})")
    L.append(f"- Configuraciones: {', '.join(c['name'] for c in cfgs)} (top_k={EVAL_TOP_K}); "
             f"juez LLM solo en la primaria ({cfgs[0]['name'] if cfgs else '–'})" if results.get("judge") else
             f"- Configuraciones: {', '.join(c['name'] for c in cfgs)} (top_k={EVAL_TOP_K}); juez LLM desactivado")
    L.append("")

    # Summary
    L.append("## 1. Resumen por configuración\n")
    L.append("| Métrica | " + " | ".join(f"{c['name']} ({c['params']['chunk_size']}/{c['params']['chunk_overlap']})" for c in cfgs) + " |")
    L.append("|---|" + "---|" * len(cfgs))
    for label, key, pct in SUMMARY_ROWS:
        L.append(f"| {label} | " + " | ".join(_f(c["metrics"].get(key), pct) for c in cfgs) + " |")
    L.append("")
    for c in cfgs:
        if c.get("aborted"):
            L.append(f"> **{c['name']} abortada:** {c['aborted']} — las métricas son parciales.\n")

    # Acceptance criteria
    if cfgs:
        m = cfgs[0]["metrics"]
        L.append("### Criterios de aceptación (configuración primaria)\n")
        h4, ab = m.get("hit@4"), m.get("correct_abstention_oos")
        ca5 = "CUMPLE" if (h4 is not None and ab is not None and h4 >= 0.85 and ab >= 0.8) else "NO CUMPLE"
        cit = m.get("citation_rate")
        ca6 = "CUMPLE" if cit == 1.0 else ("NO CUMPLE" if cit is not None else "–")
        L.append(f"- **CA5** (Hit@4 ≥ 0.85 y abstención correcta ≥ 0.8): Hit@4 = {_f(h4, True)}, abstención = {_f(ab, True)} → **{ca5}**")
        L.append(f"- **CA6** (toda respuesta no abstenida cita ≥ 1 fuente): tasa de citación = {_f(cit, True)} → **{ca6}**\n")

    # Experiments
    L.append("## 2. Experimentos\n")
    L.append("### 2.1 Tamaño de chunk: " + " vs ".join(c["name"] for c in cfgs) + "\n")
    L.append("| Config | chunk/overlap | chunks indexados | Hit@1 | Hit@4 | MRR@4 | Cobertura kw | Score top-1 medio (resp.) | Latencia media |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in cfgs:
        m = c["metrics"]
        L.append(f"| {c['name']} | {c['params']['chunk_size']}/{c['params']['chunk_overlap']} | "
                 f"{_f((c.get('ingestion') or {}).get('chunks'))} | {_f(m['hit@1'], True)} | {_f(m['hit@4'], True)} | "
                 f"{_f(m['mrr@4'])} | {_f(m['keyword_coverage'], True)} | {_f(m['top1_answerable']['mean'])} | "
                 f"{_f(m['latency_mean_s'], nd=2)} s |")
    L.append("")
    if len(cfgs) >= 2:
        a, b = cfgs[0], cfgs[1]
        for label, key in (("MRR@4", "mrr@4"), ("Hit@4", "hit@4"), ("cobertura de keywords", "keyword_coverage")):
            va, vb = a["metrics"].get(key), b["metrics"].get(key)
            if va is None or vb is None:
                continue
            if abs(va - vb) < 1e-9:
                L.append(f"- {label}: empate entre {a['name']} y {b['name']} ({_f(va)}).")
            else:
                win = a if va > vb else b
                L.append(f"- {label}: mejor **{win['name']}** ({a['name']} {_f(va)} vs {b['name']} {_f(vb)}).")
        L.append("- Nota: los chunks pequeños dan contexto más preciso pero fragmentan listas (p. ej. faltas o "
                 "categorías de peso) y elevan el número de vectores/embeddings; con un corpus de ~16 KB la diferencia "
                 "es pequeña y debe leerse con cautela (n reducido).\n")

    L.append("### 2.2 top_k = 2 vs top_k = 4 (derivado de la misma lista ordenada)\n")
    L.append("| Config | Hit@2 | Hit@4 | Δ (ganancia por recuperar 4) |")
    L.append("|---|---|---|---|")
    for c in cfgs:
        m = c["metrics"]
        delta = (m["hit@4"] - m["hit@2"]) if m["hit@4"] is not None and m["hit@2"] is not None else None
        L.append(f"| {c['name']} | {_f(m['hit@2'], True)} | {_f(m['hit@4'], True)} | {_f(delta, True)} |")
    L.append("\nCon k=2 el contexto enviado al LLM es aproximadamente la mitad (menos tokens, menor latencia y costo), "
             "pero se pierde la fuente correcta en los casos que solo aparecen en las posiciones 3–4. La generación se "
             "evaluó con k=4; el efecto de k=2 sobre la respuesta no se midió directamente.\n")

    # Threshold
    L.append("## 3. Calibración del umbral de abstención\n")
    L.append("El umbral se aplica al score top-1 de recuperación (similitud coseno). Se busca el valor que maximiza la "
             "exactitud balanceada = promedio(no abstenerse en respondibles, abstenerse en fuera de alcance), "
             "usando solo la señal de recuperación.\n")
    L.append("| Config | Top-1 respondibles (min/media/max) | Top-1 fuera de alcance (min/media/max) | Umbral actual (exactitud bal.) | Umbral sugerido (exactitud bal.) | ¿Separables? |")
    L.append("|---|---|---|---|---|---|")
    for c in cfgs:
        m, th = c["metrics"], c.get("threshold", {})
        sa, so = m["top1_answerable"], m["top1_out_of_scope"]
        L.append(f"| {c['name']} | {_f(sa['min'])} / {_f(sa['mean'])} / {_f(sa['max'])} | "
                 f"{_f(so['min'])} / {_f(so['mean'])} / {_f(so['max'])} | "
                 f"{_f(th.get('current'))} ({_f(th.get('current_balanced_accuracy'), True)}) | "
                 f"{_f(th.get('suggested'))} ({_f(th.get('balanced_accuracy'), True)}) | "
                 f"{'sí' if th.get('separable') else 'no'} |")
    L.append("")
    for c in cfgs:
        th = c.get("threshold", {})
        if th.get("note"):
            L.append(f"- {c['name']}: {th['note']}")
        elif th.get("oos_above_suggested"):
            ids = ", ".join(f"{o['id']} ({o['score']:.3f})" for o in th["oos_above_suggested"])
            L.append(f"- {c['name']}: preguntas fuera de alcance con score ≥ umbral sugerido (dependen de que el LLM "
                     f"se abstenga): {ids}.")
    L.append("\nRecomendación: el umbral por score solo filtra preguntas claramente ajenas al dominio; las preguntas "
             "cercanas (mismo vocabulario, dato ausente) solo se detectan con la instrucción de abstención en el prompt.\n")

    # Successes and failures (all configs)
    primary = cfgs[0] if cfgs else None
    L.append("## 4. Ejemplos de éxito\n")
    if primary:
        succ = [r for r in primary["records"] if is_success(r)][:3]
        if not succ:
            L.append("_No hubo ítems que cumplieran todos los criterios de éxito._\n")
        for r in succ:
            top = r["retrieved"][0] if r.get("retrieved") else {}
            j = r.get("judge") or {}
            L.append(f"**{r['id']}** ({r['lang']}) — {r['question']}  ")
            L.append(f"Top-1: `{top.get('source')}` (score {_f(top.get('score'))}); keywords 100%; "
                     f"juez F={_f(j.get('faithfulness'))} R={_f(j.get('relevance'))}  ")
            L.append(f"> {_excerpt(r.get('answer'))}\n")

    L.append("## 5. Fallos observados (todos)\n")
    any_fail = False
    for c in cfgs:
        fails = [(r, failure_reasons(r)) for r in c["records"]]
        fails = [(r, rs) for r, rs in fails if rs]
        L.append(f"### {c['name']} — {len(fails)} ítem(s) con fallos\n")
        for r, rs in fails:
            any_fail = True
            tops = ", ".join(f"{source_key(x['source'])}({x['score']:.3f})" for x in r.get("retrieved", []))
            L.append(f"- **{r['id']}** [{r['type']}, {r.get('lang')}] {r['question']}")
            L.append(f"  - Motivo(s): {'; '.join(rs)}")
            if r.get("expected_source"):
                L.append(f"  - Fuente esperada: {r['expected_source']} · Recuperado: {tops or '–'}")
            elif tops:
                L.append(f"  - Recuperado: {tops}")
            if r.get("answer"):
                L.append(f"  - Respuesta: \"{_excerpt(r['answer'], 250)}\"")
        L.append("")
    if not any_fail:
        L.append("_No se observaron fallos._\n")

    # Limitations
    L.append("## 6. Limitaciones observadas\n")
    for line in observed_limitations(results):
        L.append(f"- {line}")
    L.append("\n**Advertencias generales**\n")
    L.append("- Corpus pequeño (7 archivos, ~16 KB) y dataset de pocas preguntas: una pregunta equivale a ~5 puntos porcentuales; las diferencias entre configuraciones no son estadísticamente concluyentes.")
    L.append("- La cobertura de keywords es una métrica léxica: penaliza paráfrasis válidas y premia respuestas que mencionan el término sin ser correctas.")
    L.append("- El juez LLM es de la misma familia (Gemini) que el generador, lo que puede inflar las puntuaciones (sesgo de autoevaluación); no se validó contra anotación humana.")
    L.append("- La abstención se detecta por la bandera `abstained` o por la frase de no-información; una negativa redactada de otra forma no cuenta como abstención.")
    L.append("- Pinecone serverless es eventualmente consistente; los scores pueden variar levemente entre corridas.")
    return "\n".join(L) + "\n"


def observed_limitations(results: dict) -> list[str]:
    out: list[str] = []
    for c in results.get("configs", []):
        m, th, n = c["metrics"], c.get("threshold", {}), c["name"]
        if m.get("hit@4") is not None and m["hit@4"] < 0.85:
            missed = [r["id"] for r in c["records"] if r["type"] == "answerable" and not r.get("error") and not r.get("rank")]
            out.append(f"[{n}] Hit@4 = {m['hit@4']:.0%} por debajo del objetivo 85%; fallos de recuperación: {', '.join(missed)}.")
        if m.get("correct_abstention_oos") is not None and m["correct_abstention_oos"] < 0.8:
            out.append(f"[{n}] Abstención correcta en fuera de alcance = {m['correct_abstention_oos']:.0%} (< 80%): el sistema responde preguntas cuyo dato no está en el corpus.")
        if m.get("false_abstention_answerable"):
            out.append(f"[{n}] Falsa abstención en {m['false_abstention_answerable']:.0%} de las preguntas respondibles (umbral demasiado alto o contexto insuficiente).")
        if th.get("separable") is False:
            out.append(f"[{n}] Los scores top-1 de respondibles y fuera de alcance se solapan: ningún umbral por score separa perfectamente ambos grupos.")
        if th.get("suggested") is not None and th.get("current") is not None and abs(th["suggested"] - th["current"]) > 0.05:
            out.append(f"[{n}] El umbral actual ({th['current']}) difiere del sugerido ({th['suggested']}); considerar ajustar RAG_SCORE_THRESHOLD (validar con más datos).")
        en, es = m.get("top1_answerable_en", {}).get("mean"), m.get("top1_answerable_es", {}).get("mean")
        if en is not None and es is not None and en - es > 0.03:
            out.append(f"[{n}] Las preguntas en español obtienen scores top-1 menores (media {es:.3f} vs {en:.3f} en inglés): el corpus está en inglés y la recuperación es cross-lingüe.")
        if m.get("hit@4_es") is not None and m.get("hit@4_en") is not None and m["hit@4_es"] < m["hit@4_en"]:
            out.append(f"[{n}] Hit@4 en español ({m['hit@4_es']:.0%}) menor que en inglés ({m['hit@4_en']:.0%}).")
        if m.get("keyword_coverage") is not None and m["keyword_coverage"] < 0.8:
            out.append(f"[{n}] Cobertura de keywords media de {m['keyword_coverage']:.0%}: respuestas incompletas o redactadas con sinónimos.")
        if m.get("citation_rate") is not None and m["citation_rate"] < 1.0:
            out.append(f"[{n}] {1 - m['citation_rate']:.0%} de las respuestas no abstenidas no citan ninguna fuente (incumple RF3/CA6).")
        if m.get("adversarial_pass") == 0:
            out.append(f"[{n}] La inyección de prompt no fue contenida.")
        if m.get("invalid_pass") is not None and m["invalid_pass"] < 1:
            out.append(f"[{n}] Alguna entrada inválida no fue rechazada con InvalidInputError.")
        if m.get("latency_p95_s") and m["latency_p95_s"] > 10:
            out.append(f"[{n}] Latencia p95 alta ({m['latency_p95_s']:.1f} s), probablemente por reintentos ante límites de cuota.")
        if m.get("n_errors"):
            out.append(f"[{n}] {m['n_errors']} ítem(s) terminaron con error (ver results.json); las métricas excluyen esos ítems.")
        if c.get("aborted"):
            out.append(f"[{n}] Corrida abortada: {c['aborted']}.")
    if not out:
        out.append("No se detectaron limitaciones cuantitativas por encima de los umbrales de alerta en esta corrida.")
    return out


# --------------------------------------------------------------------------- output


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def summary_table(results: dict) -> str:
    keys = [("Hit@1", "hit@1", True), ("Hit@4", "hit@4", True), ("MRR@4", "mrr@4", False),
            ("Abst.OOS", "correct_abstention_oos", True), ("FalseAbst", "false_abstention_answerable", True),
            ("KwCov", "keyword_coverage", True), ("Cite", "citation_rate", True), ("Adv", "adversarial_pass", True),
            ("Inv", "invalid_pass", True), ("Lat(s)", "latency_mean_s", False), ("Faith", "judge_faithfulness", False),
            ("Err", "n_errors", False)]
    header = f"{'config':<8}" + "".join(f"{k[0]:>10}" for k in keys)
    lines = [header, "-" * len(header)]
    for c in results.get("configs", []):
        lines.append(f"{c['name']:<8}" + "".join(f"{_f(c['metrics'].get(k[1]), k[2], 2):>10}" for k in keys))
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the MMA RAG assistant (retrieval, abstention, generation).")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL dataset path.")
    p.add_argument("--configs", default="cs800,cs400", help=f"Comma list of presets: {', '.join(PRESETS)}.")
    p.add_argument("--judge", dest="judge", action="store_true", default=True, help="LLM judge on the first config (default).")
    p.add_argument("--judge-model", default=os.getenv("RAG_JUDGE_MODEL", "gemini-3.1-flash-lite"),
                   help="Gemini model used as LLM judge (default: gemini-3.1-flash-lite).")
    p.add_argument("--no-judge", dest="judge", action="store_false", help="Disable the LLM judge.")
    p.add_argument("--sleep", type=float, default=4.0, help="Seconds to pause after each API call (free-tier RPM).")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items.")
    p.add_argument("--skip-ingest", action="store_true", help="Reuse existing exp-<config> namespaces.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    p.add_argument("--report-only", action="store_true", help="Regenerate REPORT.md from existing results.json.")
    p.add_argument("--max-consecutive-errors", type=int, default=5, help="Abort a config after N consecutive item errors (0 = never).")
    p.add_argument("--debug", action="store_true", help="Verbose logging and tracebacks.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    args = parse_args(argv)
    global JUDGE_MODEL
    JUDGE_MODEL = args.judge_model
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.debug:
        for noisy in ("httpx", "urllib3", "google", "pinecone"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    out_dir = Path(args.out_dir)
    results_path, report_path = out_dir / "results.json", out_dir / "REPORT.md"

    if args.report_only:
        if not results_path.exists():
            logger.error("No existe %s; ejecute primero la evaluación.", results_path)
            return 2
        results = json.loads(results_path.read_text(encoding="utf-8"))
        report_path.write_text(render_report(results), encoding="utf-8")
        print(summary_table(results))
        print(f"\nReporte regenerado: {report_path}")
        return 0

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in PRESETS]
    if not names or unknown:
        logger.error("Configuraciones desconocidas: %s (disponibles: %s)", unknown or "(vacío)", ", ".join(PRESETS))
        return 2

    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]

    try:
        deps = deps or load_real_deps()
        base = deps.load_settings()
    except (RAGError, ImportError) as exc:
        # ImportError: the pipeline modules do not (yet) expose the expected contract.
        logger.error("%s: %s", type(exc).__name__, exc)
        if args.debug:
            raise
        return 1

    by_type: dict[str, int] = {}
    for it in items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_model": getattr(base, "chat_model", None),
        "embedding_model": getattr(base, "embedding_model", None),
        "dataset": {"path": os.path.relpath(args.dataset, REPO_ROOT), "n_items": len(items), "by_type": by_type},
        "judge": args.judge,
        "judge_model": args.judge_model if args.judge else None,
        "sleep_s": args.sleep,
        "configs": [],
    }
    partial_path = out_dir / "results.partial.json"

    def on_progress(cfg: dict) -> None:
        write_json(partial_path, {**results, "configs": results["configs"] + [cfg]})

    for i, name in enumerate(names):
        try:
            cfg = evaluate_config(name, items, deps, use_judge=args.judge and i == 0, sleep_s=args.sleep,
                                  skip_ingest=args.skip_ingest, on_progress=on_progress,
                                  max_consecutive_errors=args.max_consecutive_errors)
        except RAGError as exc:
            logger.error("[%s] configuración fallida: %s", name, exc)
            if args.debug:
                logger.exception("traceback")
            cfg = {"name": name, "params": {**PRESETS[name], "top_k": EVAL_TOP_K, "namespace": f"exp-{name}"},
                   "judge": False, "ingestion": None, "records": [], "aborted": f"{type(exc).__name__}: {exc}",
                   "metrics": compute_metrics([]), "threshold": suggest_threshold([])}
        results["configs"].append(cfg)
        write_json(partial_path, results)

    write_json(results_path, results)
    report_path.write_text(render_report(results), encoding="utf-8")
    try:
        partial_path.unlink()
    except OSError:
        pass
    print(summary_table(results))
    print(f"\nResultados: {results_path}\nReporte:    {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
