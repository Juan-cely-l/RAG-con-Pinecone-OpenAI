# Evaluación del asistente RAG

`eval/run_eval.py` mide recuperación, abstención y calidad de respuesta sobre `eval/dataset.jsonl`
(29 preguntas basadas solo en el contenido de `data/`) y genera `eval/results/results.json` y
`eval/results/REPORT.md`.

## Cómo correrlo

Requiere `.env` con `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` y `GOOGLE_API_KEY`. Desde la raíz del repo:

```bash
python eval/run_eval.py                          # cs800 + cs400, juez LLM en cs800
python eval/run_eval.py --configs cs800 --no-judge --limit 5   # prueba rápida
python eval/run_eval.py --skip-ingest            # reutiliza los namespaces exp-cs800 / exp-cs400
python eval/run_eval.py --report-only            # regenera REPORT.md desde results.json
```

Opciones: `--dataset`, `--configs` (`cs800` = chunk 800/overlap 120, producción; `cs400` = 400/60),
`--judge/--no-judge`, `--sleep` (segundos entre llamadas, por defecto 4 para el free tier),
`--limit N`, `--skip-ingest`, `--out-dir`, `--max-consecutive-errors`, `--debug`.

Cada configuración se ingesta en su propio namespace (`exp-<config>`, con `reset=True`), así que no toca
el namespace de producción. Tras cada ítem se guarda `results.partial.json`: si se agota la cuota no se
pierde lo ya evaluado.

## Dataset

Cada línea: `id`, `question`, `type`, `lang`, `expected_source` (ruta o lista de rutas aceptables; `null`
si no es respondible), `expected_keywords` (cada elemento es un string o una lista de alternativas, p. ej.
`["lb", "libra"]`; se comparan sin mayúsculas ni tildes), `notes`. Tipos: `answerable` (21, 8 en español),
`out_of_scope` (5), `adversarial` (1, con `forbidden_keywords`) e `invalid` (2).

## Métricas

| Métrica | Significado |
|---|---|
| Hit@k (1, 2, 4) | Fracción de preguntas respondibles cuya fuente esperada aparece entre los k primeros chunks recuperados. Hit@2 vs Hit@4 compara top_k=2 vs 4 sobre la misma lista. |
| MRR@4 | Promedio de 1/posición de la primera fuente correcta (0 si no está en el top-4). |
| Abstención correcta (OOS) | Fracción de preguntas fuera de alcance en las que el sistema dice que no tiene información. |
| Falsa abstención | Fracción de preguntas respondibles en las que el sistema se abstuvo. |
| Cobertura de keywords | Promedio de keywords esperadas presentes en la respuesta (solo respondibles no abstenidas). Métrica léxica. |
| Tasa de citación | Fracción de respuestas no abstenidas que citan una fuente (`[data/...]` o nombre de archivo). |
| Adversarial / inválida | La inyección se contiene (se abstiene o no aparecen palabras prohibidas); las entradas inválidas lanzan `InvalidInputError` sin llamar APIs. |
| Latencia media / p95 | `latency_s` de `answer_question`. |
| Juez LLM (1-5) | Gemini con temperatura 0 califica fidelidad al contexto y relevancia (solo configuración primaria). |
| Umbral sugerido | Valor del score top-1 que maximiza la exactitud balanceada entre respondibles y fuera de alcance. |

## Costo y tiempo estimados (corrida completa)

Por configuración, con 27 preguntas que llegan a la API (las 2 inválidas no llaman a nada):

- Ingesta: 1–2 solicitudes batch de embeddings (~25 chunks en cs800, ~50 en cs400) + upsert en Pinecone.
- `retrieve`: 27 embeddings de consulta; `answer_question`: 27 embeddings más y hasta 27 llamadas al chat
  (menos si se abstiene por umbral, porque entonces no llama al LLM).
- Juez: hasta 21 llamadas al chat (solo cs800).

Total con las dos configuraciones: ~108 embeddings de consulta, ≤ 54 llamadas de generación y ≤ 21 del juez
(≤ 54 llamadas al generador y ≤ 21 al juez, que usa otro modelo con cuota propia; `--judge-model`). Con `--sleep 4` tarda unos 12–15 minutos. Con `--no-judge` y una
sola configuración son ~54 embeddings y ≤ 27 llamadas al chat (~6 minutos).
