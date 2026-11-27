# Reporte de evaluación — Asistente RAG de reglamento MMA

- Fecha (UTC): 2026-11-27T02:32:07+00:00
- Modelo de chat: `gemini-3.5-flash-lite` · Embeddings: `models/gemini-embedding-001` · Juez: `gemini-3.1-flash-lite`
- Dataset: `eval/dataset.jsonl` — 29 ítems ({'answerable': 21, 'out_of_scope': 5, 'adversarial': 1, 'invalid': 2})
- Configuraciones: cs800, cs400 (top_k=4); juez LLM solo en la primaria (cs800)

## 1. Resumen por configuración

| Métrica | cs800 (800/120) | cs400 (400/60) |
|---|---|---|
| Hit@1 | 100% | 100% |
| Hit@2 | 100% | 100% |
| Hit@4 | 100% | 100% |
| MRR@4 | 1.000 | 1.000 |
| Abstención correcta (OOS) | 100% | 100% |
| Falsa abstención (respondibles) | 0% | 0% |
| Cobertura de keywords | 88% | 91% |
| Tasa de citación | 100% | 100% |
| Adversarial pasa | 100% | 100% |
| Entrada inválida rechazada | 100% | 100% |
| Latencia media (s) | 1.652 | 1.762 |
| Latencia p95 (s) | 2.212 | 2.204 |
| Juez: fidelidad (1-5) | 5.000 | – |
| Juez: relevancia (1-5) | 5.000 | – |
| Ítems con error | 0 | 0 |

### Criterios de aceptación (configuración primaria)

- **CA5** (Hit@4 ≥ 0.85 y abstención correcta ≥ 0.8): Hit@4 = 100%, abstención = 100% → **CUMPLE**
- **CA6** (toda respuesta no abstenida cita ≥ 1 fuente): tasa de citación = 100% → **CUMPLE**

## 2. Experimentos

### 2.1 Tamaño de chunk: cs800 vs cs400

| Config | chunk/overlap | chunks indexados | Hit@1 | Hit@4 | MRR@4 | Cobertura kw | Score top-1 medio (resp.) | Latencia media |
|---|---|---|---|---|---|---|---|---|
| cs800 | 800/120 | 24 | 100% | 100% | 1.000 | 88% | 0.740 | 1.65 s |
| cs400 | 400/60 | 51 | 100% | 100% | 1.000 | 91% | 0.767 | 1.76 s |

- MRR@4: empate entre cs800 y cs400 (1.000).
- Hit@4: empate entre cs800 y cs400 (1.000).
- cobertura de keywords: mejor **cs400** (cs800 0.881 vs cs400 0.913).
- Nota: los chunks pequeños dan contexto más preciso pero fragmentan listas (p. ej. faltas o categorías de peso) y elevan el número de vectores/embeddings; con un corpus de ~16 KB la diferencia es pequeña y debe leerse con cautela (n reducido).

### 2.2 top_k = 2 vs top_k = 4 (derivado de la misma lista ordenada)

| Config | Hit@2 | Hit@4 | Δ (ganancia por recuperar 4) |
|---|---|---|---|
| cs800 | 100% | 100% | 0% |
| cs400 | 100% | 100% | 0% |

Con k=2 el contexto enviado al LLM es aproximadamente la mitad (menos tokens, menor latencia y costo), pero se pierde la fuente correcta en los casos que solo aparecen en las posiciones 3–4. La generación se evaluó con k=4; el efecto de k=2 sobre la respuesta no se midió directamente.

## 3. Calibración del umbral de abstención

El umbral se aplica al score top-1 de recuperación (similitud coseno). Se busca el valor que maximiza la exactitud balanceada = promedio(no abstenerse en respondibles, abstenerse en fuera de alcance), usando solo la señal de recuperación.

| Config | Top-1 respondibles (min/media/max) | Top-1 fuera de alcance (min/media/max) | Umbral actual (exactitud bal.) | Umbral sugerido (exactitud bal.) | ¿Separables? |
|---|---|---|---|---|---|
| cs800 | 0.642 / 0.740 / 0.813 | 0.604 / 0.642 / 0.707 | 0.500 (50%) | 0.708 (88%) | no |
| cs400 | 0.685 / 0.767 / 0.848 | 0.609 / 0.648 / 0.718 | 0.500 (50%) | 0.720 (93%) | no |


Recomendación: el umbral por score solo filtra preguntas claramente ajenas al dominio; las preguntas cercanas (mismo vocabulario, dato ausente) solo se detectan con la instrucción de abstención en el prompt.

## 4. Ejemplos de éxito

**ans-01** (en) — Under the 10-point must system, how many points does the winner of a round usually receive, and how many does the other fighter get?  
Top-1: `data/scoring_criteria_detail.md` (score 0.741); keywords 100%; juez F=5 R=5  
> Under the 10-point must system, the winner of the round typically receives 10 points, while the other fighter usually receives 9 points, or fewer in more dominant rounds [data/mma_rules_and_judging.md].

**ans-02** (en) — How many rounds and how long per round is a standard professional MMA fight, and how does that change for main events and title fights?  
Top-1: `data/mma_rules_and_judging.md` (score 0.726); keywords 100%; juez F=5 R=5  
> Most professional MMA fights consist of 3 rounds of 5 minutes each [data/mma_rules_and_judging.md]. For main events and title fights, this changes to 5 rounds of 5 minutes each [data/mma_rules_and_judging.md].

**ans-04** (es) — ¿Cuándo debería un juez dar un round 10-8?  
Top-1: `data/scoring_criteria_detail.md` (score 0.809); keywords 100%; juez F=5 R=5  
> Un juez debería otorgar un round 10-8 cuando un peleador gana por un amplio margen [data/scoring_criteria_detail.md]. Para ello, los jueces consideran tres factores: - **Impacto:** la ofensiva del peleador disminuye claramente la capacidad del oponente (por ejemplo, caídas o daño visible) [data/scor…

## 5. Fallos observados (todos)

### cs800 — 2 ítem(s) con fallos

- **ans-11** [answerable, en] What happens if an intentional foul causes an injury that stops the fight?
  - Motivo(s): cobertura de keywords baja (0%; faltan: disqualif)
  - Fuente esperada: ['data/unified_rules_fouls.md', 'data/fight_outcomes_and_officials.md'] · Recuperado: unified_rules_fouls.md(0.813), fight_outcomes_and_officials.md(0.774), unified_rules_fouls.md(0.771), fight_outcomes_and_officials.md(0.748)
  - Respuesta: "Si una falta intencional causa una lesión que detiene la pelea, el peleador que cometió la falta generalmente pierde por descalificación [data/unified_rules_fouls.md]."
- **ans-15** [answerable, en] Who is the only official inside the fighting area during an MMA bout, and what is their main duty?
  - Motivo(s): cobertura de keywords baja (0%; faltan: referee, safety)
  - Fuente esperada: data/fight_outcomes_and_officials.md · Recuperado: fight_outcomes_and_officials.md(0.745), fight_outcomes_and_officials.md(0.706), fight_outcomes_and_officials.md(0.685), mma_rules_and_judging.md(0.677)
  - Respuesta: "El árbitro es el único oficial dentro del área de combate [data/fight_outcomes_and_officials.md]. Su deber principal es la seguridad de los peleadores: detener la pelea cuando un peleador ya no puede defenderse inteligentemente [data/fight_outcomes_a…"

### cs400 — 1 ítem(s) con fallos

- **ans-15** [answerable, en] Who is the only official inside the fighting area during an MMA bout, and what is their main duty?
  - Motivo(s): cobertura de keywords baja (0%; faltan: referee, safety)
  - Fuente esperada: data/fight_outcomes_and_officials.md · Recuperado: fight_outcomes_and_officials.md(0.792), fight_outcomes_and_officials.md(0.733), fight_outcomes_and_officials.md(0.700), fight_outcomes_and_officials.md(0.696)
  - Respuesta: "El árbitro es el único oficial dentro del área de combate [data/fight_outcomes_and_officials.md]. Su deber principal es la seguridad de los peleadores: detener la pelea cuando un peleador ya no puede defenderse inteligentemente [data/fight_outcomes_a…"

## 6. Limitaciones observadas

- [cs800] Los scores top-1 de respondibles y fuera de alcance se solapan: ningún umbral por score separa perfectamente ambos grupos.
- [cs800] El umbral actual (0.5) difiere del sugerido (0.708); considerar ajustar RAG_SCORE_THRESHOLD (validar con más datos).
- [cs400] Los scores top-1 de respondibles y fuera de alcance se solapan: ningún umbral por score separa perfectamente ambos grupos.
- [cs400] El umbral actual (0.5) difiere del sugerido (0.7202); considerar ajustar RAG_SCORE_THRESHOLD (validar con más datos).

**Advertencias generales**

- Corpus pequeño (7 archivos, ~16 KB) y dataset de pocas preguntas: una pregunta equivale a ~5 puntos porcentuales; las diferencias entre configuraciones no son estadísticamente concluyentes.
- La cobertura de keywords es una métrica léxica: penaliza paráfrasis válidas y premia respuestas que mencionan el término sin ser correctas.
- El juez LLM es de la misma familia (Gemini) que el generador, lo que puede inflar las puntuaciones (sesgo de autoevaluación); no se validó contra anotación humana.
- La abstención se detecta por la bandera `abstained` o por la frase de no-información; una negativa redactada de otra forma no cuenta como abstención.
- Pinecone serverless es eventualmente consistente; los scores pueden variar levemente entre corridas.
