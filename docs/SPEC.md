# SPEC — Asistente RAG de reglamento y puntuación MMA

## Problema
Jueces amateur, árbitros en formación y periodistas necesitan respuestas **rápidas y verificables** sobre reglas y criterios de puntuación de MMA (sistema 10-point must, rounds 10-8, faltas, tipos de decisión, categorías de peso). Hoy consultan PDFs largos, foros o memoria, sin trazabilidad. Un LLM genérico responde con fluidez pero puede **alucinar** reglas; un buscador por palabras clave no entiende paráfrasis ("¿cuándo doy 10-8?" ≠ "dominance, impact, duration").

## Usuarios
- **Juez amateur:** "¿cómo puntúo un round donde un peleador domina pero sin daño?"
- **Árbitro en formación:** "¿qué es una falta?, ¿cuándo se descuenta un punto?"
- **Periodista:** "¿qué es una decisión dividida?, ¿qué pasa si un peleador no da el peso?"

## Requisitos funcionales
- **RF1** Ingesta de archivos `.md/.txt` de `data/` → chunks → embeddings Gemini → Pinecone.
- **RF2** Consulta por CLI: `--question "..."` y modo interactivo.
- **RF3** Respuesta generada **solo** con el contexto recuperado, citando `[archivo]`.
- **RF4** Abstención explícita ("No tengo información suficiente…") si el score top-1 < umbral o el contexto no alcanza.
- **RF5** Mostrar las fuentes recuperadas con su score de similitud.

## Requisitos no funcionales
- Validación de entrada: vacía, solo espacios, más de `MAX_QUESTION_CHARS`, caracteres de control.
- Validación de configuración e índice al arrancar, con mensajes accionables.
- Reintentos con backoff exponencial ante 429/5xx/timeouts; errores claros sin stack trace (salvo `--debug`).
- Presupuesto de contexto (`MAX_CONTEXT_CHARS`) para respetar los límites del modelo.
- Parámetros centralizados en `config.py` (sobrescribibles por variables de entorno). Logging estándar.
- Tests automatizados que corren sin API keys.

## Criterios de aceptación
| ID | Criterio | Cómo se verifica |
|---|---|---|
| CA1 | `pytest` pasa sin red ni keys | `pytest -q` |
| CA2 | Pregunta vacía o demasiado larga → mensaje claro, código ≠ 0, sin llamar a APIs | `python main.py --question "   "` |
| CA3 | Falta una variable de entorno → mensaje con el nombre de la variable | test `test_config.py` |
| CA4 | Índice inexistente o vacío → sugiere `python ingest.py` | test `test_rag_chain.py` |
| CA5 | Hit@4 ≥ 0.85 y abstención correcta ≥ 0.8 (objetivo; se reporta el valor real) | `eval/results/REPORT.md` |
| CA6 | Toda respuesta no abstenida cita ≥ 1 fuente existente | eval + test de formato |

## Fuera de alcance
UI web; reglamento oficial completo o actualizado en tiempo real; conversación con memoria; fine-tuning; cambio de proveedor (se mantiene Gemini + Pinecone + LangChain).
