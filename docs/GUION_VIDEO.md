# Guion del video de demostración (≤ 5 min)

Asistente RAG de reglamento y puntuación MMA (LangChain + Pinecone + Gemini).

## Checklist antes de grabar

- [ ] Terminal en la raíz del repositorio con el entorno virtual activado (`source .venv/bin/activate`).
- [ ] `.env` completo con `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` y `GOOGLE_API_KEY`. Verifica que no aparezcan claves en pantalla: no abras `.env` durante la grabación.
- [ ] Ingesta hecha al menos una vez antes de grabar (`python ingest.py --reset`), para que la escena 3 sea rápida y el índice ya exista.
- [ ] Pregunta de la escena 4 ensayada una vez, para calentar la conexión y comprobar que no hay errores 429.
- [ ] Evaluación ya ejecutada: `eval/results/REPORT.md` existe y está abierto en una pestaña del editor o del navegador.
- [ ] README abierto en la vista previa de GitHub o del editor, con el diagrama Mermaid renderizado y desplazado hasta la sección 3.
- [ ] Terminal con fuente grande (18–20 pt), tema de alto contraste e historial limpio (`clear`).
- [ ] `pytest` instalado (`pip install pytest`) y `pytest -q` probado antes.
- [ ] Notificaciones del sistema silenciadas.

## Resumen de tiempos

| # | Tiempo | Escena | Qué se muestra |
|---|---|---|---|
| 1 | 0:00–0:30 | Problema | README, sección 1 |
| 2 | 0:30–1:15 | Arquitectura | Diagrama Mermaid y tabla de módulos del README |
| 3 | 1:15–1:45 | Ingesta | `python ingest.py --reset` |
| 4 | 1:45–2:30 | Pregunta normal con citas | `python main.py --question "¿Cuándo se debe puntuar un round 10-8?"` |
| 5 | 2:30–3:00 | Fuera de alcance | `python main.py --question "What was the gate revenue of UFC 205?"` |
| 6 | 3:00–3:30 | Entrada inválida y error manejado | Pregunta en blanco y clave faltante, con `echo $?` |
| 7 | 3:30–4:15 | Resultados de evaluación | `eval/results/REPORT.md` y `pytest -q` |
| 8 | 4:15–4:40 | Limitaciones y cierre | README, sección 9 |

Quedan unos 20 s de margen para no pasar de 5 minutos.

---

## Escena 1: Problema (0:00–0:30)

**Mostrar:** el README en la sección 1, con la tabla de usuarios y preguntas.

**Comando:** ninguno.

**Narración:**
> "Jueces amateur, árbitros en formación y periodistas necesitan respuestas rápidas y verificables sobre las reglas de MMA: cuándo se da un 10-8, qué es una falta o qué es una decisión dividida. Hoy consultan PDFs largos o foros, sin saber de dónde sale cada respuesta. Construimos un asistente RAG que responde solo con documentos indexados, cita la fuente y se abstiene cuando no sabe."

## Escena 2: Arquitectura (0:30–1:15)

**Mostrar:** el diagrama Mermaid del README (sección 3). Recorre primero el bloque de ingesta y después el de consulta. Termina con la tabla de módulos.

**Comando:** ninguno.

**Narración:**
> "En la ingesta, los documentos de `data/` se dividen en chunks de 800 caracteres, se convierten en embeddings con Gemini y se guardan en Pinecone con métrica coseno. En la consulta, la pregunta se valida antes de cualquier llamada a una API, se recuperan los cuatro chunks más similares con su score y, si el mejor score no supera el umbral, el sistema se abstiene sin llamar al LLM. Si lo supera, el contexto se envía delimitado como datos, lo que nos protege de prompt injection, y Gemini responde citando cada archivo. Elegimos RAG porque una búsqueda por palabras clave no entiende paráfrasis ni preguntas en español, y un LLM solo puede inventar reglas."

## Escena 3: Ingesta (1:15–1:45)

**Mostrar:** la terminal.

**Comando:**
```bash
python ingest.py --reset
```

**Narración:**
> "La ingesta lee los siete archivos del corpus, que son resúmenes educativos en inglés. Los divide y los sube por lotes a Pinecone. Con `--reset` vaciamos el namespace antes de indexar, y como los IDs son deterministas, re-ingestar nunca duplica vectores. Si las reglas cambian, basta con volver a correr este comando: no hay que re-entrenar nada."

**Qué señalar en pantalla:** la línea `Loaded 7 files -> N chunks (size=800, overlap=120)` y la línea final `Ingestion completed`.

## Escena 4: Pregunta normal con citas (1:45–2:30)

**Mostrar:** la terminal.

**Comando:**
```bash
python main.py --question "¿Cuándo se debe puntuar un round 10-8?"
```

**Narración:**
> "Hacemos la pregunta en español aunque el corpus está en inglés, y los embeddings multilingües encuentran el chunk correcto. La respuesta llega en español y cita el archivo entre corchetes, en este caso `scoring_criteria_detail.md`. Debajo aparecen las fuentes recuperadas con su chunk y su score de similitud, así que el juez puede ir al documento y verificar la regla."

**Qué señalar en pantalla:** la cita `[data/...]` dentro de la respuesta y la lista `Sources:` con los scores.

## Escena 5: Pregunta fuera de alcance (2:30–3:00)

**Mostrar:** la terminal.

**Comando:**
```bash
python main.py --question "What was the gate revenue of UFC 205?"
```

**Narración:**
> "Ahora preguntamos algo que no está en el corpus: la taquilla de UFC 205. El sistema no inventa una cifra y responde con el mensaje fijo de abstención. La línea entre paréntesis indica por qué se abstuvo: el score quedó por debajo del umbral, o el modelo indicó que el contexto era insuficiente."

**Nota:** lee en voz alta la línea `(Abstención: ...)` que aparezca. Puede ser cualquiera de las dos razones, según el score que obtenga la pregunta.

## Escena 6: Entrada inválida y error manejado (3:00–3:30)

**Mostrar:** la terminal.

**Comandos:**
```bash
python main.py --question "   "; echo $?
GOOGLE_API_KEY= python main.py --question "test"; echo $?
```

**Resultado esperado:**
- Primer comando: `Entrada inválida: La pregunta no puede estar vacía ni contener solo espacios. / ...` seguido de `2`.
- Segundo comando: `Error: Missing required environment variables: GOOGLE_API_KEY. Copy .env.example to .env and fill in the values.` seguido de `1`.

**Por qué funciona el segundo comando:** `load_dotenv()` no sobrescribe variables que ya existen en el entorno, aunque estén vacías. `config.load_settings` trata el valor vacío como ausente y reporta la variable por nombre, sin hacer ninguna llamada de red.

**Alternativa**, si prefieres mostrar un error de índice:
```bash
python main.py --question "test" --namespace no-existe; echo $?
```
Muestra lo siguiente, seguido de `1`:
```text
Error: Pinecone index '<índice>' has no vectors in namespace 'no-existe'. Run `python ingest.py --namespace no-existe` first (new vectors may take a few seconds to appear).
```

**Narración:**
> "La robustez también se evalúa. Una pregunta vacía se rechaza antes de llamar a cualquier API, con un mensaje claro y el código de salida 2. Si falta una clave, el sistema dice exactamente qué variable falta y termina con código 1, sin traza de error. Ante límites de cuota o timeouts, reintenta con backoff exponencial."

## Escena 7: Resultados de evaluación (3:30–4:15)

**Mostrar:** `eval/results/REPORT.md`, primero la tabla de la sección 1 y los criterios de aceptación, después la sección 2 (experimentos) y la sección 3 (umbral). Termina en la terminal con los tests.

**Comando:**
```bash
pytest -q
```

**Narración:**
> "Evaluamos el sistema con 29 preguntas: 21 respondibles en inglés y español, 5 fuera de alcance, 1 intento de prompt injection y 2 entradas inválidas. Medimos Hit@k, MRR, abstención correcta, citación y un juez LLM de fidelidad. Comparamos chunks de 800 y 400 caracteres y top-k de 2 y 4, y calibramos el umbral con estos datos. Además, la suite de tests corre en segundos sin red ni API keys."

**Nota:** al grabar, lee en voz alta dos o tres cifras reales del REPORT, por ejemplo Hit@4, abstención correcta y el umbral sugerido. No las escribas en este guion antes de tener los resultados.

## Escena 8: Limitaciones y cierre (4:15–4:40)

**Mostrar:** el README, sección 9 (limitaciones y trabajo futuro).

**Comando:** ninguno.

**Narración:**
> "Las limitaciones: el corpus es pequeño y no oficial, las reglas varían por comisión, la métrica de keywords es léxica y el juez es de la misma familia que el generador. Como trabajo futuro proponemos un reranker, búsqueda híbrida con BM25, el reglamento oficial en PDF y una interfaz web. Gracias."
