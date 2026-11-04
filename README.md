# RAG Retrieval Quality Lab (#evn-lf26-ai14)

Two versions of the same RAG retrieval pipeline that differ **only** in chunking strategy, indexed in Chroma and scored against questions with known relevant sources.

## What is compared
| | Pipeline A: fixed-size | Pipeline B: section-based |
|---|---|---|
| Chunking | sliding window, 400 characters, 50 overlap | one chunk per `##` section (title + heading + whole paragraphs; split at paragraph boundaries above 1200 chars) |
| Embeddings | same embedder for both (Chroma's default all-MiniLM-L6-v2) | same |
| Store / retrieval | Chroma collection, cosine distance, top-3 | same |

## Files
- `docs/` – 6 short policy documents (expense, remote work, security, onboarding, leave, office FAQ). Several answers need context from multiple sentences, paragraphs, sections or documents.
- `chunkers.py` – the two chunking strategies (the only variable)
- `embedding.py`, `stores.py` – shared embedder and Chroma store
- `questions.json` – 8 questions with expected source files and evidence snippets
- `lab.py` – indexes both pipelines, retrieves top-3, scores, writes the reports
- `results.md` – scoring table + the retrieved chunks for every question (generated)
- `analysis.md` – written analysis generated from the scores (generated)
- `tests/test_lab.py` – offline tests (evidence really exists in the expected docs, chunker behaviour, scoring maths)

## Scoring
A retrieved chunk is **relevant** if it comes from an expected source and contains at least one of the question's evidence snippets. 
- **Precision@3** = relevant chunks / 3 retrieved.
- **Coverage** = share of the question's evidence snippets found in the top-3 chunks. Multi-sentence questions only count as fully answerable at 100%.

Because chunk boundaries differ between pipelines, relevance is defined by evidence text, not chunk IDs.

## Run
```
pip install -r requirements.txt
python -m tests.test_lab      # offline checks
python lab.py                 # builds both Chroma indexes, writes results.md, analysis.md, results.json
```
The first run downloads Chroma's small embedding model (needs internet). If it can't load, `lab.py` falls back to a simple hashing embedder and says so at the top of `results.md`; for real results use the default model (`EMBEDDER=default python lab.py` fails loudly instead of falling back).

If `pip install chromadb` fails on a very new Python version, use Python 3.12 or 3.13.

## Results and analysis
See [`results.md`](results.md) for the scoring table and retrieved chunks and [`analysis.md`](analysis.md) for the analysis (which strategy won overall, the specific questions where they differed, and why).

### My notes
Section-based chunking won overall: both pipelines tied on mean precision@3 (0.42), but section-based retrieved all the evidence for every question (100% coverage, 8/8 fully answerable) while fixed-size reached 81% (6/8). The biggest differences were Q3, where fixed-size ranked irrelevant chunks above the one containing the answer, and Q6, where a fixed window cut a required sentence in half so no chunk contained it. Fixed-size's higher precision on Q1 and Q2 came from overlap duplicating a sentence (Q1) and from facts spread over two chunks (Q2), not from better retrieval, so I treated coverage as the more reliable metric.