# Analysis

*Based on the scoring table in [results.md](results.md). Embedder: Chroma default (all-MiniLM-L6-v2), same for both pipelines. Top-3 retrieval, cosine distance.*

## Verdict

**Section-based chunking performed better overall.** Mean precision@3 was a tie (0.42 vs 0.42), but section-based retrieved **all** the evidence for every question (100% mean coverage, 8/8 questions fully answerable), while fixed-size reached 81% (6/8). Coverage is the more meaningful metric here, because a question can only be answered if all the evidence is retrieved.

| | Fixed-size | Section-based |
|---|---|---|
| Mean precision@3 | 0.42 | 0.42 |
| Mean evidence coverage | 81% | 100% |
| Questions fully answerable | 6/8 | 8/8 |
| Chunks / average size | 35 / 379 chars | 22 / 547 chars |

## Questions where section-based clearly won

- **Q3 (laptop return, 0% vs 100% coverage).** The answer sentence ("must be returned within five working days") never mentions a laptop. Fixed-size chunks that contain the sentence exist, but none ranked in the top 3; it retrieved remote-work, security and leave chunks instead. The section chunk is prefixed with the title and heading ("Remote Work Policy > Home Office Equipment") and keeps the laptop sentence in the same chunk, so it ranked first. My explanation is that the added context made the match, which fits the result but is an inference from this one run.
- **Q6 (stolen laptop abroad, 50% vs 100%).** The required sentence "report it to the Security Desk within one hour of noticing the loss" is cut across a chunk boundary, so no fixed-size chunk contains it and no query could retrieve it in full. The section chunk holds the whole "Lost or Stolen Devices" section, so both required steps were retrieved together. This is the clearest case of a fixed window splitting an answer.
- **Q5 (resignation and carry-over, precision 0.33 vs 0.67).** Both strategies retrieved all the evidence. Section-based scored higher because the two facts live in different sections ("Leaving the Company" and "Annual Leave") and it returned both. In the fixed index both facts sit in a single chunk, so only one chunk counted as relevant. This shows precision@3 reflects how many chunks hold the evidence, not only how good retrieval was.

## Questions where fixed-size scored higher precision (and why that is misleading)

- **Q1 (meal allowance).** Fixed-size scored 0.67 vs 0.33, but coverage was 100% for both. The sentence falls in the 50-character overlap between two neighbouring chunks, so both contain it and both count as relevant. Duplicated text is not better retrieval.
- **Q2 (client dinner, two facts).** Fixed-size scored 0.67 vs 0.33 with 100% coverage for both. Its two facts are in two separate chunks, each counted as relevant. Section-based put both facts in one chunk, so only one result counted. Section-based returned the full answer in its first result.

## Ties

Q4 (cross-document), Q7 (quiet room) and Q8 (sick note) had identical scores. Both strategies found the evidence, and for single facts precision@3 cannot exceed 0.33 unless overlap duplicates the evidence.

## Caveats

- Only 8 questions on a small synthetic corpus, so this is evidence for this setup, not a general rule.
- Results depend on the embedder and chunk parameters (400/50 characters; 1200-character section maximum). Section chunks are larger and can mix topics in one vector, which could hurt on a bigger corpus.
- Precision@3 is capped for single-fact questions and is affected by overlap, so coverage is the better comparison.