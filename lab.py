"""Builds both pipelines, retrieves top-3 for every question, scores them, writes the report.
Run:  python lab.py            (Chroma + Chroma's default embeddings; falls back to hashing if the model can't load)
Dry run without Chroma (testing only):  STORE=memory EMBEDDER=hashing OUT_DIR=dryrun python lab.py"""
import json, os
from pathlib import Path
from chunkers import (load_docs, fixed_size_chunks, section_chunks, norm,
                      FIXED_SIZE, FIXED_OVERLAP, SECTION_MAX_CHARS)
from embedding import get_embedder
from stores import make_store

K = 3
LABEL = {"fixed": "Fixed-size", "section": "Section-based"}


# ---------------------------------------------------------------- scoring
def is_relevant(chunk, q):
    """Relevant = comes from an expected source AND contains at least one evidence snippet."""
    return chunk["source"] in q["expected_sources"] and any(norm(e) in norm(chunk["text"]) for e in q["evidence"])


def score(retrieved, q):
    flags = [is_relevant(c, q) for c in retrieved]
    found = [any(norm(e) in norm(c["text"]) for c in retrieved) for e in q["evidence"]]
    return {"relevant": sum(flags), "retrieved": len(retrieved),
            "precision": sum(flags) / len(retrieved) if retrieved else 0.0,
            "coverage": sum(found) / len(found), "full": all(found), "flags": flags,
            "missed": [e for e, f in zip(q["evidence"], found) if not f]}


def evidence_intact(chunks, q):
    """Is each evidence snippet contained, unbroken, in at least one chunk of this index?"""
    return [any(c["source"] in q["expected_sources"] and norm(e) in norm(c["text"]) for c in chunks)
            for e in q["evidence"]]


def key(s):
    return (round(s["coverage"], 6), round(s["precision"], 6))


def winner(sf, ss):
    return "tie" if key(sf) == key(ss) else ("section" if key(ss) > key(sf) else "fixed")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------- pipeline
def run(store_kind="chroma", embedder_mode="auto", out_dir="."):
    docs = load_docs("docs")
    questions = json.loads(Path("questions.json").read_text(encoding="utf-8"))
    chunks = {"fixed": [c for d in docs for c in fixed_size_chunks(d)],
              "section": [c for d in docs for c in section_chunks(d)]}
    embed = get_embedder(embedder_mode)
    store = make_store(store_kind)
    for s, cs in chunks.items():  # identical embedder + store settings; only the chunks differ
        store.build(f"{s}_chunks", cs, embed([c["text"] for c in cs]))

    rows = []
    for q in questions:
        qv = embed([q["question"]])[0]
        row = {"q": q}
        for s in chunks:
            retrieved = store.query(f"{s}_chunks", qv, K)
            row[s] = {"retrieved": retrieved, "score": score(retrieved, q),
                      "intact": evidence_intact(chunks[s], q)}
        row["winner"] = winner(row["fixed"]["score"], row["section"]["score"])
        rows.append(row)

    meta = {"embedder": embed.name, "store": store.name, "k": K,
            "chunks": {s: {"count": len(cs), "avg_chars": round(mean([len(c["text"]) for c in cs]))}
                       for s, cs in chunks.items()}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(render_results(rows, meta), encoding="utf-8")
    (out / "analysis.md").write_text(render_analysis(rows, meta, chunks), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(
        [{"id": r["q"]["id"], "winner": r["winner"],
          **{s: {"precision": r[s]["score"]["precision"], "coverage": r[s]["score"]["coverage"],
                 "retrieved_ids": [c["id"] for c in r[s]["retrieved"]]} for s in chunks}} for r in rows],
        indent=2), encoding="utf-8")
    return rows, meta


# ---------------------------------------------------------------- reporting
def totals(rows, s):
    sc = [r[s]["score"] for r in rows]
    return {"precision": mean([x["precision"] for x in sc]), "coverage": mean([x["coverage"] for x in sc]),
            "full": sum(x["full"] for x in sc), "relevant": sum(x["relevant"] for x in sc),
            "retrieved": sum(x["retrieved"] for x in sc)}


def render_results(rows, meta):
    L = ["# Retrieval results", "",
         f"- Embedder (same for both pipelines): **{meta['embedder']}**",
         f"- Vector store: **{meta['store']}**, cosine distance, top-{meta['k']} per question",
         f"- Fixed-size: {FIXED_SIZE} chars, {FIXED_OVERLAP} overlap, {meta['chunks']['fixed']['count']} chunks, "
         f"avg {meta['chunks']['fixed']['avg_chars']} chars",
         f"- Section-based: one chunk per `##` section (title + heading + paragraphs, max {SECTION_MAX_CHARS} chars), "
         f"{meta['chunks']['section']['count']} chunks, avg {meta['chunks']['section']['avg_chars']} chars", "",
         "A retrieved chunk is **relevant** if it comes from an expected source and contains at least one evidence "
         "snippet. **Precision@3** = relevant / 3. **Coverage** = share of a question's evidence snippets found "
         "somewhere in the top-3 chunks (a question is fully answerable only at 100%).", "",
         "## Scoring table", "",
         "| Q | Type | Fixed P@3 | Section P@3 | Fixed coverage | Section coverage | Better |",
         "|---|------|-----------|-------------|----------------|------------------|--------|"]
    for r in rows:
        f, s = r["fixed"]["score"], r["section"]["score"]
        L.append(f"| {r['q']['id']} | {r['q']['type']} | {f['relevant']}/{f['retrieved']} ({f['precision']:.2f}) | "
                 f"{s['relevant']}/{s['retrieved']} ({s['precision']:.2f}) | {f['coverage']:.0%} | {s['coverage']:.0%} | "
                 f"{r['winner']} |")
    tf, ts = totals(rows, "fixed"), totals(rows, "section")
    n = len(rows)
    L += [f"| **Overall** | mean of {n} questions | **{tf['precision']:.2f}** | **{ts['precision']:.2f}** | "
          f"**{tf['coverage']:.0%}** | **{ts['coverage']:.0%}** | "
          f"{winner({'coverage': tf['coverage'], 'precision': tf['precision']}, {'coverage': ts['coverage'], 'precision': ts['precision']})} |",
          f"| **Fully answerable** | evidence 100% covered | {tf['full']}/{n} | {ts['full']}/{n} | | | |",
          "", "See [analysis.md](analysis.md) for the written analysis.", "", "## Retrieved chunks per question", ""]
    for r in rows:
        L += [f"### {r['q']['id']}: {r['q']['question']}",
              f"*Expected sources:* {', '.join(r['q']['expected_sources'])}  ",
              f"*Evidence:* " + "; ".join(f'"{e}"' for e in r["q"]["evidence"]), ""]
        for s in ("fixed", "section"):
            L.append(f"**{LABEL[s]}**")
            for i, (c, ok) in enumerate(zip(r[s]["retrieved"], r[s]["score"]["flags"]), 1):
                prev = " ".join(c["text"].split())[:110]
                sec = f" / {c['section']}" if c["section"] else ""
                L.append(f"{i}. {'✅' if ok else '❌'} `{c['source']}{sec}` (distance {c['distance']:.3f}): {prev}…")
            L.append("")
    return "\n".join(L)


def diagnose(strategy, snippet, q, chunks):
    present = any(c["source"] in q["expected_sources"] and norm(snippet) in norm(c["text"]) for c in chunks[strategy])
    return ("it is split across a chunk boundary, so no chunk in this index contains the full sentence" if not present
            else "a chunk containing it exists but was not ranked in the top 3")


def render_analysis(rows, meta, chunks):
    tf, ts = totals(rows, "fixed"), totals(rows, "section")
    n = len(rows)
    o = winner({"coverage": tf["coverage"], "precision": tf["precision"]},
               {"coverage": ts["coverage"], "precision": ts["precision"]})
    L = ["# Analysis", "", f"*Generated from the scores in `results.md` (embedder: {meta['embedder']}).*", ""]
    if o == "tie":
        L.append(f"**Overall verdict: no clear winner.** Both strategies scored mean precision@3 of "
                 f"{tf['precision']:.2f} and mean evidence coverage of {tf['coverage']:.0%}.")
    else:
        w, l = (ts, tf) if o == "section" else (tf, ts)
        L.append(f"**Overall winner: {LABEL[o]} chunking.** Mean precision@3 {w['precision']:.2f} vs {l['precision']:.2f}, "
                 f"mean evidence coverage {w['coverage']:.0%} vs {l['coverage']:.0%}, and "
                 f"{w['full']}/{n} vs {l['full']}/{n} questions had all of their evidence retrieved.")
    L += ["", "## Where the strategies differed", ""]
    diff = [r for r in rows if r["winner"] != "tie"]
    for r in diff:
        w, l = r["winner"], ("fixed" if r["winner"] == "section" else "section")
        ws, ls = r[w]["score"], r[l]["score"]
        line = (f"- **{r['q']['id']}** ({r['q']['type']}): {LABEL[w]} beat {LABEL[l]} "
                f"(coverage {ws['coverage']:.0%} vs {ls['coverage']:.0%}, precision@3 {ws['precision']:.2f} vs {ls['precision']:.2f}).")
        if ls["missed"]:
            why = "; ".join(f"“{m}” — {diagnose(l, m, r['q'], chunks)}" for m in ls["missed"])
            line += f" {LABEL[l]} missed: {why}."
        L.append(line)
    if not diff:
        L.append("- No question separated the two strategies.")
    ties = [r["q"]["id"] for r in rows if r["winner"] == "tie"]
    if ties:
        L += ["", f"Tied on identical scores: {', '.join(ties)}."]

    split = {s: sum(not ok for r in rows for ok in r[s]["intact"]) for s in ("fixed", "section")}
    total_ev = sum(len(r["q"]["evidence"]) for r in rows)
    L += ["", "## Structural facts (independent of the embedder)", "",
          f"- Evidence snippets left intact inside a single chunk: fixed {total_ev - split['fixed']}/{total_ev}, "
          f"section {total_ev - split['section']}/{total_ev}. "
          "A snippet that no chunk contains cannot be retrieved by any query.",
          f"- Average chunk size: fixed {meta['chunks']['fixed']['avg_chars']} chars "
          f"({meta['chunks']['fixed']['count']} chunks), section {meta['chunks']['section']['avg_chars']} chars "
          f"({meta['chunks']['section']['count']} chunks). Larger section chunks keep context together but can mix "
          "several topics in one vector.",
          "- Section chunks carry the document title and section heading, so facts such as “five working days” "
          "stay attached to what they refer to.",
          "", "## Caveats", "",
          f"- {n} questions on a small synthetic corpus: treat the result as evidence for this setup, not a general rule.",
          "- Scores depend on the embedder and on the chunk parameters (400/50 chars, 1200 max)."]
    return "\n".join(L)


if __name__ == "__main__":
    rows, meta = run(os.getenv("STORE", "chroma"), os.getenv("EMBEDDER", "auto"), os.getenv("OUT_DIR", "."))
    tf, ts = totals(rows, "fixed"), totals(rows, "section")
    print(f"Embedder: {meta['embedder']} | store: {meta['store']}")
    print(f"{'Q':4} {'Fixed P@3':>10} {'Section P@3':>12} {'Fixed cov':>10} {'Section cov':>12}  Better")
    for r in rows:
        f, s = r["fixed"]["score"], r["section"]["score"]
        print(f"{r['q']['id']:4} {f['precision']:>10.2f} {s['precision']:>12.2f} {f['coverage']:>10.0%} {s['coverage']:>12.0%}  {r['winner']}")
    print(f"{'ALL':4} {tf['precision']:>10.2f} {ts['precision']:>12.2f} {tf['coverage']:>10.0%} {ts['coverage']:>12.0%}")
    print("Wrote results.md, analysis.md, results.json to", os.getenv("OUT_DIR", "."))
