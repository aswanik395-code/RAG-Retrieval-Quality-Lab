"""Offline tests (no Chroma, no model download). Run: python -m tests.test_lab"""
import json, tempfile
from pathlib import Path
import chunkers, lab
from chunkers import norm

DOCS = chunkers.load_docs("docs")
QS = json.loads(Path("questions.json").read_text(encoding="utf-8"))

def test_evidence_exists_in_expected_sources():
    for q in QS:
        for e in q["evidence"]:
            hits = [d["source"] for d in DOCS if norm(e) in norm(d["text"])]
            assert hits and all(h in q["expected_sources"] for h in hits), (q["id"], e, hits)

def test_at_least_5_questions():
    assert len(QS) >= 5

def test_fixed_chunks_respect_size_and_cover_text():
    for d in DOCS:
        cs = chunkers.fixed_size_chunks(d)
        assert all(len(c["text"]) <= chunkers.FIXED_SIZE for c in cs)
        assert norm(cs[0]["text"])[:30] == norm(d["text"])[:30]
        assert norm(d["text"])[-30:] in norm(cs[-1]["text"])

def test_section_chunks_keep_paragraphs_whole():
    for d in DOCS:
        cs = chunkers.section_chunks(d)
        body = [p.strip() for p in d["text"].split("\n\n") if p.strip() and not p.startswith("#")]
        joined = "\n".join(c["text"] for c in cs)
        for p in body:
            assert p.split("\n")[-1].strip() in joined
        assert all(c["text"].startswith(f"{d['text'].splitlines()[0][2:]} > ") for c in cs)

def test_score_math():
    q = {"expected_sources": ["a.md"], "evidence": ["alpha beta", "gamma"]}
    r = [{"source": "a.md", "text": "ALPHA   beta"}, {"source": "b.md", "text": "gamma"}, {"source": "a.md", "text": "none"}]
    s = lab.score(r, q)
    assert s["relevant"] == 1 and abs(s["precision"] - 1 / 3) < 1e-9 and s["coverage"] == 1.0 and s["full"]

def test_dry_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        rows, meta = lab.run("memory", "hashing", tmp)
        assert len(rows) == len(QS)
        assert all(len(r[s]["retrieved"]) == 3 for r in rows for s in ("fixed", "section"))
        for f in ("results.md", "analysis.md", "results.json"):
            assert (Path(tmp) / f).read_text(encoding="utf-8").strip()

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
