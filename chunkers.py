"""The ONLY thing that differs between the two pipelines: how documents are split into chunks."""
import re
from pathlib import Path

FIXED_SIZE = 400
FIXED_OVERLAP = 50
SECTION_MAX_CHARS = 1200


def norm(s):
    """Lower-case and collapse whitespace so evidence matching ignores line breaks."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_docs(folder="docs"):
    return [{"source": p.name, "text": p.read_text(encoding="utf-8")}
            for p in sorted(Path(folder).glob("*.md"))]


def fixed_size_chunks(doc, size=FIXED_SIZE, overlap=FIXED_OVERLAP):
    """Strategy A: blind sliding window of `size` characters with `overlap`."""
    text, step, chunks, i, n = doc["text"], size - overlap, [], 0, 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append({"id": f"fixed::{doc['source']}::{n}", "text": piece, "source": doc["source"],
                           "section": "", "strategy": "fixed"})
            n += 1
        if i + size >= len(text):
            break
        i += step
    return chunks


def section_chunks(doc, max_chars=SECTION_MAX_CHARS):
    """Strategy B: one chunk per '## section' (title + heading + whole paragraphs).
    Oversized sections are split at paragraph boundaries, never mid-sentence."""
    lines = doc["text"].splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), doc["source"])
    sections, heading, buf = [], None, []
    for line in lines:
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(buf)))
            heading, buf = line[3:].strip(), []
        elif heading is not None:
            buf.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(buf)))

    chunks, n = [], 0
    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        pieces, current = [], []
        for p in paragraphs:
            if current and len("\n\n".join(current + [p])) > max_chars:
                pieces.append(current)
                current = []
            current.append(p)
        if current:
            pieces.append(current)
        for piece in pieces:
            chunks.append({"id": f"section::{doc['source']}::{n}",
                           "text": f"{title} > {heading}\n\n" + "\n\n".join(piece),
                           "source": doc["source"], "section": heading, "strategy": "section"})
            n += 1
    return chunks
