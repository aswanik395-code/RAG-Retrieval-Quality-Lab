"""One embedder shared by BOTH pipelines, so chunking is the only variable."""
import re, zlib

STOP = set("a an the of to in on for and or is are be it its as at by with from that this if can i my do does what how when "
           "which who will must may not than then so".split())


class HashingEmbedder:
    """Dependency-light fallback: hashed bag of words + bigrams (deterministic, lexical)."""
    name = "hashing-bow-1024 (fallback)"
    dim = 1024

    def __call__(self, texts):
        import numpy as np
        out = []
        for t in texts:
            toks = [w for w in re.findall(r"[a-z0-9$%]+", t.lower()) if w not in STOP]
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            v = np.zeros(self.dim)
            for f in feats:
                v[zlib.crc32(f.encode()) % self.dim] += 1.0
            v = np.log1p(v)
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


class ChromaDefaultEmbedder:
    """Chroma's built-in local model (all-MiniLM-L6-v2 via ONNX). No API key needed."""
    name = "chroma-default (all-MiniLM-L6-v2)"

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self.ef = DefaultEmbeddingFunction()
        self.ef(["warm-up"])  # downloads the model on first use; raises if unavailable

    def __call__(self, texts):
        return [[float(x) for x in v] for v in self.ef(list(texts))]


def get_embedder(mode="auto"):
    if mode in ("auto", "default"):
        try:
            return ChromaDefaultEmbedder()
        except Exception as e:
            if mode == "default":
                raise
            print(f"[embedder] Chroma default model unavailable ({type(e).__name__}); using hashing fallback.")
    return HashingEmbedder()
