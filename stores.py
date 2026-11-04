"""Vector stores. Chroma is the real one; MemoryStore is only for offline dry runs/tests."""


class ChromaStore:
    name = "chroma"

    def __init__(self, path="chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.cols = {}

    def build(self, name, chunks, vectors):
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        col = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"},
                                            embedding_function=None)  # vectors are supplied by us
        col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                embeddings=vectors,
                metadatas=[{"source": c["source"], "section": c["section"], "strategy": c["strategy"]}
                           for c in chunks])
        self.cols[name] = col

    def query(self, name, vector, k):
        res = self.cols[name].query(query_embeddings=[vector], n_results=k,
                                    include=["documents", "metadatas", "distances"])
        return [{"id": res["ids"][0][i], "text": res["documents"][0][i],
                 "source": res["metadatas"][0][i]["source"], "section": res["metadatas"][0][i]["section"],
                 "distance": float(res["distances"][0][i])} for i in range(len(res["ids"][0]))]


class MemoryStore:
    name = "memory (dry run only)"

    def __init__(self):
        self.cols = {}

    def build(self, name, chunks, vectors):
        import numpy as np
        self.cols[name] = (chunks, np.array(vectors, dtype=float))

    def query(self, name, vector, k):
        import numpy as np
        chunks, mat = self.cols[name]
        q = np.array(vector, dtype=float)
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
        order = np.argsort(-sims, kind="stable")[:k]
        return [{"id": chunks[i]["id"], "text": chunks[i]["text"], "source": chunks[i]["source"],
                 "section": chunks[i]["section"], "distance": float(1 - sims[i])} for i in order]


def make_store(kind="chroma", path="chroma_db"):
    return MemoryStore() if kind == "memory" else ChromaStore(path)
