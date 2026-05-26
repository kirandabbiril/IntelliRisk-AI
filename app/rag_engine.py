# rag_engine.py — lightweight RAG without ChromaDB
# Uses simple TF-IDF style cosine similarity — works on any Python version

import math
import re
from knowledge_base import ALL_DOCUMENTS


def tokenize(text: str) -> list:
    """Simple word tokenizer."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def build_tf(tokens: list) -> dict:
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    return {k: v / total for k, v in tf.items()}


def build_idf(docs: list) -> dict:
    n = len(docs)
    idf = {}
    all_words = set(w for d in docs for w in d)
    for w in all_words:
        df = sum(1 for d in docs if w in d)
        idf[w] = math.log((n + 1) / (df + 1)) + 1
    return idf


def tfidf_vector(tf: dict, idf: dict) -> dict:
    return {w: tf.get(w, 0) * idf.get(w, 0) for w in idf}


def cosine_similarity(v1: dict, v2: dict) -> float:
    keys = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ── Build index once at module load ──────────────────────────
_doc_tokens = [tokenize(d["content"]) for d in ALL_DOCUMENTS]
_doc_tfs    = [build_tf(t) for t in _doc_tokens]
_idf        = build_idf(_doc_tokens)
_doc_vecs   = [tfidf_vector(tf, _idf) for tf in _doc_tfs]


def retrieve_relevant_docs(query: str, n_results: int = 3) -> list:
    """Find the most relevant documents for a query using TF-IDF cosine similarity."""
    q_tokens = tokenize(query)
    q_tf     = build_tf(q_tokens)
    q_vec    = tfidf_vector(q_tf, _idf)

    scores = [cosine_similarity(q_vec, dv) for dv in _doc_vecs]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return [
        {
            "title"  : ALL_DOCUMENTS[i]["title"],
            "content": ALL_DOCUMENTS[i]["content"].strip()
        }
        for i in top_indices
    ]
