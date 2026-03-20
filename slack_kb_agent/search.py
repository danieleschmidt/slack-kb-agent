import math
import re
from typing import List, Dict, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{1,}\b', text.lower())


def build_tfidf_index(docs: List[Dict]) -> Dict:
    """Build TF-IDF index from docs list. Returns index dict."""
    # compute term frequencies per doc
    tf = []
    all_terms = set()
    for doc in docs:
        tokens = tokenize(doc["content"])
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = max(len(tokens), 1)
        tf.append({t: c / total for t, c in freq.items()})
        all_terms.update(freq.keys())

    # compute IDF
    N = len(docs)
    idf = {}
    for term in all_terms:
        df = sum(1 for d in tf if term in d)
        idf[term] = math.log((N + 1) / (df + 1)) + 1

    # compute TF-IDF vectors
    tfidf = []
    for d in tf:
        vec = {t: v * idf.get(t, 1) for t, v in d.items()}
        tfidf.append(vec)

    return {"docs": docs, "tfidf": tfidf, "idf": idf}


def search(index: Dict, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
    """Search index for query, return top_k (score, doc) pairs."""
    tokens = tokenize(query)
    idf = index["idf"]
    query_vec = {}
    for t in tokens:
        query_vec[t] = query_vec.get(t, 0) + idf.get(t, 1)

    results = []
    for doc, vec in zip(index["docs"], index["tfidf"]):
        score = sum(query_vec.get(t, 0) * v for t, v in vec.items())
        results.append((score, doc))

    results.sort(key=lambda x: -x[0])
    return [(s, d) for s, d in results[:top_k] if s > 0]


def excerpt(content: str, query: str, max_chars: int = 400) -> str:
    """Extract best matching excerpt from content."""
    tokens = set(tokenize(query))
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    best = ""
    best_score = -1
    for p in paragraphs:
        score = sum(1 for t in tokenize(p) if t in tokens)
        if score > best_score:
            best_score = score
            best = p
    return (best[:max_chars] + "...") if len(best) > max_chars else best
