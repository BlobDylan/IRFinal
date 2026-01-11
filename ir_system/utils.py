from itertools import product
import numpy as np
from pyserini.search import LuceneSearcher
import re
from pathlib import Path
from typing import Dict, Tuple, List

def load_queries_tsv(path: Path) -> Dict[str, str]:
    """Loads queries from a simple TSV: topic_id \t query_text"""
    queries = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on first whitespace or tab (file is tab-separated in the provided version)
        parts = line.split("\t", 1)
        if len(parts) == 2:
            qid, q = parts[0].strip(), parts[1].strip()
        else:
            # fallback: split by whitespace
            qid, q = line.split(maxsplit=1)
        queries[qid] = q
    return queries

def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Loads qrels in standard format: qid 0 docid rel"""
    qrels: Dict[str, Dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        qid, _, docid, rel = line.split()
        rel_i = int(rel)
        qrels.setdefault(qid, {})[docid] = rel_i
    return qrels

def split_train_test(queries: Dict[str, str], train_n: int = 50) -> Tuple[List[str], List[str]]:
    """Train = first train_n topic IDs in ascending numeric order."""
    qids_sorted = sorted(queries.keys(), key=lambda x: int(x))
    train_qids = qids_sorted[:train_n]
    test_qids  = qids_sorted[train_n:]
    return train_qids, test_qids

def validate_run_file(run_path: Path, expected_k: int = 1000) -> None:
    by_qid = {}
    with run_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 6:
                raise ValueError(f"{run_path.name}: line {line_no} has {len(parts)} columns (expected 6)")
            qid, q0, _, rank_s, score_s, _ = parts
            if q0 != "Q0":
                raise ValueError(f"{run_path.name}: line {line_no} col2 must be Q0")
            rank = int(rank_s)
            score = float(score_s)
            by_qid.setdefault(qid, []).append((rank, score))
    for qid, lst in by_qid.items():
        # rank check
        ranks = [r for r, _ in lst]
        if ranks != list(range(1, len(ranks)+1)):
            raise ValueError(f"{run_path.name}: query {qid} ranks are not consecutive starting at 1")
        if len(lst) != expected_k:
            # For very rare cases (index issues) you might get <1000 hits; flag it so you can decide how to handle.
            print(f"{run_path.name}: query {qid} has {len(lst)} hits (expected {expected_k})")
        # score monotonicity
        scores = [s for _, s in lst]
        if any(scores[i] < scores[i+1] for i in range(len(scores)-1)):
            raise ValueError(f"{run_path.name}: query {qid} scores are not non-increasing")
    print(f"{run_path.name}: OK ({len(by_qid)} queries × {expected_k} docs)")

def write_trec_run(run_path: Path, run_tag: str, rankings: Dict[str, List[Tuple[str, float]]]) -> None:
    """Write rankings dict {qid: [(docid, score), ...]} to TREC 6-column format."""
    with run_path.open("w", encoding="utf-8") as f:
        for qid in sorted(rankings.keys(), key=lambda x: int(x)):
            hits = rankings[qid]
            for rank, (docid, score) in enumerate(hits, start=1):
                # topic Q0 docid rank score run_tag
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_tag}\n")

def read_trec_run(run_path: Path) -> Dict[str, List[Tuple[str, float]]]:
    """
    Read a TREC run file:
      qid Q0 docid rank score tag
    Returns: {qid: [(docid, score), ...]} ordered by rank ascending.
    """
    by_qid: Dict[str, List[Tuple[int, str, float]]] = {}
    with run_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 6:
                raise ValueError(f"{run_path.name}: line {line_no} has {len(parts)} columns (expected 6)")
            qid, q0, docid, rank_s, score_s, _tag = parts
            if q0 != "Q0":
                raise ValueError(f"{run_path.name}: line {line_no} col2 must be Q0 (got {q0})")
            rank = int(rank_s)
            score = float(score_s)
            by_qid.setdefault(qid, []).append((rank, docid, score))

    rankings: Dict[str, List[Tuple[str, float]]] = {}
    for qid, triples in by_qid.items():
        triples.sort(key=lambda x: x[0])
        rankings[qid] = [(docid, float(score)) for _, docid, score in triples]
    return rankings

def ensure_run_has_all_qids(rankings: Dict[str, List[Tuple[str, float]]], expected_qids: List[str]) -> None:
    missing = sorted(set(expected_qids) - set(rankings.keys()), key=lambda x: int(x))
    if missing:
        raise ValueError(f"Run missing {len(missing)} qids (e.g., {missing[:10]})")

def grid_from_dict(d: Dict[str, List]):
    """Turn {'a':[1,2],'b':[3]} into list of param dicts."""
    keys = list(d.keys())
    combos = []
    for vals in product(*[d[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

def safe_raw(searcher: LuceneSearcher, docid: str, cache: Dict[str, str]) -> str:
    """Retrieves raw document text with caching."""
    if docid in cache:
        return cache[docid]
    d = searcher.doc(docid)
    raw = d.raw() if d is not None else ""
    cache[docid] = raw
    return raw

def normalize_scores(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    """Min-max normalize scores into [0,1]."""
    if not pairs:
        return {}
    vals = np.array([s for _, s in pairs], dtype=np.float32)
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-9:
        return {docid: 0.0 for docid, _ in pairs}
    return {docid: (float(s) - mn) / (mx - mn) for docid, s in pairs}

def merge_rerank(
    base: List[Tuple[str, float]],
    reranked_scores: Dict[str, float],
    topn: int,
    lam: float = 0.5,
    keep_rest: bool = True,
) -> List[Tuple[str, float]]:
    """
    Merges normalized base scores with normalized reranker scores.
    Formula: (1-lam)*base + lam*rerank
    Ensures tail scores are shifted to maintain monotonic sorting.
    """
    def normalize_scores(hits):
        if not hits: return {}
        scores = [s for _, s in hits]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return {d: 1.0 for d, _ in hits}
        return {d: (s - min_s) / (max_s - min_s) for d, s in hits}

    head = base[:topn]
    base_norm = normalize_scores(head)
    
    rerank_pairs = [(d, reranked_scores.get(d, 0.0)) for d, _ in head]
    rerank_norm = normalize_scores(rerank_pairs)

    combined = []
    for d, _ in head:
        s = (1 - lam) * base_norm.get(d, 0.0) + lam * rerank_norm.get(d, 0.0)
        combined.append((d, float(s)))

    combined.sort(key=lambda x: x[1], reverse=True)
    
    if not keep_rest:
        return combined

    seen = set(d for d, _ in combined)
    tail = [(d, float(s)) for d, s in base if d not in seen]
    
    if combined and tail:
        min_head_score = combined[-1][1]
        max_tail_score = tail[0][1]
        
        if max_tail_score >= min_head_score:
            offset = max_tail_score - min_head_score + 1e-4
            tail = [(d, s - offset) for d, s in tail]

    return combined + tail