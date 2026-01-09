from typing import List, Dict, Tuple

def average_precision(ranked_docids: List[str], relevant_set: set) -> float:
    if not relevant_set:
        return 0.0
    hit_count = 0
    sum_prec = 0.0
    for i, docid in enumerate(ranked_docids, start=1):
        if docid in relevant_set:
            hit_count += 1
            sum_prec += hit_count / i
    return sum_prec / len(relevant_set)

def mean_average_precision(rankings: Dict[str, List[Tuple[str, float]]], qrels: Dict[str, Dict[str, int]], qids: List[str]) -> float:
    aps = []
    for qid in qids:
        rels = qrels.get(qid, {})
        relevant_set = {docid for docid, rel in rels.items() if rel > 0}
        ranked_docids = [docid for docid, _ in rankings.get(qid, [])]
        aps.append(average_precision(ranked_docids, relevant_set))
    return sum(aps) / len(aps) if aps else 0.0
