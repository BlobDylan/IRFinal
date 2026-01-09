from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RunConfig:
    """Base configuration for retrieval parameters."""
    topk: int = 1000

@dataclass
class BM25Config(RunConfig):
    k1: float = 0.9
    b: float = 0.4

@dataclass
class RM3Config(BM25Config):
    fb_docs: int = 10
    fb_terms: int = 10
    original_query_weight: float = 0.5

@dataclass
class PassageConfig(RM3Config):
    candidate_k: int = 200
    window_size: int = 120
    stride: int = 60
    alpha: float = 0.5

@dataclass
class LTRConfig(RM3Config):
    """Configuration for Learning to Rank (LightGBM)."""
    candidate_k: int = 600
    passage_win: int = 120
    passage_stride: int = 60
    prox_cap: int = 100

@dataclass
class NeuralConfig(RunConfig):
    """Configuration for Neural methods (CrossEncoder, MonoT5)."""
    # Base retrieval params (usually BM25+RM3)
    k1: float = 0.9
    b: float = 0.4
    fb_docs: int = 10
    fb_terms: int = 10
    orig_w: float = 0.5
    
    # Reranking params
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_topn: int = 400
    lam: float = 0.85
    
    # Specifics for sliding windows (PARADE)
    passage_win: int = 120
    passage_stride: int = 60
    top_passages: int = 2