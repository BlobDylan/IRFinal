from dataclasses import dataclass
from enum import Enum

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
class RocchioConfig(BM25Config):
    fb_docs: int = 10
    fb_terms: int = 10
    alpha: float = 1.0
    beta: float = 0.75
    gamma: int = 0

class PassageStrategy(str, Enum):
    MAX = 'max'
    AVG = 'avg'

@dataclass
class PassageConfig(RM3Config):
    candidate_k: int = 200
    window_size: int = 120
    stride: int = 60
    alpha: float = 0.5
    strategy: PassageStrategy = PassageStrategy.MAX

class ProximityMode(str, Enum):
    PAIR = 'pair'
    SPAN = 'span'

@dataclass
class ProximityConfig(RunConfig):
    k1: float = 0.9
    b: float = 0.4
    mode: ProximityMode = ProximityMode.PAIR