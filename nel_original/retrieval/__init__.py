"""Candidate retrieval methods.

Each retrieval implementation is kept in its own module. The public names retain
compatibility with the original repository.
"""

from .base import BaseBiEncoder
from .experiments import FaissExperimentConfig, default_faiss_experiment_configs, profile_faiss_biencoder_configs
from .faiss import FaissBiEncoder
from .matrix import MatrixBiEncoder
from .sentence_transformer import DenseRetriever, SentenceTransformerBiEncoder
from .transformer_faiss import HerbertFaissBiEncoder
from .workflow import CandidateRetrievalPipeline, RetrievalResult, build_vocabulary

__all__ = [
    "BaseBiEncoder",
    "CandidateRetrievalPipeline",
    "DenseRetriever",
    "FaissBiEncoder",
    "FaissExperimentConfig",
    "HerbertFaissBiEncoder",
    "MatrixBiEncoder",
    "RetrievalResult",
    "SentenceTransformerBiEncoder",
    "build_vocabulary",
    "default_faiss_experiment_configs",
    "profile_faiss_biencoder_configs",
]
