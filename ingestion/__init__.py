"""
Ingestion package initialization.
"""

from .parser import ParserEngine, ParsedFile
from .chunker import Chunker, CodeChunk
from .cross_repo_linker import CrossRepoLinker, CrossRepoLinkResult, EndpointMatch
from .pipeline import IngestionPipeline, RepoConfig, IngestionResult, PipelineResult

__all__ = [
    "ParserEngine",
    "ParsedFile",
    "Chunker",
    "CodeChunk",
    "CrossRepoLinker",
    "CrossRepoLinkResult",
    "EndpointMatch",
    "IngestionPipeline",
    "RepoConfig",
    "IngestionResult",
    "PipelineResult",
]
