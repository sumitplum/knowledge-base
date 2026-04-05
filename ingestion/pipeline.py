"""
Ingestion pipeline orchestrator.
Coordinates parsing, extraction, linking, and loading.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import Progress, TaskID

from .parser import ParserEngine, ParsedFile
from .extractors import TypeScriptExtractor, JavaExtractor, ExtractionResult, NodeType
from .cross_repo_linker import CrossRepoLinker, CrossRepoLinkResult
from .chunker import Chunker, CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class RepoConfig:
    """Configuration for a repository to ingest."""
    name: str
    path: Path
    language: str  # "typescript" or "java"
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Result of ingesting a repository."""
    repo_name: str
    files_processed: int = 0
    nodes_extracted: int = 0
    relationships_extracted: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
    extraction_results: list[ExtractionResult] = field(default_factory=list)
    chunks: list[CodeChunk] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result of full pipeline run."""
    repos: dict[str, IngestionResult] = field(default_factory=dict)
    cross_repo_result: Optional[CrossRepoLinkResult] = None
    total_nodes: int = 0
    total_relationships: int = 0
    total_chunks: int = 0


class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline.
    
    Steps:
    1. Discover files in repositories
    2. Parse files with Tree-sitter
    3. Extract code entities
    4. Link cross-repo relationships
    5. Chunk for embeddings
    """
    
    def __init__(self, max_workers: int = 4):
        self.parser = ParserEngine()
        self.chunker = Chunker()
        self.linker = CrossRepoLinker()
        self.max_workers = max_workers
    
    def run(
        self,
        repos: list[RepoConfig],
        progress: Optional[Progress] = None,
    ) -> PipelineResult:
        """
        Run the full ingestion pipeline.
        
        Args:
            repos: List of repository configurations
            progress: Optional Rich progress bar
            
        Returns:
            PipelineResult with all extraction and linking results
        """
        result = PipelineResult()
        
        # Process each repository
        for repo_config in repos:
            logger.info(f"Processing repository: {repo_config.name}")
            repo_result = self._process_repo(repo_config, progress)
            result.repos[repo_config.name] = repo_result
            result.total_nodes += repo_result.nodes_extracted
            result.total_relationships += repo_result.relationships_extracted
            result.total_chunks += repo_result.chunks_created
        
        # Cross-repo linking
        if len(repos) >= 2:
            logger.info("Running cross-repo linking...")
            result.cross_repo_result = self._run_cross_repo_linking(result.repos)
            if result.cross_repo_result:
                result.total_relationships += len(result.cross_repo_result.relationships)
        
        logger.info(
            f"Pipeline complete: {result.total_nodes} nodes, "
            f"{result.total_relationships} relationships, "
            f"{result.total_chunks} chunks"
        )
        
        return result
    
    def _process_repo(
        self,
        config: RepoConfig,
        progress: Optional[Progress] = None,
    ) -> IngestionResult:
        """Process a single repository."""
        result = IngestionResult(repo_name=config.name)
        
        # Discover files
        files = self._discover_files(config)
        logger.info(f"Found {len(files)} files in {config.name}")
        
        # Create extractor
        if config.language == "typescript":
            extractor = TypeScriptExtractor(config.name, config.path)
        elif config.language == "java":
            extractor = JavaExtractor(config.name, config.path)
        else:
            raise ValueError(f"Unsupported language: {config.language}")
        
        # Process files
        task_id = None
        if progress:
            task_id = progress.add_task(
                f"Processing {config.name}",
                total=len(files)
            )
        
        all_nodes = []
        
        for file_path in files:
            try:
                # Parse file
                parsed = self.parser.parse_file(file_path)
                if not parsed:
                    continue
                
                # Extract entities
                extraction = extractor.extract(parsed)
                result.extraction_results.append(extraction)
                result.files_processed += 1
                result.nodes_extracted += len(extraction.nodes)
                result.relationships_extracted += len(extraction.relationships)
                
                all_nodes.extend(extraction.nodes)
                
                if extraction.errors:
                    result.errors.extend(extraction.errors)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                result.errors.append(f"{file_path}: {str(e)}")
            
            if progress and task_id is not None:
                progress.advance(task_id)
        
        # Create chunks
        chunks = self.chunker.chunk_nodes(
            nodes=all_nodes,
            repo=config.name,
            language=config.language,
        )
        result.chunks = chunks
        result.chunks_created = len(chunks)
        
        return result
    
    def _discover_files(self, config: RepoConfig) -> list[Path]:
        """Discover files to process in a repository."""
        files = []
        
        # Default patterns by language
        if not config.include_patterns:
            if config.language == "typescript":
                config.include_patterns = ["**/*.ts", "**/*.tsx"]
            elif config.language == "java":
                config.include_patterns = ["**/*.java"]
        
        # Default exclude patterns
        if not config.exclude_patterns:
            config.exclude_patterns = [
                "**/node_modules/**",
                "**/.git/**",
                "**/dist/**",
                "**/build/**",
                "**/target/**",
                "**/__tests__/**",
                "**/*.test.*",
                "**/*.spec.*",
            ]
        
        for pattern in config.include_patterns:
            for file_path in config.path.glob(pattern):
                # Check excludes
                should_exclude = False
                for exclude in config.exclude_patterns:
                    if file_path.match(exclude):
                        should_exclude = True
                        break
                
                if not should_exclude and file_path.is_file():
                    files.append(file_path)
        
        return sorted(files)
    
    def _run_cross_repo_linking(
        self,
        repos: dict[str, IngestionResult]
    ) -> Optional[CrossRepoLinkResult]:
        """Run cross-repo linking between FE and BE."""
        # Collect FE API calls and BE endpoints
        fe_calls = []
        be_endpoints = []
        
        for repo_name, repo_result in repos.items():
            for extraction in repo_result.extraction_results:
                fe_calls.extend(extraction.api_calls)
                
                for node in extraction.nodes:
                    if node.node_type == NodeType.API_ENDPOINT:
                        be_endpoints.append(node)
        
        if not fe_calls or not be_endpoints:
            logger.warning("No API calls or endpoints found for cross-repo linking")
            return None
        
        return self.linker.link(fe_calls, be_endpoints)
