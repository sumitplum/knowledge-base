"""
Hybrid search combining vector similarity with graph expansion.
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from neo4j import Driver

from .embedder import Embedder
from .store import VectorStore
from graph.queries import QueryHelper

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with context."""
    chunk_id: str
    name: str
    node_type: str
    file_path: str
    repo: str
    content: str
    score: float
    start_line: int
    end_line: int
    graph_context: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class HybridSearch:
    """
    Hybrid search combining vector similarity with graph context.
    
    Strategy:
    1. Vector search -> top-N candidates
    2. For each candidate, fetch Neo4j neighbors
    3. Combine and re-rank results
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        neo4j_driver: Driver,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_helper = QueryHelper(neo4j_driver)
    
    def search(
        self,
        query: str,
        repos: Optional[list[str]] = None,
        node_types: Optional[list[str]] = None,
        limit: int = 20,
        include_graph_context: bool = True,
        graph_depth: int = 1,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Natural language search query
            repos: Optional list of repositories to search
            node_types: Optional list of node types to filter
            limit: Maximum results per collection
            include_graph_context: Whether to fetch graph neighbors
            graph_depth: Depth for graph traversal
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_vector = self.embedder.embed_text(query)
        
        # Build filters
        filters = {}
        if node_types:
            filters["node_type"] = node_types
        
        # Determine which collections to search
        from config import QDRANT_COLLECTIONS
        
        collections_to_search = []
        if repos:
            for repo in repos:
                if repo in QDRANT_COLLECTIONS:
                    collections_to_search.append(QDRANT_COLLECTIONS[repo])
        else:
            collections_to_search = list(QDRANT_COLLECTIONS.values())
        
        # Search each collection
        all_results = []
        for collection in collections_to_search:
            try:
                hits = self.vector_store.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=limit,
                    filters=filters if filters else None,
                )
                
                for hit in hits:
                    payload = hit["payload"]
                    result = SearchResult(
                        chunk_id=payload.get("chunk_id", ""),
                        name=payload.get("name", ""),
                        node_type=payload.get("node_type", ""),
                        file_path=payload.get("file_path", ""),
                        repo=payload.get("repo", ""),
                        content=payload.get("content", ""),
                        score=hit["score"],
                        start_line=payload.get("start_line", 0),
                        end_line=payload.get("end_line", 0),
                        metadata={k: v for k, v in payload.items() 
                                 if k not in ["chunk_id", "name", "node_type", "file_path", 
                                             "repo", "content", "start_line", "end_line"]}
                    )
                    all_results.append(result)
                    
            except Exception as e:
                logger.warning(f"Failed to search collection {collection}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda r: r.score, reverse=True)
        
        # Limit total results
        all_results = all_results[:limit]
        
        # Enrich with graph context
        if include_graph_context:
            for result in all_results:
                try:
                    neighbors = self.query_helper.get_node_neighbors(
                        node_name=result.name,
                        repo=result.repo,
                        depth=graph_depth,
                        limit=10,
                    )
                    result.graph_context = neighbors
                except Exception as e:
                    logger.debug(f"Failed to get graph context for {result.name}: {e}")
        
        return all_results
    
    def search_similar(
        self,
        chunk_id: str,
        collection_name: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the source chunk
            collection_name: Collection containing the chunk
            limit: Maximum results
            
        Returns:
            List of similar chunks
        """
        # Get the source chunk's embedding
        from qdrant_client.http import models
        
        # Search by chunk_id in payload
        results = self.vector_store.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchValue(value=chunk_id),
                    )
                ]
            ),
            limit=1,
            with_vectors=True,
        )
        
        if not results[0]:
            return []
        
        source_point = results[0][0]
        source_vector = source_point.vector
        
        # Search for similar
        hits = self.vector_store.search(
            collection_name=collection_name,
            query_vector=source_vector,
            limit=limit + 1,  # +1 because source will be in results
        )
        
        # Filter out source and convert
        search_results = []
        for hit in hits:
            payload = hit["payload"]
            if payload.get("chunk_id") == chunk_id:
                continue
            
            result = SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                name=payload.get("name", ""),
                node_type=payload.get("node_type", ""),
                file_path=payload.get("file_path", ""),
                repo=payload.get("repo", ""),
                content=payload.get("content", ""),
                score=hit["score"],
                start_line=payload.get("start_line", 0),
                end_line=payload.get("end_line", 0),
            )
            search_results.append(result)
        
        return search_results[:limit]
