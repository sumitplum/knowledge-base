"""
Qdrant vector store management.
Handles collection setup, document storage, and retrieval.
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config import settings, QDRANT_COLLECTIONS
from ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """A document stored in the vector store."""
    id: str
    content: str
    embedding: list[float]
    metadata: dict


class VectorStore:
    """
    Manages Qdrant collections for code embeddings.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
    ):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.dimensions = settings.embedding_dimensions
        
        # Minor client/server version skew is common with Docker; API remains compatible.
        self.client = QdrantClient(
            host=self.host, port=self.port, check_compatibility=False
        )
    
    def create_collection(self, name: str, recreate: bool = False):
        """
        Create a collection for storing embeddings.
        
        Args:
            name: Collection name
            recreate: If True, delete existing collection first
        """
        if recreate:
            try:
                self.client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
            except Exception:
                pass
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(c.name == name for c in collections):
            logger.info(f"Collection already exists: {name}")
            return
        
        # Create collection
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self.dimensions,
                distance=Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
        
        # Create payload indexes for filtering
        self._create_payload_indexes(name)
        
        logger.info(f"Created collection: {name}")
    
    def _create_payload_indexes(self, collection_name: str):
        """Create indexes on payload fields for efficient filtering."""
        indexed_fields = [
            ("repo", models.PayloadSchemaType.KEYWORD),
            ("file_path", models.PayloadSchemaType.KEYWORD),
            ("node_type", models.PayloadSchemaType.KEYWORD),
            ("name", models.PayloadSchemaType.KEYWORD),
            ("language", models.PayloadSchemaType.KEYWORD),
        ]
        
        for field_name, field_type in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.debug(f"Index may already exist for {field_name}: {e}")
    
    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[CodeChunk],
        embeddings: list[list[float]],
    ):
        """
        Upsert code chunks with embeddings into the collection.
        
        Args:
            collection_name: Target collection
            chunks: List of code chunks
            embeddings: Corresponding embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=hash(chunk.chunk_id) % (2**63),  # Qdrant needs int64 IDs
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content[:5000],  # Limit stored content
                    "node_type": chunk.node_type,
                    "name": chunk.name,
                    "file_path": chunk.file_path,
                    "repo": chunk.repo,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    **{k: v for k, v in chunk.metadata.items() 
                       if isinstance(v, (str, int, float, bool))}
                }
            )
            points.append(point)
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )
        
        logger.info(f"Upserted {len(points)} vectors to {collection_name}")
    
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 20,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search
            query_vector: Query embedding
            limit: Maximum results
            filters: Optional payload filters
            
        Returns:
            List of matching documents with scores
        """
        # Build filter conditions
        filter_conditions = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            
            if conditions:
                filter_conditions = models.Filter(must=conditions)
        
        # qdrant-client 1.7+ uses query_points; .search() was removed.
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=filter_conditions,
            limit=limit,
            with_payload=True,
        )
        points = getattr(response, "points", None) or []
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": dict(hit.payload) if hit.payload else {},
            }
            for hit in points
        ]
    
    def get_collection_stats(self, collection_name: str) -> dict:
        """Get statistics for a collection."""
        info = self.client.get_collection(collection_name)
        d = info.model_dump()
        return {
            "name": collection_name,
            "vectors_count": d.get("vectors_count") or d.get("points_count", 0),
            "points_count": d.get("points_count", 0),
            "indexed_vectors_count": d.get("indexed_vectors_count", 0),
            "status": str(d.get("status", "unknown")).split(".")[-1].lower() if d.get("status") else "unknown",
        }
    
    def delete_by_repo(self, collection_name: str, repo: str):
        """Delete all vectors for a specific repository."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo",
                            match=models.MatchValue(value=repo),
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted vectors for repo {repo} from {collection_name}")
