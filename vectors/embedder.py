"""
Embedding generator using OpenAI.
Handles batch embedding with retry/backoff and local caching.
"""

import logging
import hashlib
import time
from pathlib import Path
from typing import Optional
import sqlite3
import json

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

# OpenAI embeddings API rejects empty strings; whitespace-only is allowed.
_MIN_EMBEDDING_INPUT = " "


def _normalize_embedding_input(text: str) -> str:
    if text is None:
        return _MIN_EMBEDDING_INPUT
    if not text.strip():
        return _MIN_EMBEDDING_INPUT
    return text


class Embedder:
    """
    Generates embeddings using OpenAI API.
    Includes caching to avoid re-embedding unchanged code.
    """
    
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        model: str = None,
        dimensions: int = None,
    ):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        
        # Setup cache
        self.cache_path = cache_path or Path("~/.cache/knowledge-base/embeddings.db").expanduser()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite cache."""
        self.conn = sqlite3.connect(str(self.cache_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                embedding TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def _content_hash(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cached(self, content_hash: str) -> Optional[list[float]]:
        """Get embedding from cache."""
        cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE content_hash = ? AND model = ?",
            (content_hash, self.model)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def _cache_embedding(self, content_hash: str, embedding: list[float]):
        """Cache an embedding."""
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (content_hash, embedding, model) VALUES (?, ?, ?)",
            (content_hash, json.dumps(embedding), self.model)
        )
        self.conn.commit()
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        text = _normalize_embedding_input(text)
        # Check cache
        content_hash = self._content_hash(text)
        cached = self._get_cached(content_hash)
        if cached:
            return cached
        
        # Generate embedding
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        
        embedding = response.data[0].embedding
        
        # Cache result
        self._cache_embedding(content_hash, embedding)
        
        return embedding
    
    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress
            
        Returns:
            List of embedding vectors
        """
        normalized = [_normalize_embedding_input(t) for t in texts]
        embeddings = [None] * len(normalized)
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        for i, text in enumerate(normalized):
            content_hash = self._content_hash(text)
            cached = self._get_cached(content_hash)
            if cached:
                embeddings[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        if show_progress:
            logger.info(f"Found {len(texts) - len(texts_to_embed)} cached embeddings, generating {len(texts_to_embed)} new")
        
        # Process uncached texts in batches
        for batch_start in range(0, len(texts_to_embed), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[batch_start:batch_end]
            batch_indices = indices_to_embed[batch_start:batch_end]
            
            # Retry with exponential backoff
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                        dimensions=self.dimensions,
                    )
                    
                    for j, data in enumerate(response.data):
                        idx = batch_indices[j]
                        embedding = data.embedding
                        embeddings[idx] = embedding
                        
                        # Cache result
                        content_hash = self._content_hash(batch_texts[j])
                        self._cache_embedding(content_hash, embedding)
                    
                    break
                    
                except Exception as e:
                    if attempt < 2:
                        wait_time = 2 ** attempt
                        logger.warning(f"Embedding failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Embedding failed after 3 attempts: {e}")
                        raise
            
            if show_progress and batch_end % 500 == 0:
                logger.info(f"Embedded {batch_end}/{len(texts_to_embed)} texts")
        
        return embeddings
    
    def close(self):
        """Close cache connection."""
        self.conn.close()
