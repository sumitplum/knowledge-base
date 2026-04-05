"""
Code chunker for vector embedding.
Splits code into chunks suitable for embedding while preserving context.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import tiktoken

from .extractors.base import ExtractedNode, NodeType

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """A chunk of code ready for embedding."""
    chunk_id: str
    content: str
    token_count: int
    node_type: str
    name: str
    file_path: str
    repo: str
    language: str
    start_line: int
    end_line: int
    parent_chunk_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Chunker:
    """
    Chunks code for vector embedding.
    
    Strategy:
    - Each extracted node (function/class/component) becomes a chunk
    - Chunk content = signature + docstring + body
    - Large chunks are split into sub-chunks with parent reference
    """
    
    def __init__(self, max_tokens: int = 1500, model: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding(model)
    
    def chunk_nodes(
        self,
        nodes: list[ExtractedNode],
        repo: str,
        language: str,
    ) -> list[CodeChunk]:
        """
        Convert extracted nodes into chunks for embedding.
        
        Args:
            nodes: List of extracted code nodes
            repo: Repository name
            language: Programming language
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        
        for node in nodes:
            # Build chunk content
            content_parts = []
            
            # Add signature
            if node.signature:
                content_parts.append(f"// Signature: {node.signature}")
            
            # Add docstring if present
            if node.docstring:
                content_parts.append(node.docstring)
            
            # Add body
            if node.body:
                content_parts.append(node.body)
            
            content = "\n".join(content_parts)
            content = self._fallback_if_empty(node, content)
            token_count = len(self.encoder.encode(content))
            
            if token_count <= self.max_tokens:
                # Single chunk
                chunks.append(self._create_chunk(
                    node=node,
                    content=content,
                    token_count=token_count,
                    repo=repo,
                    language=language,
                    chunk_index=0,
                ))
            else:
                # Split into sub-chunks
                sub_chunks = self._split_chunk(
                    node=node,
                    content=content,
                    repo=repo,
                    language=language,
                )
                chunks.extend(sub_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(nodes)} nodes")
        return chunks

    def _fallback_if_empty(self, node: ExtractedNode, content: str) -> str:
        """OpenAI rejects empty embedding inputs; keep minimal searchable context."""
        if content.strip():
            return content
        return "\n".join([
            f"// {node.node_type.value}: {node.name}",
            f"// File: {node.file_path}",
            f"// Lines: {node.start_line}-{node.end_line}",
            "// (no signature, docstring, or body captured)",
        ])
    
    def _create_chunk(
        self,
        node: ExtractedNode,
        content: str,
        token_count: int,
        repo: str,
        language: str,
        chunk_index: int = 0,
        parent_id: Optional[str] = None,
    ) -> CodeChunk:
        """Create a CodeChunk from an extracted node."""
        chunk_id = f"{node.unique_id}:{chunk_index}"
        
        return CodeChunk(
            chunk_id=chunk_id,
            content=content,
            token_count=token_count,
            node_type=node.node_type.value,
            name=node.name,
            file_path=node.file_path,
            repo=repo,
            language=language,
            start_line=node.start_line,
            end_line=node.end_line,
            parent_chunk_id=parent_id,
            metadata={
                "exported": node.exported,
                "annotations": node.annotations,
                **node.metadata,
            }
        )
    
    def _split_chunk(
        self,
        node: ExtractedNode,
        content: str,
        repo: str,
        language: str,
    ) -> list[CodeChunk]:
        """Split a large chunk into smaller sub-chunks."""
        chunks = []
        lines = content.split("\n")
        
        current_lines = []
        current_tokens = 0
        chunk_index = 0
        parent_id = f"{node.unique_id}:parent"
        
        # Create parent chunk with summary
        summary = self._create_summary(node)
        chunks.append(CodeChunk(
            chunk_id=parent_id,
            content=summary,
            token_count=len(self.encoder.encode(summary)),
            node_type=node.node_type.value,
            name=node.name,
            file_path=node.file_path,
            repo=repo,
            language=language,
            start_line=node.start_line,
            end_line=node.end_line,
            metadata={
                "is_summary": True,
                "total_chunks": 0,  # Updated later
                **node.metadata,
            }
        ))
        
        for line in lines:
            line_tokens = len(self.encoder.encode(line + "\n"))
            
            if current_tokens + line_tokens > self.max_tokens and current_lines:
                # Create chunk from current lines
                chunk_content = self._fallback_if_empty(node, "\n".join(current_lines))
                chunks.append(self._create_chunk(
                    node=node,
                    content=chunk_content,
                    token_count=current_tokens,
                    repo=repo,
                    language=language,
                    chunk_index=chunk_index,
                    parent_id=parent_id,
                ))
                chunk_index += 1
                current_lines = []
                current_tokens = 0
            
            current_lines.append(line)
            current_tokens += line_tokens
        
        # Don't forget the last chunk
        if current_lines:
            chunk_content = self._fallback_if_empty(node, "\n".join(current_lines))
            chunks.append(self._create_chunk(
                node=node,
                content=chunk_content,
                token_count=current_tokens,
                repo=repo,
                language=language,
                chunk_index=chunk_index,
                parent_id=parent_id,
            ))
        
        # Update parent with total chunks count
        chunks[0].metadata["total_chunks"] = len(chunks) - 1
        
        return chunks
    
    def _create_summary(self, node: ExtractedNode) -> str:
        """Create a summary for a large node."""
        parts = [
            f"// {node.node_type.value}: {node.name}",
            f"// File: {node.file_path}",
            f"// Lines: {node.start_line}-{node.end_line}",
        ]
        
        if node.signature:
            parts.append(f"// Signature: {node.signature}")
        
        if node.docstring:
            # Truncate docstring if needed
            doc = node.docstring[:500] + "..." if len(node.docstring) > 500 else node.docstring
            parts.append(doc)
        
        if node.annotations:
            parts.append(f"// Annotations: {', '.join(node.annotations)}")
        
        return "\n".join(parts)
