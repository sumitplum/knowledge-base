"""
Base extractor interface and common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from enum import Enum


class NodeType(str, Enum):
    """Types of code nodes we extract."""
    FUNCTION = "Function"
    CLASS = "Class"
    COMPONENT = "Component"
    HOOK = "Hook"
    TYPE = "Type"
    INTERFACE = "Interface"
    ROUTE = "Route"
    API_ENDPOINT = "APIEndpoint"
    API_CALL = "APICall"
    IMPORT = "Import"
    EXPORT = "Export"


@dataclass
class ExtractedNode:
    """Represents an extracted code entity."""
    node_type: NodeType
    name: str
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    body: str = ""
    docstring: str = ""
    exported: bool = False
    annotations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def unique_id(self) -> str:
        """Generate unique identifier for this node."""
        return f"{self.file_path}:{self.node_type.value}:{self.name}:{self.start_line}"


@dataclass
class ExtractedRelationship:
    """Represents a relationship between code entities."""
    source_id: str
    target_id: str
    relationship_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass 
class ExtractedImport:
    """Represents an import statement."""
    source_file: str
    imported_name: str
    imported_from: str
    is_default: bool = False
    is_namespace: bool = False
    alias: Optional[str] = None


@dataclass
class ExtractedAPICall:
    """Represents an API call (fetch, apiClient, etc.)."""
    file_path: str
    line_number: int
    method: str  # GET, POST, PUT, DELETE, PATCH
    url_pattern: str
    function_context: Optional[str] = None  # Name of function containing the call


@dataclass
class ExtractionResult:
    """Complete extraction result for a file."""
    file_path: str
    language: str
    repo: str
    nodes: list[ExtractedNode] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    imports: list[ExtractedImport] = field(default_factory=list)
    api_calls: list[ExtractedAPICall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseExtractor(ABC):
    """Abstract base class for language-specific extractors."""
    
    def __init__(self, repo_name: str, repo_root: Path):
        self.repo_name = repo_name
        self.repo_root = repo_root
    
    @abstractmethod
    def extract(self, parsed_file) -> ExtractionResult:
        """
        Extract code entities from a parsed file.
        
        Args:
            parsed_file: ParsedFile object from parser engine
            
        Returns:
            ExtractionResult containing all extracted entities
        """
        pass
    
    def get_relative_path(self, file_path: Path) -> str:
        """Get path relative to repo root."""
        try:
            return str(file_path.relative_to(self.repo_root))
        except ValueError:
            return str(file_path)
