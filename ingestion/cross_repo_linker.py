"""
Cross-repo linker: matches FE API calls with BE endpoints.
Creates EXPOSES/CONSUMES relationships between Orbit and Trinity-v2.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from .extractors.base import ExtractedNode, ExtractedAPICall, ExtractedRelationship, NodeType

logger = logging.getLogger(__name__)


@dataclass
class EndpointMatch:
    """Represents a matched FE call -> BE endpoint pair."""
    fe_file: str
    fe_line: int
    fe_method: str
    fe_url: str
    be_file: str
    be_method: str
    be_path: str
    be_controller: str
    be_handler: str
    confidence: float = 1.0


@dataclass
class CrossRepoLinkResult:
    """Result of cross-repo linking."""
    matches: list[EndpointMatch] = field(default_factory=list)
    unmatched_fe_calls: list[ExtractedAPICall] = field(default_factory=list)
    unmatched_be_endpoints: list[ExtractedNode] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)


class CrossRepoLinker:
    """
    Links frontend API calls to backend endpoints.
    
    Algorithm:
    1. Normalize all paths (strip base URLs, handle path params)
    2. Build lookup tables for BE endpoints
    3. For each FE call, find matching BE endpoint
    4. Handle dynamic path segments ({id}, :id, ${variable})
    """
    
    def __init__(self):
        self.be_endpoints: dict[str, list[ExtractedNode]] = {}  # method:path -> endpoints
        self.fe_calls: list[ExtractedAPICall] = []
    
    def link(
        self,
        fe_api_calls: list[ExtractedAPICall],
        be_endpoints: list[ExtractedNode],
    ) -> CrossRepoLinkResult:
        """
        Link frontend API calls to backend endpoints.
        
        Args:
            fe_api_calls: List of API calls from frontend
            be_endpoints: List of API endpoint nodes from backend
            
        Returns:
            CrossRepoLinkResult with matches and unmatched items
        """
        result = CrossRepoLinkResult()
        
        # Build BE endpoint lookup
        be_lookup = self._build_be_lookup(be_endpoints)
        
        matched_be_keys = set()
        
        # Match each FE call
        for fe_call in fe_api_calls:
            match = self._find_match(fe_call, be_lookup)
            
            if match:
                result.matches.append(match)
                matched_be_keys.add(f"{match.be_method}:{match.be_path}")
                
                # Create relationship
                result.relationships.append(ExtractedRelationship(
                    source_id=f"{fe_call.file_path}:APICall:{fe_call.url_pattern}:{fe_call.line_number}",
                    target_id=f"{match.be_file}:APIEndpoint:{match.be_method} {match.be_path}:*",
                    relationship_type="CONSUMES",
                    metadata={
                        "fe_method": fe_call.method,
                        "fe_url": fe_call.url_pattern,
                        "confidence": match.confidence,
                    }
                ))
            else:
                result.unmatched_fe_calls.append(fe_call)
        
        # Find unmatched BE endpoints
        for endpoint in be_endpoints:
            method = endpoint.metadata.get("http_method", "GET")
            path = endpoint.metadata.get("path", "")
            key = f"{method}:{path}"
            
            if key not in matched_be_keys:
                result.unmatched_be_endpoints.append(endpoint)
        
        logger.info(
            f"Cross-repo linking: {len(result.matches)} matches, "
            f"{len(result.unmatched_fe_calls)} unmatched FE calls, "
            f"{len(result.unmatched_be_endpoints)} unmatched BE endpoints"
        )
        
        return result
    
    def _build_be_lookup(
        self, 
        endpoints: list[ExtractedNode]
    ) -> dict[str, list[ExtractedNode]]:
        """Build lookup table from BE endpoints."""
        lookup = {}
        
        for endpoint in endpoints:
            if endpoint.node_type != NodeType.API_ENDPOINT:
                continue
            
            method = endpoint.metadata.get("http_method", "GET")
            path = endpoint.metadata.get("path", "")
            
            # Normalize path
            normalized = self._normalize_path(path)
            key = f"{method}:{normalized}"
            
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(endpoint)
        
        return lookup
    
    def _find_match(
        self,
        fe_call: ExtractedAPICall,
        be_lookup: dict[str, list[ExtractedNode]]
    ) -> Optional[EndpointMatch]:
        """Find matching BE endpoint for a FE call."""
        fe_method = fe_call.method.upper()
        fe_path = self._normalize_fe_url(fe_call.url_pattern)
        
        # Try exact match first
        key = f"{fe_method}:{fe_path}"
        if key in be_lookup:
            endpoint = be_lookup[key][0]
            return self._create_match(fe_call, endpoint, confidence=1.0)
        
        # Try pattern matching for dynamic segments
        for be_key, endpoints in be_lookup.items():
            be_method, be_path = be_key.split(":", 1)
            
            if be_method != fe_method:
                continue
            
            confidence = self._paths_match(fe_path, be_path)
            if confidence > 0:
                return self._create_match(fe_call, endpoints[0], confidence)
        
        return None
    
    def _normalize_path(self, path: str) -> str:
        """Normalize a path for matching."""
        # Remove leading/trailing slashes
        path = path.strip("/")
        
        # Convert Spring path params {id} to :id
        path = re.sub(r'\{(\w+)\}', r':\1', path)
        
        # Lowercase
        path = path.lower()
        
        return f"/{path}" if path else "/"
    
    def _normalize_fe_url(self, url: str) -> str:
        """Normalize frontend URL for matching."""
        # Remove base URL if present
        url = re.sub(r'^https?://[^/]+', '', url)
        
        # Remove query params
        url = url.split("?")[0]
        
        # Handle template literals ${...}
        url = re.sub(r'\$\{[^}]+\}', ':param', url)
        
        # Convert :param to normalized form
        url = re.sub(r':(\w+)', r':\1', url)
        
        return self._normalize_path(url)
    
    def _paths_match(self, fe_path: str, be_path: str) -> float:
        """
        Check if paths match, handling dynamic segments.
        Returns confidence score 0-1.
        """
        fe_parts = fe_path.strip("/").split("/")
        be_parts = be_path.strip("/").split("/")
        
        if len(fe_parts) != len(be_parts):
            return 0.0
        
        matches = 0
        params = 0
        
        for fe_part, be_part in zip(fe_parts, be_parts):
            if fe_part == be_part:
                matches += 1
            elif be_part.startswith(":") or fe_part.startswith(":"):
                # Dynamic segment
                params += 1
            else:
                return 0.0  # Mismatch
        
        total = len(fe_parts)
        if total == 0:
            return 1.0
        
        # Higher confidence for more exact matches
        return (matches + params * 0.8) / total
    
    def _create_match(
        self,
        fe_call: ExtractedAPICall,
        be_endpoint: ExtractedNode,
        confidence: float
    ) -> EndpointMatch:
        """Create an EndpointMatch from FE call and BE endpoint."""
        return EndpointMatch(
            fe_file=fe_call.file_path,
            fe_line=fe_call.line_number,
            fe_method=fe_call.method,
            fe_url=fe_call.url_pattern,
            be_file=be_endpoint.file_path,
            be_method=be_endpoint.metadata.get("http_method", "GET"),
            be_path=be_endpoint.metadata.get("path", ""),
            be_controller=be_endpoint.metadata.get("controller", ""),
            be_handler=be_endpoint.metadata.get("handler_method", ""),
            confidence=confidence,
        )
