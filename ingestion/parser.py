"""
Tree-sitter AST parsing engine.
Provides unified interface for parsing TypeScript/TSX and Java files.
"""

import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import tree_sitter_typescript as ts_typescript
import tree_sitter_java as ts_java
from tree_sitter import Language, Parser, Node, Tree

logger = logging.getLogger(__name__)


@dataclass
class ParsedFile:
    """Represents a parsed source file."""
    path: Path
    language: str
    tree: Tree
    source_code: str
    
    @property
    def root_node(self) -> Node:
        return self.tree.root_node


class ParserEngine:
    """
    Tree-sitter based parser engine supporting TypeScript/TSX and Java.
    """
    
    def __init__(self):
        self._parsers: dict[str, Parser] = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize language-specific parsers."""
        # TypeScript/TSX parser
        ts_parser = Parser(Language(ts_typescript.language_tsx()))
        self._parsers["typescript"] = ts_parser
        self._parsers["tsx"] = ts_parser
        self._parsers["ts"] = ts_parser
        
        # Java parser
        java_parser = Parser(Language(ts_java.language()))
        self._parsers["java"] = java_parser
    
    def get_language(self, file_path: Path) -> Optional[str]:
        """Determine language from file extension."""
        suffix = file_path.suffix.lower()
        mapping = {
            ".ts": "typescript",
            ".tsx": "tsx",
            ".js": "typescript",  # Parse JS with TS parser
            ".jsx": "tsx",
            ".java": "java",
        }
        return mapping.get(suffix)
    
    def parse_file(self, file_path: Path) -> Optional[ParsedFile]:
        """
        Parse a source file and return the AST.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            ParsedFile with AST tree, or None if parsing fails
        """
        language = self.get_language(file_path)
        if not language:
            logger.debug(f"Unsupported file type: {file_path}")
            return None
        
        parser = self._parsers.get(language)
        if not parser:
            logger.warning(f"No parser for language: {language}")
            return None
        
        try:
            source_code = file_path.read_text(encoding="utf-8")
            tree = parser.parse(source_code.encode("utf-8"))
            
            return ParsedFile(
                path=file_path,
                language=language,
                tree=tree,
                source_code=source_code,
            )
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def parse_string(self, source_code: str, language: str) -> Optional[Tree]:
        """
        Parse source code string directly.
        
        Args:
            source_code: The source code string
            language: Language identifier (typescript, tsx, java)
            
        Returns:
            Tree-sitter Tree object
        """
        parser = self._parsers.get(language)
        if not parser:
            return None
        
        return parser.parse(source_code.encode("utf-8"))


def walk_tree(node: Node, callback: callable, depth: int = 0):
    """
    Walk the AST tree and call callback for each node.
    
    Args:
        node: Current node
        callback: Function to call with (node, depth)
        depth: Current depth in tree
    """
    callback(node, depth)
    for child in node.children:
        walk_tree(child, callback, depth + 1)


def find_nodes_by_type(node: Node, node_types: list[str]) -> list[Node]:
    """
    Find all nodes of specific types in the subtree.
    
    Args:
        node: Root node to search from
        node_types: List of node type strings to match
        
    Returns:
        List of matching nodes
    """
    results = []
    
    def collect(n: Node, depth: int):
        if n.type in node_types:
            results.append(n)
    
    walk_tree(node, collect)
    return results


def get_node_text(node: Node, source_code: str) -> str:
    """Extract the text content of a node."""
    return source_code[node.start_byte:node.end_byte]


def get_node_location(node: Node) -> dict:
    """Get the location information for a node."""
    return {
        "start_line": node.start_point[0] + 1,  # 1-indexed
        "end_line": node.end_point[0] + 1,
        "start_column": node.start_point[1],
        "end_column": node.end_point[1],
    }
