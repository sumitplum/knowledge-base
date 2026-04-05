"""
TypeScript/TSX extractor for Orbit codebase.
Extracts: components, hooks, functions, types, routes, API calls.
"""

import re
import logging
from pathlib import Path
from typing import Optional, Any

from tree_sitter import Node

from ..parser import ParsedFile, get_node_text, get_node_location, find_nodes_by_type
from .base import (
    BaseExtractor, 
    ExtractionResult, 
    ExtractedNode, 
    ExtractedRelationship,
    ExtractedImport,
    ExtractedAPICall,
    NodeType,
)

logger = logging.getLogger(__name__)


class TypeScriptExtractor(BaseExtractor):
    """
    Extracts code entities from TypeScript/TSX files.
    Handles: functions, React components, hooks, types, imports, API calls.
    """
    
    # Patterns for API call detection
    API_PATTERNS = [
        r'fetch\s*\(\s*[`"\']([^`"\']+)[`"\']',
        r'apiClient\.(get|post|put|patch|delete)\s*\(\s*[`"\']([^`"\']+)[`"\']',
        r'authenticatedFetch\s*\(\s*[`"\']([^`"\']+)[`"\']',
    ]
    
    def extract(self, parsed_file: ParsedFile) -> ExtractionResult:
        """Extract all entities from a TypeScript/TSX file."""
        result = ExtractionResult(
            file_path=self.get_relative_path(parsed_file.path),
            language=parsed_file.language,
            repo=self.repo_name,
        )
        
        try:
            root = parsed_file.root_node
            source = parsed_file.source_code
            
            # Extract imports first (needed for relationship resolution)
            result.imports = self._extract_imports(root, source, result.file_path)
            
            # Extract functions and components
            self._extract_functions(root, source, result)
            
            # Extract types and interfaces
            self._extract_types(root, source, result)
            
            # Extract API calls
            result.api_calls = self._extract_api_calls(source, result.file_path)
            
            # Detect Next.js routes from file path
            route = self._detect_nextjs_route(parsed_file.path)
            if route:
                result.nodes.append(route)
            
            # Build relationships
            self._build_relationships(root, source, result)
            
        except Exception as e:
            logger.error(f"Extraction error in {parsed_file.path}: {e}")
            result.errors.append(str(e))
        
        return result
    
    def _extract_imports(
        self, 
        root: Node, 
        source: str, 
        file_path: str
    ) -> list[ExtractedImport]:
        """Extract import statements."""
        imports = []
        import_nodes = find_nodes_by_type(root, ["import_statement"])
        
        for node in import_nodes:
            import_clause = None
            source_node = None
            
            for child in node.children:
                if child.type == "import_clause":
                    import_clause = child
                elif child.type == "string":
                    source_node = child
            
            if not source_node:
                continue
                
            import_from = get_node_text(source_node, source).strip("'\"")
            
            if import_clause:
                for clause_child in import_clause.children:
                    if clause_child.type == "identifier":
                        # Default import
                        imports.append(ExtractedImport(
                            source_file=file_path,
                            imported_name=get_node_text(clause_child, source),
                            imported_from=import_from,
                            is_default=True,
                        ))
                    elif clause_child.type == "named_imports":
                        # Named imports
                        for spec in clause_child.children:
                            if spec.type == "import_specifier":
                                name = get_node_text(spec.child_by_field_name("name"), source) if spec.child_by_field_name("name") else get_node_text(spec, source)
                                alias_node = spec.child_by_field_name("alias")
                                imports.append(ExtractedImport(
                                    source_file=file_path,
                                    imported_name=name,
                                    imported_from=import_from,
                                    is_default=False,
                                    alias=get_node_text(alias_node, source) if alias_node else None,
                                ))
                    elif clause_child.type == "namespace_import":
                        # import * as X
                        for ns_child in clause_child.children:
                            if ns_child.type == "identifier":
                                imports.append(ExtractedImport(
                                    source_file=file_path,
                                    imported_name=get_node_text(ns_child, source),
                                    imported_from=import_from,
                                    is_namespace=True,
                                ))
        
        return imports
    
    def _extract_functions(
        self, 
        root: Node, 
        source: str, 
        result: ExtractionResult
    ):
        """Extract functions, React components, and hooks."""
        # Function declarations
        func_nodes = find_nodes_by_type(root, [
            "function_declaration",
            "arrow_function",
            "function",
        ])
        
        # Also get exported variable declarations with arrow functions
        export_nodes = find_nodes_by_type(root, ["export_statement"])
        
        processed_names = set()
        
        # Process export statements first
        for export_node in export_nodes:
            decl = export_node.child_by_field_name("declaration")
            if not decl:
                continue
                
            if decl.type == "lexical_declaration":
                for child in decl.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        value_node = child.child_by_field_name("value")
                        
                        if name_node and value_node and value_node.type == "arrow_function":
                            name = get_node_text(name_node, source)
                            if name in processed_names:
                                continue
                            processed_names.add(name)
                            
                            node = self._create_function_node(
                                name=name,
                                func_node=value_node,
                                source=source,
                                file_path=result.file_path,
                                exported=True,
                            )
                            if node:
                                result.nodes.append(node)
            
            elif decl.type == "function_declaration":
                name_node = decl.child_by_field_name("name")
                if name_node:
                    name = get_node_text(name_node, source)
                    if name in processed_names:
                        continue
                    processed_names.add(name)
                    
                    node = self._create_function_node(
                        name=name,
                        func_node=decl,
                        source=source,
                        file_path=result.file_path,
                        exported=True,
                    )
                    if node:
                        result.nodes.append(node)
        
        # Process function declarations
        for func_node in func_nodes:
            if func_node.type == "function_declaration":
                name_node = func_node.child_by_field_name("name")
                if name_node:
                    name = get_node_text(name_node, source)
                    if name in processed_names:
                        continue
                    processed_names.add(name)
                    
                    # Check if exported via separate export statement
                    exported = self._is_separately_exported(name, root, source)
                    
                    node = self._create_function_node(
                        name=name,
                        func_node=func_node,
                        source=source,
                        file_path=result.file_path,
                        exported=exported,
                    )
                    if node:
                        result.nodes.append(node)
    
    def _create_function_node(
        self,
        name: str,
        func_node: Node,
        source: str,
        file_path: str,
        exported: bool,
    ) -> Optional[ExtractedNode]:
        """Create an ExtractedNode from a function AST node."""
        loc = get_node_location(func_node)
        body = get_node_text(func_node, source)
        
        # Determine node type (Component, Hook, or Function)
        node_type = self._classify_function(name, body)
        
        # Extract signature (parameters)
        params_node = func_node.child_by_field_name("parameters")
        params = get_node_text(params_node, source) if params_node else "()"
        
        # Extract return type if present
        return_type_node = func_node.child_by_field_name("return_type")
        return_type = get_node_text(return_type_node, source) if return_type_node else ""
        
        signature = f"{name}{params}{return_type}"
        
        # Extract JSDoc comment if present
        docstring = self._extract_jsdoc(func_node, source)
        
        # Check for JSX return (React component indicator)
        has_jsx = self._contains_jsx(func_node)
        
        return ExtractedNode(
            node_type=node_type,
            name=name,
            file_path=file_path,
            start_line=loc["start_line"],
            end_line=loc["end_line"],
            signature=signature,
            body=body[:3000],  # Limit body size
            docstring=docstring,
            exported=exported,
            metadata={
                "has_jsx": has_jsx,
                "is_async": "async" in get_node_text(func_node, source)[:20],
            }
        )
    
    def _classify_function(self, name: str, body: str) -> NodeType:
        """Classify function as Component, Hook, or regular Function."""
        # Hooks start with "use"
        if name.startswith("use") and name[3:4].isupper():
            return NodeType.HOOK
        
        # Components: PascalCase and returns JSX
        if name[0].isupper() and ("jsx" in body.lower() or "<" in body):
            return NodeType.COMPONENT
        
        return NodeType.FUNCTION
    
    def _contains_jsx(self, node: Node) -> bool:
        """Check if node contains JSX elements."""
        jsx_types = ["jsx_element", "jsx_fragment", "jsx_self_closing_element"]
        return len(find_nodes_by_type(node, jsx_types)) > 0
    
    def _extract_jsdoc(self, node: Node, source: str) -> str:
        """Extract JSDoc comment before a node."""
        # Look for comment node before the function
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type == "comment":
            comment = get_node_text(prev_sibling, source)
            if comment.startswith("/**"):
                return comment
        return ""
    
    def _is_separately_exported(self, name: str, root: Node, source: str) -> bool:
        """Check if a name is exported via a separate export statement."""
        export_nodes = find_nodes_by_type(root, ["export_statement"])
        for node in export_nodes:
            text = get_node_text(node, source)
            if f"export {{ {name}" in text or f"export {{{name}" in text:
                return True
        return False
    
    def _extract_types(self, root: Node, source: str, result: ExtractionResult):
        """Extract type aliases and interfaces."""
        # Type aliases
        type_nodes = find_nodes_by_type(root, ["type_alias_declaration"])
        for node in type_nodes:
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            
            name = get_node_text(name_node, source)
            loc = get_node_location(node)
            
            result.nodes.append(ExtractedNode(
                node_type=NodeType.TYPE,
                name=name,
                file_path=result.file_path,
                start_line=loc["start_line"],
                end_line=loc["end_line"],
                signature=get_node_text(node, source)[:500],
                exported=self._check_node_exported(node, root, source),
            ))
        
        # Interfaces
        interface_nodes = find_nodes_by_type(root, ["interface_declaration"])
        for node in interface_nodes:
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            
            name = get_node_text(name_node, source)
            loc = get_node_location(node)
            
            result.nodes.append(ExtractedNode(
                node_type=NodeType.INTERFACE,
                name=name,
                file_path=result.file_path,
                start_line=loc["start_line"],
                end_line=loc["end_line"],
                signature=get_node_text(node, source)[:500],
                exported=self._check_node_exported(node, root, source),
            ))
    
    def _check_node_exported(self, node: Node, root: Node, source: str) -> bool:
        """Check if a node is exported (directly or via export statement)."""
        # Check parent for export_statement
        parent = node.parent
        if parent and parent.type == "export_statement":
            return True
        
        # Check for separate export
        name_node = node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node, source)
            return self._is_separately_exported(name, root, source)
        
        return False
    
    def _extract_api_calls(self, source: str, file_path: str) -> list[ExtractedAPICall]:
        """Extract API calls from source code."""
        calls = []
        lines = source.split("\n")
        
        for line_num, line in enumerate(lines, 1):
            # fetch calls
            fetch_match = re.search(r'fetch\s*\(\s*[`"\']([^`"\']+)[`"\']', line)
            if fetch_match:
                calls.append(ExtractedAPICall(
                    file_path=file_path,
                    line_number=line_num,
                    method="GET",  # Default, would need more context
                    url_pattern=fetch_match.group(1),
                ))
            
            # apiClient calls
            api_match = re.search(
                r'apiClient\.(get|post|put|patch|delete)\s*[<(]\s*[`"\']?([^`"\'<>()]+)[`"\']?',
                line, 
                re.IGNORECASE
            )
            if api_match:
                calls.append(ExtractedAPICall(
                    file_path=file_path,
                    line_number=line_num,
                    method=api_match.group(1).upper(),
                    url_pattern=api_match.group(2).strip(),
                ))
            
            # authenticatedFetch
            auth_match = re.search(
                r'authenticatedFetch\s*\(\s*[`"\']([^`"\']+)[`"\']',
                line
            )
            if auth_match:
                calls.append(ExtractedAPICall(
                    file_path=file_path,
                    line_number=line_num,
                    method="GET",
                    url_pattern=auth_match.group(1),
                ))
        
        return calls
    
    def _detect_nextjs_route(self, file_path: Path) -> Optional[ExtractedNode]:
        """Detect Next.js route from file path."""
        path_str = str(file_path)
        
        # Match Next.js app router patterns
        app_match = re.search(r'/app/(.+?)/(page|route)\.(tsx?|jsx?)$', path_str)
        if not app_match:
            return None
        
        route_path = app_match.group(1)
        route_type = app_match.group(2)
        
        # Convert file path segments to route pattern
        # [[...slug]] -> :slug*
        # [id] -> :id
        # (group) -> removed
        route = "/" + route_path
        route = re.sub(r'\[\[\.\.\.(\w+)\]\]', r':\1*', route)
        route = re.sub(r'\[\.\.\.(\w+)\]', r':\1+', route)
        route = re.sub(r'\[(\w+)\]', r':\1', route)
        route = re.sub(r'\([^)]+\)/', '', route)
        
        return ExtractedNode(
            node_type=NodeType.ROUTE,
            name=route,
            file_path=self.get_relative_path(file_path),
            start_line=1,
            end_line=1,
            metadata={
                "route_type": route_type,
                "framework": "nextjs",
            }
        )
    
    def _build_relationships(
        self,
        root: Node,
        source: str,
        result: ExtractionResult
    ):
        """Build relationships between extracted nodes."""
        # Build CALLS relationships from function calls
        call_nodes = find_nodes_by_type(root, ["call_expression"])
        
        for call in call_nodes:
            func_node = call.child_by_field_name("function")
            if not func_node:
                continue
            
            called_name = get_node_text(func_node, source)
            
            # Find the containing function
            parent = call.parent
            while parent:
                if parent.type in ["function_declaration", "arrow_function"]:
                    break
                parent = parent.parent
            
            if parent:
                # We'd need to resolve caller name - simplified here
                pass
        
        # Build USES_HOOK relationships for components
        for node in result.nodes:
            if node.node_type == NodeType.COMPONENT:
                # Find hook calls in the component body
                hook_pattern = r'\buse[A-Z]\w*\s*\('
                hooks = re.findall(hook_pattern, node.body)
                for hook in hooks:
                    hook_name = hook.rstrip('(').strip()
                    result.relationships.append(ExtractedRelationship(
                        source_id=node.unique_id,
                        target_id=f"*:Hook:{hook_name}:*",  # Wildcard for resolution later
                        relationship_type="USES_HOOK",
                    ))
