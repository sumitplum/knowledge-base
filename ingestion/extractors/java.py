"""
Java extractor for Trinity-v2 Spring Boot codebase.
Extracts: controllers, services, repositories, models, endpoints, dependencies.
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
    NodeType,
)

logger = logging.getLogger(__name__)


# Spring annotation patterns
HTTP_METHOD_ANNOTATIONS = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "PatchMapping": "PATCH",
    "DeleteMapping": "DELETE",
    "RequestMapping": None,  # Needs to check method attribute
}


class JavaExtractor(BaseExtractor):
    """
    Extracts code entities from Java files.
    Handles: classes, controllers, services, repositories, endpoints, dependencies.
    """
    
    def extract(self, parsed_file: ParsedFile) -> ExtractionResult:
        """Extract all entities from a Java file."""
        result = ExtractionResult(
            file_path=self.get_relative_path(parsed_file.path),
            language=parsed_file.language,
            repo=self.repo_name,
        )
        
        try:
            root = parsed_file.root_node
            source = parsed_file.source_code
            
            # Extract imports
            result.imports = self._extract_imports(root, source, result.file_path)
            
            # Extract classes
            self._extract_classes(root, source, result)
            
            # Build dependency relationships
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
        import_nodes = find_nodes_by_type(root, ["import_declaration"])
        
        for node in import_nodes:
            import_text = get_node_text(node, source)
            # Extract the imported class/package
            match = re.search(r'import\s+(?:static\s+)?([a-zA-Z0-9_.]+);', import_text)
            if match:
                full_import = match.group(1)
                parts = full_import.rsplit(".", 1)
                
                imports.append(ExtractedImport(
                    source_file=file_path,
                    imported_name=parts[-1] if len(parts) > 1 else full_import,
                    imported_from=parts[0] if len(parts) > 1 else "",
                    is_default=False,
                ))
        
        return imports
    
    def _extract_classes(
        self,
        root: Node,
        source: str,
        result: ExtractionResult
    ):
        """Extract class definitions and their members."""
        class_nodes = find_nodes_by_type(root, ["class_declaration"])
        
        for class_node in class_nodes:
            name_node = class_node.child_by_field_name("name")
            if not name_node:
                continue
            
            class_name = get_node_text(name_node, source)
            loc = get_node_location(class_node)
            
            # Extract annotations
            annotations = self._extract_annotations(class_node, source)
            annotation_names = [a["name"] for a in annotations]
            
            # Determine class type based on annotations
            class_type = self._classify_class(annotation_names)
            
            # Get base path for controllers
            base_path = ""
            if "RestController" in annotation_names or "Controller" in annotation_names:
                for ann in annotations:
                    if ann["name"] == "RequestMapping":
                        base_path = ann.get("value", "")
                        break
            
            # Extract Javadoc
            docstring = self._extract_javadoc(class_node, source)
            
            # Create class node
            class_extracted = ExtractedNode(
                node_type=NodeType.CLASS,
                name=class_name,
                file_path=result.file_path,
                start_line=loc["start_line"],
                end_line=loc["end_line"],
                signature=self._get_class_signature(class_node, source),
                docstring=docstring,
                annotations=annotation_names,
                exported=True,  # Java classes are always "exported"
                metadata={
                    "class_type": class_type,
                    "base_path": base_path,
                    "annotations_detail": annotations,
                }
            )
            result.nodes.append(class_extracted)
            
            # Extract methods
            self._extract_methods(class_node, source, result, class_name, base_path, class_type)
            
            # Extract constructor dependencies
            self._extract_constructor_dependencies(class_node, source, result, class_name)
    
    def _extract_annotations(self, node: Node, source: str) -> list[dict]:
        """Extract annotations from a node."""
        annotations = []
        
        # Look for modifiers node which contains annotations
        for child in node.children:
            if child.type == "modifiers":
                for mod_child in child.children:
                    if mod_child.type in ["annotation", "marker_annotation"]:
                        ann = self._parse_annotation(mod_child, source)
                        if ann:
                            annotations.append(ann)
        
        return annotations
    
    def _parse_annotation(self, node: Node, source: str) -> Optional[dict]:
        """Parse a single annotation node."""
        text = get_node_text(node, source)
        
        # Simple annotation like @Service
        simple_match = re.match(r'@(\w+)$', text)
        if simple_match:
            return {"name": simple_match.group(1)}
        
        # Annotation with value like @RequestMapping("/api/v1")
        value_match = re.match(r'@(\w+)\s*\(\s*["\']([^"\']+)["\']\s*\)', text)
        if value_match:
            return {"name": value_match.group(1), "value": value_match.group(2)}
        
        # Annotation with attributes like @RequestMapping(value = "/api", method = GET)
        attr_match = re.match(r'@(\w+)\s*\((.+)\)', text, re.DOTALL)
        if attr_match:
            name = attr_match.group(1)
            attrs_str = attr_match.group(2)
            attrs = {}
            
            # Extract value attribute
            val_match = re.search(r'value\s*=\s*["\']([^"\']+)["\']', attrs_str)
            if val_match:
                attrs["value"] = val_match.group(1)
            
            # Extract method attribute
            method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', attrs_str)
            if method_match:
                attrs["method"] = method_match.group(1)
            
            # Handle bare string value
            if not attrs:
                bare_match = re.search(r'["\']([^"\']+)["\']', attrs_str)
                if bare_match:
                    attrs["value"] = bare_match.group(1)
            
            return {"name": name, **attrs}
        
        # Fallback
        name_match = re.match(r'@(\w+)', text)
        if name_match:
            return {"name": name_match.group(1)}
        
        return None
    
    def _classify_class(self, annotations: list[str]) -> str:
        """Classify class based on annotations."""
        if "RestController" in annotations or "Controller" in annotations:
            return "controller"
        elif "Service" in annotations:
            return "service"
        elif "Repository" in annotations:
            return "repository"
        elif "Component" in annotations:
            return "component"
        elif "Configuration" in annotations:
            return "config"
        elif "Entity" in annotations:
            return "entity"
        return "class"
    
    def _get_class_signature(self, class_node: Node, source: str) -> str:
        """Get class signature (declaration line)."""
        text = get_node_text(class_node, source)
        # Get just the first line (declaration)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "{" in line:
                return " ".join(lines[:i+1]).strip()
        return lines[0] if lines else ""
    
    def _extract_javadoc(self, node: Node, source: str) -> str:
        """Extract Javadoc comment before a node."""
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type == "block_comment":
            comment = get_node_text(prev_sibling, source)
            if comment.startswith("/**"):
                return comment
        return ""
    
    def _extract_methods(
        self,
        class_node: Node,
        source: str,
        result: ExtractionResult,
        class_name: str,
        base_path: str,
        class_type: str,
    ):
        """Extract methods from a class."""
        body = class_node.child_by_field_name("body")
        if not body:
            return
        
        method_nodes = find_nodes_by_type(body, ["method_declaration"])
        
        for method_node in method_nodes:
            name_node = method_node.child_by_field_name("name")
            if not name_node:
                continue
            
            method_name = get_node_text(name_node, source)
            loc = get_node_location(method_node)
            
            # Extract annotations
            annotations = self._extract_annotations(method_node, source)
            annotation_names = [a["name"] for a in annotations]
            
            # Check if this is an endpoint method
            endpoint_info = self._extract_endpoint_info(annotations, base_path)
            
            # Extract return type
            return_type_node = method_node.child_by_field_name("type")
            return_type = get_node_text(return_type_node, source) if return_type_node else "void"
            
            # Extract parameters
            params = self._extract_parameters(method_node, source)
            
            # Build signature
            signature = f"{return_type} {method_name}({', '.join(params)})"
            
            # Create method node
            method_extracted = ExtractedNode(
                node_type=NodeType.FUNCTION,
                name=method_name,
                file_path=result.file_path,
                start_line=loc["start_line"],
                end_line=loc["end_line"],
                signature=signature,
                docstring=self._extract_javadoc(method_node, source),
                annotations=annotation_names,
                metadata={
                    "class_name": class_name,
                    "class_type": class_type,
                    "return_type": return_type,
                    "parameters": params,
                }
            )
            result.nodes.append(method_extracted)
            
            # Add HAS_METHOD relationship
            result.relationships.append(ExtractedRelationship(
                source_id=f"{result.file_path}:Class:{class_name}:{loc['start_line']}",
                target_id=method_extracted.unique_id,
                relationship_type="HAS_METHOD",
            ))
            
            # Create API endpoint node if applicable
            if endpoint_info:
                endpoint = ExtractedNode(
                    node_type=NodeType.API_ENDPOINT,
                    name=f"{endpoint_info['method']} {endpoint_info['path']}",
                    file_path=result.file_path,
                    start_line=loc["start_line"],
                    end_line=loc["end_line"],
                    signature=f"{endpoint_info['method']} {endpoint_info['path']}",
                    metadata={
                        "http_method": endpoint_info["method"],
                        "path": endpoint_info["path"],
                        "controller": class_name,
                        "handler_method": method_name,
                    }
                )
                result.nodes.append(endpoint)
                
                # Add EXPOSED_BY relationship
                result.relationships.append(ExtractedRelationship(
                    source_id=endpoint.unique_id,
                    target_id=method_extracted.unique_id,
                    relationship_type="EXPOSED_BY",
                ))
    
    def _extract_endpoint_info(
        self,
        annotations: list[dict],
        base_path: str
    ) -> Optional[dict]:
        """Extract HTTP endpoint info from method annotations."""
        for ann in annotations:
            name = ann.get("name", "")
            
            if name in HTTP_METHOD_ANNOTATIONS:
                method = HTTP_METHOD_ANNOTATIONS[name]
                path = ann.get("value", "")
                
                # For @RequestMapping, get method from attribute
                if name == "RequestMapping":
                    method = ann.get("method", "GET")
                
                # Combine base path and method path
                full_path = base_path.rstrip("/")
                if path:
                    full_path = f"{full_path}/{path.lstrip('/')}"
                
                if not full_path:
                    full_path = "/"
                
                return {
                    "method": method or "GET",
                    "path": full_path,
                }
        
        return None
    
    def _extract_parameters(self, method_node: Node, source: str) -> list[str]:
        """Extract method parameters as strings."""
        params = []
        params_node = method_node.child_by_field_name("parameters")
        
        if params_node:
            for child in params_node.children:
                if child.type == "formal_parameter":
                    params.append(get_node_text(child, source))
        
        return params
    
    def _extract_constructor_dependencies(
        self,
        class_node: Node,
        source: str,
        result: ExtractionResult,
        class_name: str
    ):
        """Extract constructor injection dependencies."""
        body = class_node.child_by_field_name("body")
        if not body:
            return
        
        # Find constructor
        constructor_nodes = find_nodes_by_type(body, ["constructor_declaration"])
        
        for constructor in constructor_nodes:
            params_node = constructor.child_by_field_name("parameters")
            if not params_node:
                continue
            
            for param in params_node.children:
                if param.type == "formal_parameter":
                    # Get the type
                    type_node = param.child_by_field_name("type")
                    if type_node:
                        dep_type = get_node_text(type_node, source)
                        
                        # Create DEPENDS_ON relationship
                        result.relationships.append(ExtractedRelationship(
                            source_id=f"{result.file_path}:Class:{class_name}:*",
                            target_id=f"*:Class:{dep_type}:*",
                            relationship_type="DEPENDS_ON",
                            metadata={"injection_type": "constructor"}
                        ))
    
    def _build_relationships(
        self,
        root: Node,
        source: str,
        result: ExtractionResult
    ):
        """Build additional relationships."""
        # Find method invocations for CALLS relationships
        call_nodes = find_nodes_by_type(root, ["method_invocation"])
        
        # This would need more sophisticated analysis to properly resolve
        # which method is being called - simplified for MVP
        pass
