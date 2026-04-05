"""
Neo4j batch loader for nodes and relationships.
Uses MERGE and UNWIND for efficient bulk loading.
"""

import json
import logging
from typing import Any
from neo4j import Driver

from ingestion.extractors.base import ExtractedNode, ExtractedRelationship, NodeType
from ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)


def _neo4j_safe_value(value: Any) -> Any:
    """
    Neo4j properties must be primitives or arrays of primitives (no nested maps).
    Lists of dicts (e.g. Java annotation details) are JSON-encoded as strings.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(x, (str, int, float, bool)) for x in value):
            return value
        return json.dumps(value, default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)


def _neo4j_safe_props(props: dict) -> dict:
    return {k: _neo4j_safe_value(v) for k, v in props.items()}


class GraphLoader:
    """
    Batch loads nodes and relationships into Neo4j.
    Uses MERGE for upserts and UNWIND for batch operations.
    """
    
    BATCH_SIZE = 500
    
    def __init__(self, driver: Driver):
        self.driver = driver
    
    def load_repo(self, name: str, path: str, language: str):
        """Create or update a Repo node."""
        with self.driver.session() as session:
            session.run("""
                MERGE (r:Repo {name: $name})
                SET r.path = $path, r.language = $language, r.updated_at = datetime()
            """, name=name, path=path, language=language)
    
    def load_files(self, files: list[dict]):
        """Batch load File nodes."""
        if not files:
            return
        
        with self.driver.session() as session:
            for batch_start in range(0, len(files), self.BATCH_SIZE):
                batch = files[batch_start:batch_start + self.BATCH_SIZE]
                session.run("""
                    UNWIND $files AS f
                    MERGE (file:File {repo: f.repo, path: f.path})
                    SET file.language = f.language, file.updated_at = datetime()
                    WITH file, f
                    MATCH (r:Repo {name: f.repo})
                    MERGE (r)-[:CONTAINS]->(file)
                """, files=batch)
        
        logger.info(f"Loaded {len(files)} file nodes")
    
    def load_nodes(self, nodes: list[ExtractedNode], repo: str):
        """Batch load extracted nodes."""
        if not nodes:
            return
        
        # Group nodes by type for efficient loading
        nodes_by_type: dict[NodeType, list[dict]] = {}
        
        for node in nodes:
            if node.node_type not in nodes_by_type:
                nodes_by_type[node.node_type] = []
            
            raw = {
                "name": node.name,
                "file_path": node.file_path,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "signature": node.signature,
                "docstring": node.docstring[:2000] if node.docstring else "",
                "exported": node.exported,
                "annotations": node.annotations,
                "repo": repo,
                **node.metadata,
            }
            nodes_by_type[node.node_type].append(_neo4j_safe_props(raw))
        
        with self.driver.session() as session:
            for node_type, type_nodes in nodes_by_type.items():
                label = node_type.value
                
                for batch_start in range(0, len(type_nodes), self.BATCH_SIZE):
                    batch = type_nodes[batch_start:batch_start + self.BATCH_SIZE]
                    
                    # Dynamic label query
                    session.run(f"""
                        UNWIND $nodes AS n
                        MERGE (node:{label} {{repo: n.repo, file_path: n.file_path, name: n.name, start_line: n.start_line}})
                        SET node += n, node.updated_at = datetime()
                        WITH node, n
                        MATCH (f:File {{repo: n.repo, path: n.file_path}})
                        MERGE (f)-[:CONTAINS]->(node)
                    """, nodes=batch)
                
                logger.debug(f"Loaded {len(type_nodes)} {label} nodes")
        
        logger.info(f"Loaded {len(nodes)} total nodes")
    
    def load_relationships(self, relationships: list[ExtractedRelationship]):
        """Batch load relationships."""
        if not relationships:
            return
        
        # Group by relationship type
        rels_by_type: dict[str, list[dict]] = {}
        
        for rel in relationships:
            if rel.relationship_type not in rels_by_type:
                rels_by_type[rel.relationship_type] = []
            
            rels_by_type[rel.relationship_type].append({
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "metadata": rel.metadata,
            })
        
        with self.driver.session() as session:
            for rel_type, rels in rels_by_type.items():
                # For relationships with wildcard IDs, we need fuzzy matching
                for batch_start in range(0, len(rels), self.BATCH_SIZE):
                    batch = rels[batch_start:batch_start + self.BATCH_SIZE]
                    
                    # Parse IDs and create relationships
                    for rel in batch:
                        source_parts = self._parse_node_id(rel["source_id"])
                        target_parts = self._parse_node_id(rel["target_id"])
                        
                        if source_parts and target_parts:
                            self._create_relationship(
                                session, 
                                source_parts, 
                                target_parts, 
                                rel_type,
                                rel["metadata"]
                            )
                
                logger.debug(f"Loaded {len(rels)} {rel_type} relationships")
        
        logger.info(f"Loaded {len(relationships)} total relationships")
    
    def _parse_node_id(self, node_id: str) -> dict:
        """Parse a node ID string into components."""
        # Format: file_path:NodeType:name:line
        parts = node_id.split(":")
        if len(parts) < 3:
            return None
        
        return {
            "file_path": parts[0] if parts[0] != "*" else None,
            "node_type": parts[1] if parts[1] != "*" else None,
            "name": parts[2] if parts[2] != "*" else None,
            "line": int(parts[3]) if len(parts) > 3 and parts[3] != "*" else None,
        }
    
    def _create_relationship(
        self,
        session,
        source: dict,
        target: dict,
        rel_type: str,
        metadata: dict,
    ):
        """Create a relationship between nodes."""
        # Build match conditions
        source_conditions = []
        target_conditions = []
        params = {}
        
        if source["file_path"]:
            source_conditions.append("s.file_path = $source_file")
            params["source_file"] = source["file_path"]
        if source["name"]:
            source_conditions.append("s.name = $source_name")
            params["source_name"] = source["name"]
        if source["line"]:
            source_conditions.append("s.start_line = $source_line")
            params["source_line"] = source["line"]
        
        if target["file_path"]:
            target_conditions.append("t.file_path = $target_file")
            params["target_file"] = target["file_path"]
        if target["name"]:
            target_conditions.append("t.name = $target_name")
            params["target_name"] = target["name"]
        if target["line"]:
            target_conditions.append("t.start_line = $target_line")
            params["target_line"] = target["line"]
        
        if not source_conditions or not target_conditions:
            return
        
        # Build query
        source_label = source["node_type"] or ""
        target_label = target["node_type"] or ""
        
        query = f"""
            MATCH (s{':' + source_label if source_label else ''})
            WHERE {' AND '.join(source_conditions)}
            MATCH (t{':' + target_label if target_label else ''})
            WHERE {' AND '.join(target_conditions)}
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += $metadata
        """
        
        params["metadata"] = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        
        try:
            session.run(query, **params)
        except Exception as e:
            logger.debug(f"Failed to create relationship: {e}")
    
    def load_cross_repo_links(self, matches: list[dict]):
        """Load cross-repo API linkages."""
        if not matches:
            return
        
        loaded = 0
        with self.driver.session() as session:
            for match in matches:
                result = session.run("""
                    MATCH (fe:Function {file_path: $fe_file})
                    WHERE fe.start_line <= $fe_line AND fe.end_line >= $fe_line
                    MATCH (be:APIEndpoint {path: $be_path, http_method: $be_method})
                    MERGE (fe)-[r:CONSUMES]->(be)
                    SET r.confidence = $confidence,
                        r.fe_url = $fe_url,
                        r.updated_at = datetime()
                    RETURN count(r) as cnt
                """, 
                    fe_file=match["fe_file"],
                    fe_line=match["fe_line"],
                    be_path=match["be_path"],
                    be_method=match["be_method"],
                    confidence=match.get("confidence", 1.0),
                    fe_url=match.get("fe_url", ""),
                )
                rec = result.single()
                if rec and rec["cnt"] > 0:
                    loaded += 1
        
        logger.info(f"Loaded {loaded}/{len(matches)} cross-repo links")
