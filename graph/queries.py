"""
Neo4j Cypher query helpers.
Pre-built queries for common operations.
"""

import logging
from typing import Optional, Any
from neo4j import Driver

logger = logging.getLogger(__name__)


class QueryHelper:
    """
    Helper class for common Neo4j queries.
    """
    
    def __init__(self, driver: Driver):
        self.driver = driver
    
    def get_node_neighbors(
        self,
        node_name: str,
        repo: Optional[str] = None,
        depth: int = 1,
        limit: int = 50,
    ) -> list[dict]:
        """
        Get neighbors of a node using BFS traversal.
        
        Args:
            node_name: Name of the node to start from
            repo: Optional repository filter
            depth: Traversal depth (1-3)
            limit: Maximum number of results
            
        Returns:
            List of neighbor nodes with relationship info
        """
        depth = min(max(depth, 1), 3)  # Clamp to 1-3
        
        repo_filter = "AND n.repo = $repo" if repo else ""
        
        query = f"""
            MATCH (n)
            WHERE n.name = $name {repo_filter}
            CALL apoc.path.subgraphNodes(n, {{
                maxLevel: $depth,
                limit: $limit
            }})
            YIELD node
            WITH n, node
            OPTIONAL MATCH (n)-[r]-(node)
            RETURN node, type(r) as rel_type, 
                   CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END as direction
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                name=node_name,
                repo=repo,
                depth=depth,
                limit=limit,
            )
            
            neighbors = []
            for record in result:
                node = dict(record["node"])
                neighbors.append({
                    "node": node,
                    "relationship": record["rel_type"],
                    "direction": record["direction"],
                })
            
            return neighbors
    
    def find_callers(
        self,
        function_name: str,
        repo: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Find all functions that call the given function.
        
        Args:
            function_name: Name of the function to find callers for
            repo: Optional repository filter
            
        Returns:
            List of calling functions
        """
        repo_filter = "AND f.repo = $repo AND caller.repo = $repo" if repo else ""
        
        query = f"""
            MATCH (caller)-[:CALLS]->(f:Function)
            WHERE f.name = $name {repo_filter}
            RETURN caller, f
            LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, name=function_name, repo=repo, limit=limit)
            return [{"caller": dict(r["caller"]), "callee": dict(r["f"])} for r in result]
    
    def get_api_contracts(
        self,
        repo: Optional[str] = None,
    ) -> list[dict]:
        """
        Get all API endpoints with their relationships.
        
        Args:
            repo: Optional repository filter
            
        Returns:
            List of API endpoint info
        """
        repo_filter = "WHERE e.repo = $repo" if repo else ""
        
        query = f"""
            MATCH (e:APIEndpoint)
            {repo_filter}
            OPTIONAL MATCH (e)-[:EXPOSED_BY]->(handler:Function)
            OPTIONAL MATCH (consumer)-[:CONSUMES]->(e)
            RETURN e, handler, collect(DISTINCT consumer) as consumers
        """
        
        with self.driver.session() as session:
            result = session.run(query, repo=repo)
            
            endpoints = []
            for record in result:
                endpoint = dict(record["e"])
                endpoint["handler"] = dict(record["handler"]) if record["handler"] else None
                endpoint["consumers"] = [dict(c) for c in record["consumers"]]
                endpoints.append(endpoint)
            
            return endpoints
    
    def cross_repo_impact(
        self,
        endpoint_path: str,
        depth: int = 2,
    ) -> dict:
        """
        Trace impact across repositories for an endpoint.
        
        Args:
            endpoint_path: The API endpoint path (e.g., "/api/v1/batch")
            depth: How deep to trace
            
        Returns:
            Impact analysis with BE and FE affected nodes
        """
        query = """
            // Find the endpoint
            MATCH (endpoint:APIEndpoint)
            WHERE endpoint.path CONTAINS $path
            
            // Get backend impact (handler and its callers)
            OPTIONAL MATCH (endpoint)-[:EXPOSED_BY]->(handler:Function)
            OPTIONAL MATCH (handler)<-[:CALLS*1..2]-(be_caller)
            
            // Get frontend consumers
            OPTIONAL MATCH (fe_consumer)-[:CONSUMES]->(endpoint)
            OPTIONAL MATCH (fe_consumer)<-[:CALLS*1..2]-(fe_caller)
            
            RETURN endpoint,
                   handler,
                   collect(DISTINCT be_caller) as be_callers,
                   collect(DISTINCT fe_consumer) as fe_consumers,
                   collect(DISTINCT fe_caller) as fe_callers
        """
        
        with self.driver.session() as session:
            result = session.run(query, path=endpoint_path)
            record = result.single()
            
            if not record:
                return {"endpoint": None, "impact": []}
            
            return {
                "endpoint": dict(record["endpoint"]) if record["endpoint"] else None,
                "handler": dict(record["handler"]) if record["handler"] else None,
                "backend_impact": [dict(n) for n in record["be_callers"]],
                "frontend_consumers": [dict(n) for n in record["fe_consumers"]],
                "frontend_impact": [dict(n) for n in record["fe_callers"]],
            }
    
    def module_dependency_graph(
        self,
        module_path: str,
        repo: str,
    ) -> dict:
        """
        Get the dependency graph for a module.
        
        Args:
            module_path: Path to the module
            repo: Repository name
            
        Returns:
            Module with its files and dependencies
        """
        query = """
            MATCH (m:Module {path: $path, repo: $repo})
            OPTIONAL MATCH (m)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
            OPTIONAL MATCH (f)-[:CONTAINS]->(node)
            RETURN m, 
                   collect(DISTINCT f) as files,
                   collect(DISTINCT imported) as imports,
                   collect(DISTINCT node) as nodes
        """
        
        with self.driver.session() as session:
            result = session.run(query, path=module_path, repo=repo)
            record = result.single()
            
            if not record:
                return None
            
            return {
                "module": dict(record["m"]),
                "files": [dict(f) for f in record["files"]],
                "imports": [dict(i) for i in record["imports"]],
                "nodes": [dict(n) for n in record["nodes"]],
            }
    
    def find_path(
        self,
        start_name: str,
        end_name: str,
        max_depth: int = 5,
    ) -> list[dict]:
        """
        Find shortest path between two nodes.
        
        Args:
            start_name: Name of the starting node
            end_name: Name of the ending node
            max_depth: Maximum path length
            
        Returns:
            Path as list of nodes and relationships
        """
        query = """
            MATCH (start), (end)
            WHERE start.name = $start_name AND end.name = $end_name
            MATCH path = shortestPath((start)-[*..{max_depth}]-(end))
            RETURN nodes(path) as nodes, relationships(path) as rels
        """.replace("{max_depth}", str(max_depth))
        
        with self.driver.session() as session:
            result = session.run(query, start_name=start_name, end_name=end_name)
            record = result.single()
            
            if not record:
                return []
            
            path = []
            nodes = record["nodes"]
            rels = record["rels"]
            
            for i, node in enumerate(nodes):
                path.append({"type": "node", "data": dict(node)})
                if i < len(rels):
                    path.append({"type": "relationship", "data": {"type": rels[i].type}})
            
            return path
    
    def search_nodes(
        self,
        query: str,
        node_types: Optional[list[str]] = None,
        repo: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Full-text search across nodes.
        
        Args:
            query: Search query
            node_types: Optional list of node types to search
            repo: Optional repository filter
            limit: Maximum results
            
        Returns:
            List of matching nodes
        """
        # Use fulltext index if available
        cypher = """
            CALL db.index.fulltext.queryNodes('idx_fulltext_code', $query)
            YIELD node, score
            WHERE ($repo IS NULL OR node.repo = $repo)
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, query=query, repo=repo, limit=limit)
                return [{"node": dict(r["node"]), "score": r["score"]} for r in result]
            except Exception:
                # Fallback to simple CONTAINS search
                fallback = """
                    MATCH (n)
                    WHERE (n.name CONTAINS $query OR n.signature CONTAINS $query)
                    AND ($repo IS NULL OR n.repo = $repo)
                    RETURN n as node, 1.0 as score
                    LIMIT $limit
                """
                result = session.run(fallback, query=query, repo=repo, limit=limit)
                return [{"node": dict(r["node"]), "score": r["score"]} for r in result]
