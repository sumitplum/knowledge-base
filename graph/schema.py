"""
Neo4j schema definition: constraints, indexes, and initialization.
"""

import logging
from neo4j import Driver

logger = logging.getLogger(__name__)


# Constraint definitions
CONSTRAINTS = [
    # Unique constraints
    ("repo_unique", "Repo", "CREATE CONSTRAINT repo_unique IF NOT EXISTS FOR (r:Repo) REQUIRE r.name IS UNIQUE"),
    ("file_unique", "File", "CREATE CONSTRAINT file_unique IF NOT EXISTS FOR (f:File) REQUIRE (f.repo, f.path) IS UNIQUE"),
    ("module_unique", "Module", "CREATE CONSTRAINT module_unique IF NOT EXISTS FOR (m:Module) REQUIRE (m.repo, m.path) IS UNIQUE"),
]

# Index definitions
INDEXES = [
    # Node label indexes for fast lookup
    ("idx_function_name", "CREATE INDEX idx_function_name IF NOT EXISTS FOR (f:Function) ON (f.name)"),
    ("idx_function_repo", "CREATE INDEX idx_function_repo IF NOT EXISTS FOR (f:Function) ON (f.repo)"),
    ("idx_class_name", "CREATE INDEX idx_class_name IF NOT EXISTS FOR (c:Class) ON (c.name)"),
    ("idx_class_repo", "CREATE INDEX idx_class_repo IF NOT EXISTS FOR (c:Class) ON (c.repo)"),
    ("idx_component_name", "CREATE INDEX idx_component_name IF NOT EXISTS FOR (c:Component) ON (c.name)"),
    ("idx_component_repo", "CREATE INDEX idx_component_repo IF NOT EXISTS FOR (c:Component) ON (c.repo)"),
    ("idx_hook_name", "CREATE INDEX idx_hook_name IF NOT EXISTS FOR (h:Hook) ON (h.name)"),
    ("idx_type_name", "CREATE INDEX idx_type_name IF NOT EXISTS FOR (t:Type) ON (t.name)"),
    ("idx_apiendpoint_path", "CREATE INDEX idx_apiendpoint_path IF NOT EXISTS FOR (a:APIEndpoint) ON (a.path)"),
    ("idx_apiendpoint_method", "CREATE INDEX idx_apiendpoint_method IF NOT EXISTS FOR (a:APIEndpoint) ON (a.method)"),
    ("idx_route_path", "CREATE INDEX idx_route_path IF NOT EXISTS FOR (r:Route) ON (r.path)"),
    ("idx_file_repo", "CREATE INDEX idx_file_repo IF NOT EXISTS FOR (f:File) ON (f.repo)"),
    
    # Full-text indexes for search
    ("idx_fulltext_code", "CREATE FULLTEXT INDEX idx_fulltext_code IF NOT EXISTS FOR (n:Function|Class|Component|Hook) ON EACH [n.name, n.signature]"),
]


class SchemaManager:
    """
    Manages Neo4j schema: constraints and indexes.
    """
    
    def __init__(self, driver: Driver):
        self.driver = driver
    
    def initialize_schema(self):
        """Create all constraints and indexes."""
        logger.info("Initializing Neo4j schema...")
        
        with self.driver.session() as session:
            # Create constraints
            for name, label, query in CONSTRAINTS:
                try:
                    session.run(query)
                    logger.debug(f"Created constraint: {name}")
                except Exception as e:
                    logger.warning(f"Constraint {name} may already exist: {e}")
            
            # Create indexes
            for name, query in INDEXES:
                try:
                    session.run(query)
                    logger.debug(f"Created index: {name}")
                except Exception as e:
                    logger.warning(f"Index {name} may already exist: {e}")
        
        logger.info("Schema initialization complete")
    
    def drop_all(self):
        """Drop all data (use with caution!)."""
        logger.warning("Dropping all nodes and relationships...")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.info("All data dropped")
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.driver.session() as session:
            # Count nodes by label
            result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
                RETURN label, value.count as count
            """)
            node_counts = {record["label"]: record["count"] for record in result}
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            
            return {
                "nodes": node_counts,
                "total_nodes": sum(node_counts.values()),
                "relationships": rel_count,
            }
