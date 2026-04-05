#!/usr/bin/env python3
"""
Validation script - Run queries to verify knowledge base integrity.

Usage:
    python scripts/validate.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import logging
from rich.console import Console
from rich.table import Table

from neo4j import GraphDatabase

from config import settings
from graph.queries import QueryHelper

console = Console()
logger = logging.getLogger(__name__)


# Validation queries
VALIDATION_QUERIES = [
    {
        "name": "Count nodes by label",
        "query": """
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
            RETURN label, value.count as count
            ORDER BY count DESC
        """,
        "expected": lambda results: sum(r["count"] for r in results) > 0,
    },
    {
        "name": "Count relationships by type",
        "query": """
            CALL db.relationshipTypes() YIELD relationshipType
            CALL apoc.cypher.run('MATCH ()-[r:`' + relationshipType + '`]->() RETURN count(r) as count', {}) YIELD value
            RETURN relationshipType as type, value.count as count
            ORDER BY count DESC
        """,
        "expected": lambda results: len(results) > 0,
    },
    {
        "name": "Find API endpoints",
        "query": """
            MATCH (e:APIEndpoint)
            RETURN e.method as method, e.path as path, e.repo as repo
            LIMIT 10
        """,
        "expected": lambda results: True,  # May be empty if no endpoints
    },
    {
        "name": "Find functions with callers",
        "query": """
            MATCH (caller)-[:CALLS]->(f:Function)
            RETURN f.name as function, count(caller) as caller_count
            ORDER BY caller_count DESC
            LIMIT 5
        """,
        "expected": lambda results: True,
    },
    {
        "name": "Cross-repo API links",
        "query": """
            MATCH (consumer)-[:CONSUMES]->(endpoint:APIEndpoint)
            RETURN consumer.name as consumer, endpoint.path as endpoint, consumer.repo as consumer_repo
            LIMIT 10
        """,
        "expected": lambda results: True,
    },
]


def run_validation(driver) -> bool:
    """Run validation queries and report results."""
    console.print("\n[bold]Running Validation Queries[/bold]\n")
    
    query_helper = QueryHelper(driver)
    all_passed = True
    
    for validation in VALIDATION_QUERIES:
        name = validation["name"]
        query = validation["query"]
        expected = validation["expected"]
        
        console.print(f"[blue]{name}[/blue]")
        
        try:
            with driver.session() as session:
                result = session.run(query)
                records = [dict(r) for r in result]
                
                # Check expectation
                passed = expected(records)
                
                if passed:
                    console.print(f"  [green]PASS[/green] - {len(records)} results")
                else:
                    console.print(f"  [red]FAIL[/red] - Expectation not met")
                    all_passed = False
                
                # Print sample results
                if records:
                    table = Table(show_header=True)
                    for key in records[0].keys():
                        table.add_column(str(key))
                    
                    for record in records[:5]:
                        table.add_row(*[str(v)[:50] for v in record.values()])
                    
                    console.print(table)
                
        except Exception as e:
            console.print(f"  [red]ERROR[/red] - {e}")
            all_passed = False
        
        console.print()
    
    return all_passed


def run_sample_searches(driver):
    """Run sample search queries."""
    console.print("\n[bold]Sample Searches[/bold]\n")
    
    searches = [
        ("Find BatchController", "MATCH (c:Class) WHERE c.name CONTAINS 'BatchController' RETURN c LIMIT 5"),
        ("Find React components", "MATCH (c:Component) WHERE c.exported = true RETURN c.name, c.file_path LIMIT 10"),
        ("Find hooks", "MATCH (h:Hook) RETURN h.name, h.file_path LIMIT 10"),
        ("Find services", "MATCH (s:Class) WHERE 'Service' IN s.annotations RETURN s.name, s.file_path LIMIT 10"),
    ]
    
    with driver.session() as session:
        for name, query in searches:
            console.print(f"[blue]{name}[/blue]")
            try:
                result = session.run(query)
                records = list(result)
                console.print(f"  Found {len(records)} results")
                for record in records[:3]:
                    console.print(f"    - {dict(record)}")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
            console.print()


@click.command()
@click.option("--searches", is_flag=True, help="Also run sample searches")
def main(searches: bool):
    """Validate knowledge base integrity."""
    console.print("[bold blue]Knowledge Base Validation[/bold blue]\n")
    
    # Connect to Neo4j
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        console.print("[green]Connected to Neo4j[/green]")
    except Exception as e:
        console.print(f"[red]Failed to connect to Neo4j: {e}[/red]")
        sys.exit(1)
    
    try:
        # Run validation
        passed = run_validation(driver)
        
        # Run sample searches if requested
        if searches:
            run_sample_searches(driver)
        
        if passed:
            console.print("[bold green]All validations passed![/bold green]")
        else:
            console.print("[bold yellow]Some validations failed[/bold yellow]")
            sys.exit(1)
            
    finally:
        driver.close()


if __name__ == "__main__":
    main()
