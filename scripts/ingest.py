#!/usr/bin/env python3
"""
Ingestion CLI - Run the full knowledge base ingestion pipeline.

Usage:
    python scripts/ingest.py --all              # Ingest both repos
    python scripts/ingest.py --repo orbit       # Ingest only Orbit
    python scripts/ingest.py --repo trinity     # Ingest only Trinity-v2
    python scripts/ingest.py --validate         # Run validation after ingestion
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.table import Table

from neo4j import GraphDatabase

from config import settings, QDRANT_COLLECTIONS
from ingestion import IngestionPipeline, RepoConfig, PipelineResult
from graph import SchemaManager, GraphLoader
from vectors import Embedder, VectorStore

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def get_neo4j_driver():
    """Get Neo4j driver."""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def setup_repos(repos: list[str]) -> list[RepoConfig]:
    """Setup repository configurations."""
    configs = []
    
    if "orbit" in repos or "all" in repos:
        if settings.orbit_repo_path:
            configs.append(RepoConfig(
                name="orbit",
                path=Path(settings.orbit_repo_path),
                language="typescript",
                include_patterns=["**/*.ts", "**/*.tsx"],
                exclude_patterns=[
                    "**/node_modules/**",
                    "**/.next/**",
                    "**/dist/**",
                    "**/*.test.*",
                    "**/*.spec.*",
                    "**/__tests__/**",
                    "**/__mocks__/**",
                ],
            ))
        else:
            console.print("[yellow]Warning: ORBIT_REPO_PATH not set, skipping Orbit[/yellow]")
    
    if "trinity" in repos or "all" in repos:
        if settings.trinity_repo_path:
            configs.append(RepoConfig(
                name="trinity",
                path=Path(settings.trinity_repo_path),
                language="java",
                include_patterns=["**/*.java"],
                exclude_patterns=[
                    "**/target/**",
                    "**/build/**",
                    "**/*Test.java",
                    "**/*Tests.java",
                    "**/test/**",
                ],
            ))
        else:
            console.print("[yellow]Warning: TRINITY_REPO_PATH not set, skipping Trinity-v2[/yellow]")
    
    return configs


def run_ingestion(repos: list[RepoConfig]) -> PipelineResult:
    """Run the ingestion pipeline."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        pipeline = IngestionPipeline()
        return pipeline.run(repos, progress=progress)


def load_to_graph(result: PipelineResult, driver):
    """Load extraction results to Neo4j."""
    console.print("\n[bold]Loading to Neo4j...[/bold]")
    
    # Initialize schema
    schema_manager = SchemaManager(driver)
    schema_manager.initialize_schema()
    
    # Load data
    loader = GraphLoader(driver)
    
    for repo_name, repo_result in result.repos.items():
        console.print(f"  Loading {repo_name}...")
        
        # Load repo node
        config = next((r for r in setup_repos([repo_name]) if r.name == repo_name), None)
        if config:
            loader.load_repo(repo_name, str(config.path), config.language)
        
        file_list = [
            {"repo": repo_name, "path": f.file_path, "language": f.language}
            for f in repo_result.extraction_results
        ]
        loader.load_files(file_list)
        
        # Load nodes
        all_nodes = []
        for extraction in repo_result.extraction_results:
            all_nodes.extend(extraction.nodes)
        loader.load_nodes(all_nodes, repo_name)
        
        # Load relationships
        all_rels = []
        for extraction in repo_result.extraction_results:
            all_rels.extend(extraction.relationships)
        loader.load_relationships(all_rels)
    
    # Load cross-repo links
    if result.cross_repo_result:
        matches = [
            {
                "fe_file": m.fe_file,
                "fe_line": m.fe_line,
                "be_path": m.be_path,
                "be_method": m.be_method,
                "confidence": m.confidence,
                "fe_url": m.fe_url,
            }
            for m in result.cross_repo_result.matches
        ]
        loader.load_cross_repo_links(matches)
    
    console.print("[green]Graph loading complete![/green]")


def load_to_vectors(result: PipelineResult):
    """Load chunks to vector store."""
    console.print("\n[bold]Loading to Qdrant...[/bold]")
    
    embedder = Embedder()
    store = VectorStore()
    
    for repo_name, repo_result in result.repos.items():
        if not repo_result.chunks:
            continue
        
        collection_name = QDRANT_COLLECTIONS.get(repo_name, f"{repo_name}_code")
        
        console.print(f"  Creating collection {collection_name}...")
        store.create_collection(collection_name, recreate=False)
        
        console.print(f"  Embedding {len(repo_result.chunks)} chunks...")
        texts = [chunk.content for chunk in repo_result.chunks]
        embeddings = embedder.embed_batch(texts)
        
        console.print(f"  Storing vectors...")
        store.upsert_chunks(collection_name, repo_result.chunks, embeddings)
    
    embedder.close()
    console.print("[green]Vector loading complete![/green]")


def print_summary(result: PipelineResult, driver):
    """Print ingestion summary."""
    console.print("\n[bold]Ingestion Summary[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Repository")
    table.add_column("Files", justify="right")
    table.add_column("Nodes", justify="right")
    table.add_column("Relationships", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Errors", justify="right")
    
    for repo_name, repo_result in result.repos.items():
        table.add_row(
            repo_name,
            str(repo_result.files_processed),
            str(repo_result.nodes_extracted),
            str(repo_result.relationships_extracted),
            str(repo_result.chunks_created),
            str(len(repo_result.errors)),
        )
    
    console.print(table)
    
    if result.cross_repo_result:
        console.print(f"\n[bold]Cross-Repo Links:[/bold] {len(result.cross_repo_result.matches)} matches")
        console.print(f"  Unmatched FE calls: {len(result.cross_repo_result.unmatched_fe_calls)}")
        console.print(f"  Unmatched BE endpoints: {len(result.cross_repo_result.unmatched_be_endpoints)}")
    
    # Get graph stats
    try:
        schema_manager = SchemaManager(driver)
        stats = schema_manager.get_stats()
        console.print(f"\n[bold]Graph Stats:[/bold]")
        console.print(f"  Total nodes: {stats['total_nodes']}")
        console.print(f"  Total relationships: {stats['relationships']}")
    except Exception as e:
        logger.debug(f"Could not get graph stats: {e}")


@click.command()
@click.option("--all", "ingest_all", is_flag=True, help="Ingest all repositories")
@click.option("--repo", multiple=True, help="Specific repository to ingest (orbit, trinity)")
@click.option("--validate", is_flag=True, help="Run validation after ingestion")
@click.option("--skip-vectors", is_flag=True, help="Skip vector embedding step")
@click.option("--clean", is_flag=True, help="Clean existing data before ingestion")
def main(ingest_all: bool, repo: tuple, validate: bool, skip_vectors: bool, clean: bool):
    """Knowledge Base Ingestion CLI."""
    console.print("[bold blue]Knowledge Base Ingestion Pipeline[/bold blue]\n")
    
    # Determine repos to process
    if ingest_all:
        repos_to_process = ["all"]
    elif repo:
        repos_to_process = list(repo)
    else:
        console.print("[red]Error: Specify --all or --repo[/red]")
        console.print("Usage: python scripts/ingest.py --all")
        console.print("       python scripts/ingest.py --repo orbit --repo trinity")
        sys.exit(1)
    
    # Setup
    repo_configs = setup_repos(repos_to_process)
    if not repo_configs:
        console.print("[red]Error: No valid repositories configured[/red]")
        sys.exit(1)

    for rc in repo_configs:
        if not rc.path.is_dir():
            console.print(
                f"[red]Error: Repository path is missing or not a directory:[/red] {rc.path}"
            )
            console.print(
                "Set ORBIT_REPO_PATH / TRINITY_REPO_PATH in .env. "
                "Use paths relative to the knowledge-base folder (e.g. [cyan]../orbit[/cyan]), "
                "not [cyan]./orbit[/cyan] unless the repo is inside knowledge-base."
            )
            sys.exit(1)
    
    console.print(f"Repositories to process: {[r.name for r in repo_configs]}")
    
    # Get Neo4j driver
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        console.print("[green]Connected to Neo4j[/green]")
    except Exception as e:
        console.print(f"[red]Failed to connect to Neo4j: {e}[/red]")
        console.print("Make sure Neo4j is running: docker-compose up -d")
        sys.exit(1)
    
    try:
        # Clean if requested
        if clean:
            console.print("\n[yellow]Cleaning existing data...[/yellow]")
            schema_manager = SchemaManager(driver)
            schema_manager.drop_all()
        
        # Run ingestion
        console.print("\n[bold]Starting ingestion...[/bold]")
        result = run_ingestion(repo_configs)
        
        # Load to graph
        load_to_graph(result, driver)
        
        # Load to vectors
        if not skip_vectors:
            try:
                load_to_vectors(result)
            except Exception as e:
                console.print(f"[yellow]Warning: Vector loading failed: {e}[/yellow]")
                console.print("Make sure Qdrant is running: docker-compose up -d")
        
        # Print summary
        print_summary(result, driver)
        
        # Validate if requested
        if validate:
            console.print("\n[bold]Running validation...[/bold]")
            from scripts.validate import run_validation
            run_validation(driver)
        
        console.print("\n[bold green]Ingestion complete![/bold green]")
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()
