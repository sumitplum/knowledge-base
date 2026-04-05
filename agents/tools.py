"""
LangChain tools for knowledge base operations.
These tools are used by the LangGraph agents.
"""

import logging
from typing import Optional, Any
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from neo4j import GraphDatabase

from config import settings
from graph.queries import QueryHelper
from vectors import HybridSearch, Embedder, VectorStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Feature Builder tool schemas
# ------------------------------------------------------------------

class EditFileInput(BaseModel):
    """Input for editing an existing file."""
    file_path: str = Field(description="Path to the file relative to the repo root")
    repo: str = Field(description="Repository name: orbit or trinity")
    instructions: str = Field(description="Clear instructions describing what to change in this file")
    description: str = Field(default="", description="Human-readable summary of the change (for PR body)")


class CreateFileInput(BaseModel):
    """Input for creating a new file."""
    file_path: str = Field(description="Path for the new file, relative to the repo root")
    repo: str = Field(description="Repository name: orbit or trinity")
    content: str = Field(description="Complete content for the new file")
    description: str = Field(default="", description="Human-readable summary of what this file is")


class DeleteFileInput(BaseModel):
    """Input for deleting a file."""
    file_path: str = Field(description="Path to the file relative to the repo root")
    repo: str = Field(description="Repository name: orbit or trinity")
    reason: str = Field(description="Why this file is being deleted")


# Tool input schemas
class SearchCodeInput(BaseModel):
    """Input for code search."""
    query: str = Field(description="Natural language search query")
    repo: Optional[str] = Field(default=None, description="Repository to search (orbit, trinity, or None for both)")
    node_type: Optional[str] = Field(default=None, description="Filter by node type (Function, Class, Component, Hook)")
    limit: int = Field(default=10, description="Maximum number of results")


class GetNodeGraphInput(BaseModel):
    """Input for graph traversal."""
    node_name: str = Field(description="Name of the node to get neighbors for")
    repo: Optional[str] = Field(default=None, description="Repository filter")
    depth: int = Field(default=1, description="Traversal depth (1-3)")


class FindCallersInput(BaseModel):
    """Input for finding callers."""
    function_name: str = Field(description="Name of the function")
    repo: Optional[str] = Field(default=None, description="Repository filter")


class CrossRepoTraceInput(BaseModel):
    """Input for cross-repo tracing."""
    endpoint_path: str = Field(description="API endpoint path (e.g., /api/v1/batch)")


class GetFileContentInput(BaseModel):
    """Input for getting file content."""
    file_path: str = Field(description="Relative path to the file")
    repo: str = Field(description="Repository name (orbit or trinity)")


class ModuleStructureInput(BaseModel):
    """Input for getting module structure."""
    module_path: str = Field(description="Path to the module/directory")
    repo: str = Field(description="Repository name")


# Global instances (initialized when tools are created)
_neo4j_driver = None
_query_helper = None
_hybrid_search = None


def _get_driver():
    """Get or create Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _neo4j_driver


def _get_query_helper():
    """Get or create query helper."""
    global _query_helper
    if _query_helper is None:
        _query_helper = QueryHelper(_get_driver())
    return _query_helper


def _get_hybrid_search():
    """Get or create hybrid search."""
    global _hybrid_search
    if _hybrid_search is None:
        embedder = Embedder()
        vector_store = VectorStore()
        _hybrid_search = HybridSearch(vector_store, embedder, _get_driver())
    return _hybrid_search


@tool(args_schema=SearchCodeInput)
def search_code(query: str, repo: Optional[str] = None, node_type: Optional[str] = None, limit: int = 10) -> str:
    """
    Search for code using natural language.
    Returns relevant functions, classes, components, or other code entities.
    Use this to find code related to a specific concept or functionality.
    """
    try:
        search = _get_hybrid_search()
        
        repos = [repo] if repo else None
        node_types = [node_type] if node_type else None
        
        results = search.search(
            query=query,
            repos=repos,
            node_types=node_types,
            limit=limit,
            include_graph_context=True,
        )
        
        if not results:
            return "No results found for the search query."
        
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"\n{i}. {result.node_type}: {result.name}")
            output.append(f"   File: {result.file_path}:{result.start_line}")
            output.append(f"   Repo: {result.repo}")
            output.append(f"   Score: {result.score:.3f}")
            
            # Include snippet
            snippet = result.content[:300].replace("\n", "\n   ")
            output.append(f"   Preview: {snippet}...")
            
            # Include graph context
            if result.graph_context:
                neighbors = [n["node"].get("name", "unknown") for n in result.graph_context[:3]]
                output.append(f"   Related: {', '.join(neighbors)}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {str(e)}"


@tool(args_schema=GetNodeGraphInput)
def get_node_graph(node_name: str, repo: Optional[str] = None, depth: int = 1) -> str:
    """
    Get the graph neighborhood of a code entity.
    Shows what the entity depends on and what depends on it.
    Use this to understand relationships and dependencies.
    """
    try:
        helper = _get_query_helper()
        neighbors = helper.get_node_neighbors(node_name, repo, depth)
        
        if not neighbors:
            return f"No graph data found for '{node_name}'"
        
        output = [f"Graph neighbors for '{node_name}':"]
        
        incoming = [n for n in neighbors if n["direction"] == "incoming"]
        outgoing = [n for n in neighbors if n["direction"] == "outgoing"]
        
        if incoming:
            output.append("\nIncoming (depends on this):")
            for n in incoming[:10]:
                node = n["node"]
                output.append(f"  - [{n['relationship']}] {node.get('name', 'unknown')} ({list(node.keys())[0] if node else 'unknown'})")
        
        if outgoing:
            output.append("\nOutgoing (this depends on):")
            for n in outgoing[:10]:
                node = n["node"]
                output.append(f"  - [{n['relationship']}] {node.get('name', 'unknown')}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        return f"Failed to get graph data: {str(e)}"


@tool(args_schema=FindCallersInput)
def find_callers(function_name: str, repo: Optional[str] = None) -> str:
    """
    Find all code that calls a specific function.
    Use this to understand the impact of changing a function.
    """
    try:
        helper = _get_query_helper()
        callers = helper.find_callers(function_name, repo)
        
        if not callers:
            return f"No callers found for '{function_name}'"
        
        output = [f"Callers of '{function_name}':"]
        for caller in callers:
            c = caller["caller"]
            output.append(f"  - {c.get('name', 'unknown')} in {c.get('file_path', 'unknown')}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Find callers failed: {e}")
        return f"Failed to find callers: {str(e)}"


@tool
def find_api_contracts(repo: Optional[str] = None) -> str:
    """
    Get all API endpoints with their handlers and consumers.
    Use this to understand the API surface of the codebase.
    """
    try:
        helper = _get_query_helper()
        contracts = helper.get_api_contracts(repo)
        
        if not contracts:
            return "No API contracts found."
        
        output = [f"API Contracts ({len(contracts)} endpoints):"]
        for contract in contracts[:20]:
            method = contract.get("http_method") or contract.get("method") or "?"
            path = contract.get("path", "unknown")
            output.append(f"\n  {method} {path}")
            
            if contract.get("handler"):
                handler = contract["handler"]
                output.append(f"    Handler: {handler.get('name', 'unknown')}")
            
            if contract.get("consumers"):
                consumers = [c.get("name", "unknown") for c in contract["consumers"]]
                output.append(f"    Consumers: {', '.join(consumers)}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Get API contracts failed: {e}")
        return f"Failed to get API contracts: {str(e)}"


@tool(args_schema=CrossRepoTraceInput)
def cross_repo_trace(endpoint_path: str) -> str:
    """
    Trace an API endpoint across frontend and backend.
    Shows which backend handler exposes the endpoint and which frontend code consumes it.
    Use this to understand cross-repository dependencies.
    """
    try:
        helper = _get_query_helper()
        impact = helper.cross_repo_impact(endpoint_path)
        
        if not impact.get("endpoint"):
            return f"No endpoint found matching '{endpoint_path}'"
        
        output = [f"Cross-repo trace for '{endpoint_path}':"]
        
        endpoint = impact["endpoint"]
        output.append(f"\nEndpoint: {endpoint.get('http_method') or endpoint.get('method') or '?'} {endpoint.get('path', 'unknown')}")
        
        if impact.get("handler"):
            handler = impact["handler"]
            output.append(f"\nBackend Handler:")
            output.append(f"  {handler.get('name', 'unknown')} in {handler.get('file_path', 'unknown')}")
        
        if impact.get("backend_impact"):
            output.append(f"\nBackend Impact ({len(impact['backend_impact'])} nodes):")
            for node in impact["backend_impact"][:5]:
                output.append(f"  - {node.get('name', 'unknown')}")
        
        if impact.get("frontend_consumers"):
            output.append(f"\nFrontend Consumers ({len(impact['frontend_consumers'])} nodes):")
            for node in impact["frontend_consumers"][:5]:
                output.append(f"  - {node.get('name', 'unknown')} in {node.get('file_path', 'unknown')}")
        
        if impact.get("frontend_impact"):
            output.append(f"\nFrontend Impact ({len(impact['frontend_impact'])} nodes):")
            for node in impact["frontend_impact"][:5]:
                output.append(f"  - {node.get('name', 'unknown')}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Cross-repo trace failed: {e}")
        return f"Failed to trace endpoint: {str(e)}"


def _resolve_file_path(file_path: str, repo: str) -> Optional[Path]:
    """
    Try multiple base paths to find where a file actually lives.

    The configured repo path may point to a subdirectory (e.g. orbit/apps/trinity)
    while Neo4j/Qdrant index paths relative to the git root (e.g. orbit/).
    Search results therefore return paths like 'apps/trinity/app/features/...' OR
    just 'app/features/...' depending on how they were indexed.  We try both the
    git root and the configured subpath so either form resolves correctly.
    """
    candidates: list[Path] = []

    # 1. Git root (what _get_repo_path returns)
    git_root = _get_repo_path(repo)
    if git_root:
        candidates.append(git_root)

    # 2. The raw configured path (the subdir, e.g. orbit/apps/trinity)
    if repo == "orbit":
        raw = settings.orbit_repo_path
    elif repo == "trinity":
        raw = settings.trinity_repo_path
    else:
        raw = None

    if raw:
        raw_path = Path(raw).resolve()
        if raw_path not in candidates:
            candidates.append(raw_path)

    clean = file_path.lstrip("/")
    for base in candidates:
        candidate = base / clean
        if candidate.exists():
            return candidate

    return None


@tool(args_schema=GetFileContentInput)
def get_file_content(file_path: str, repo: str) -> str:
    """
    Get the content of a source file.
    Use this when you need to see the full implementation of a function or class.
    """
    try:
        resolved = _resolve_file_path(file_path, repo)
        if resolved is None:
            return f"File not found: {file_path}"

        content = resolved.read_text(encoding="utf-8")

        # Limit content size
        if len(content) > 10000:
            content = content[:10000] + "\n\n... (truncated)"

        return f"File: {file_path}\n\n{content}"

    except Exception as e:
        logger.error(f"Get file content failed: {e}")
        return f"Failed to read file: {str(e)}"


@tool(args_schema=ModuleStructureInput)
def get_module_structure(module_path: str, repo: str) -> str:
    """
    Get the structure of a module/directory.
    Shows files and exported entities within the module.
    """
    try:
        helper = _get_query_helper()
        structure = helper.module_dependency_graph(module_path, repo)
        
        if not structure:
            return f"No module found at '{module_path}' in {repo}"
        
        output = [f"Module: {module_path} ({repo})"]
        
        if structure.get("files"):
            output.append(f"\nFiles ({len(structure['files'])}):")
            for f in structure["files"][:20]:
                output.append(f"  - {f.get('path', 'unknown')}")
        
        if structure.get("nodes"):
            output.append(f"\nEntities ({len(structure['nodes'])}):")
            for n in structure["nodes"][:20]:
                output.append(f"  - {n.get('name', 'unknown')} ({list(n.keys())[0] if n else 'unknown'})")
        
        if structure.get("imports"):
            output.append(f"\nExternal Dependencies ({len(structure['imports'])}):")
            for i in structure["imports"][:10]:
                output.append(f"  - {i.get('path', 'unknown')}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Get module structure failed: {e}")
        return f"Failed to get module structure: {str(e)}"


def _get_repo_path(repo: str) -> Optional[Path]:
    """
    Return the git repository root for the given repo name.
    The configured path may point to a subdirectory (e.g. apps/trinity);
    we walk up to find the .git root so file writes land in the right place.
    """
    if repo == "orbit":
        base = settings.orbit_repo_path
    elif repo == "trinity":
        base = settings.trinity_repo_path
    else:
        return None

    if not base:
        return None

    p = Path(base).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / ".git").exists():
            return candidate
    return p  # Fallback: use as-is


def _is_dry_run() -> bool:
    """Read dry_run mode set by the orchestrator node."""
    import os
    return os.environ.get("KB_DRY_RUN", "1") != "0"


@tool(args_schema=EditFileInput)
def edit_file(file_path: str, repo: str, instructions: str, description: str = "") -> str:
    """
    Edit an existing source file using AI code generation.
    The file is read, the LLM applies the instructions, and the result is
    recorded in the ChangeTracker.  In dry-run mode the file is not written
    to disk (only the diff is recorded).  In real mode the file is written.

    Use this tool to implement code changes in existing files.
    Always provide specific, actionable instructions.
    """
    try:
        from codegen.code_generator import CodeGenerator
        from codegen.change_tracker import get_tracker
        from agents.rate_limiter import get_rate_limiter

        # Check file write rate limit
        rate_limiter = get_rate_limiter()
        rate_limiter.check_file_write()

        git_root = _get_repo_path(repo)
        if not git_root:
            return f"Repository path not configured for '{repo}'"

        # Resolve which base the file actually lives under, then derive the
        # relative path from that base so CodeGenerator finds it correctly.
        resolved_abs = _resolve_file_path(file_path, repo)
        if resolved_abs is None:
            return f"File not found: {file_path} in repo '{repo}'"

        # Determine which base the resolved path is under
        actual_base = git_root
        clean_path = file_path.lstrip("/")
        for candidate_base in [git_root]:
            try:
                resolved_abs.relative_to(candidate_base)
                actual_base = candidate_base
                clean_path = str(resolved_abs.relative_to(candidate_base))
                break
            except ValueError:
                pass

        generator = CodeGenerator(
            repo=repo,
            repo_path=actual_base,
            tracker=get_tracker(),
            dry_run=_is_dry_run(),
        )

        results = generator.generate_changes(
            file_instructions=[{
                "file_path": clean_path,
                "instructions": instructions,
                "is_new_file": False,
                "description": description or instructions[:80],
            }]
        )

        result = results[0] if results else {}
        if result.get("success"):
            # Track the file write
            rate_limiter.increment_file_writes()
            
            added = result.get("lines_added", 0)
            removed = result.get("lines_removed", 0)
            return (
                f"Successfully generated edit for '{file_path}' in {repo}.\n"
                f"Changes: +{added} / -{removed} lines.\n"
                f"Edit has been recorded and will be committed in the next step."
            )
        else:
            return f"Failed to edit '{file_path}': {result.get('error', 'unknown error')}"

    except Exception as e:
        logger.error(f"edit_file failed: {e}")
        return f"edit_file tool error: {str(e)}"


@tool(args_schema=CreateFileInput)
def create_file(file_path: str, repo: str, content: str, description: str = "") -> str:
    """
    Create a new source file with the provided content.
    The file is recorded in the ChangeTracker.  In real mode it is also
    written to disk immediately so the git node can stage and commit it.

    Use this tool to add entirely new files (components, services, tests, etc.).
    """
    try:
        from codegen.change_tracker import get_tracker
        from agents.rate_limiter import get_rate_limiter

        # Check file write rate limit
        rate_limiter = get_rate_limiter()
        rate_limiter.check_file_write()

        base_path = _get_repo_path(repo)
        if not base_path:
            return f"Repository path not configured for '{repo}'"

        abs_path = Path(base_path) / file_path.lstrip("/")

        tracker = get_tracker()
        change = tracker.record_edit(
            repo=repo,
            file_path=file_path,
            abs_path=abs_path,
            original_content=None,
            new_content=content,
            description=description or f"Create {file_path}",
            is_new_file=True,
        )

        if not _is_dry_run():
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")

        # Track the file write
        rate_limiter.increment_file_writes()

        lines = len(content.splitlines())
        mode = "Recorded (dry-run)" if _is_dry_run() else "Written to disk"
        return (
            f"{mode}: new file '{file_path}' in {repo} ({lines} lines).\n"
            f"File will be committed in the next step."
        )

    except Exception as e:
        logger.error(f"create_file failed: {e}")
        return f"create_file tool error: {str(e)}"


class LintFileInput(BaseModel):
    """Input for linting a file."""
    file_path: str = Field(description="Path to the file relative to the repo root")
    repo: str = Field(description="Repository name: orbit or trinity")


class VerifyFileInput(BaseModel):
    """Input for verifying a file exists and re-reading it."""
    file_path: str = Field(description="Path to the file relative to the repo root")
    repo: str = Field(description="Repository name: orbit or trinity")


@tool(args_schema=LintFileInput)
def lint_file(file_path: str, repo: str) -> str:
    """
    Run linting on a file to check for syntax and style errors.
    
    Runs the appropriate linter based on file extension:
    - Python (.py): ruff check
    - TypeScript/JavaScript (.ts, .tsx, .js, .jsx): eslint
    - Java (.java): javac -Xlint (syntax check only)
    
    Returns lint errors or "PASS" if no issues found.
    """
    import subprocess
    
    try:
        base_path = _get_repo_path(repo)
        if not base_path:
            return f"Repository path not configured for '{repo}'"
        
        full_path = Path(base_path) / file_path.lstrip("/")
        
        if not full_path.exists():
            return f"File not found: {file_path}"
        
        suffix = full_path.suffix.lower()
        
        # Select linter based on file extension
        if suffix == ".py":
            # Python: use ruff
            result = subprocess.run(
                ["ruff", "check", "--output-format=text", str(full_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(base_path),
            )
            if result.returncode == 0:
                return "PASS"
            else:
                errors = result.stdout.strip() or result.stderr.strip()
                return f"Lint errors found:\n{errors}"
        
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            # TypeScript/JavaScript: use eslint
            result = subprocess.run(
                ["npx", "eslint", "--format=compact", str(full_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(base_path),
            )
            if result.returncode == 0:
                return "PASS"
            else:
                errors = result.stdout.strip() or result.stderr.strip()
                return f"Lint errors found:\n{errors}"
        
        elif suffix == ".java":
            # Java: syntax check with javac
            result = subprocess.run(
                ["javac", "-Xlint:all", "-d", "/tmp", str(full_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(base_path),
            )
            if result.returncode == 0:
                return "PASS"
            else:
                errors = result.stderr.strip() or result.stdout.strip()
                return f"Compilation errors found:\n{errors}"
        
        else:
            return f"No linter configured for {suffix} files. Skipping lint check."
    
    except subprocess.TimeoutExpired:
        return f"Lint check timed out for {file_path}"
    except FileNotFoundError as e:
        return f"Linter not found: {e.filename}. Make sure the linter is installed."
    except Exception as e:
        logger.error(f"lint_file failed: {e}")
        return f"lint_file error: {str(e)}"


@tool(args_schema=VerifyFileInput)
def verify_file(file_path: str, repo: str) -> str:
    """
    Re-read a file from disk to verify its contents.
    
    Use this to confirm that code changes were written correctly.
    This forces a fresh read from disk, not from LLM memory.
    
    Returns the file content or an error if the file doesn't exist.
    """
    try:
        base_path = _get_repo_path(repo)
        if not base_path:
            return f"Repository path not configured for '{repo}'"
        
        full_path = Path(base_path) / file_path.lstrip("/")
        
        if not full_path.exists():
            return f"VERIFICATION FAILED: File not found: {file_path}"
        
        content = full_path.read_text(encoding="utf-8")
        
        lines = len(content.splitlines())
        size = len(content)
        
        # Truncate very large files but include enough to verify
        if len(content) > 15000:
            content = content[:15000] + f"\n\n... (truncated, {size} bytes total)"
        
        return f"VERIFIED: {file_path} ({lines} lines, {size} bytes)\n\n{content}"
    
    except Exception as e:
        logger.error(f"verify_file failed: {e}")
        return f"VERIFICATION FAILED: {str(e)}"


@tool(args_schema=DeleteFileInput)
def delete_file(file_path: str, repo: str, reason: str) -> str:
    """
    Mark a file for deletion.
    The file will be removed and the deletion committed by the git node.

    Use this tool only when a file is genuinely no longer needed.
    """
    try:
        from codegen.change_tracker import get_tracker

        base_path = _get_repo_path(repo)
        if not base_path:
            return f"Repository path not configured for '{repo}'"

        abs_path = Path(base_path) / file_path.lstrip("/")
        if not abs_path.exists():
            return f"File not found: {file_path} in {repo}"

        original = abs_path.read_text(encoding="utf-8", errors="replace")
        tracker = get_tracker()
        change = tracker.record_edit(
            repo=repo,
            file_path=file_path,
            abs_path=abs_path,
            original_content=original,
            new_content="",   # Empty signals deletion to git node
            description=f"Delete: {reason}",
            is_new_file=False,
        )
        change.is_deleted = True

        return (
            f"Marked '{file_path}' in {repo} for deletion.\n"
            f"Reason: {reason}\n"
            f"File will be removed and committed in the next step."
        )

    except Exception as e:
        logger.error(f"delete_file failed: {e}")
        return f"delete_file tool error: {str(e)}"


# Export all tools
ALL_TOOLS = [
    search_code,
    get_node_graph,
    find_callers,
    find_api_contracts,
    cross_repo_trace,
    get_file_content,
    get_module_structure,
]

# Code-generation tools (added to subagents in build mode)
CODEGEN_TOOLS = [
    edit_file,
    create_file,
    delete_file,
]

# Verification tools for the verify → lint → fix loop (Item 6)
VERIFY_TOOLS = [
    lint_file,
    verify_file,
]
