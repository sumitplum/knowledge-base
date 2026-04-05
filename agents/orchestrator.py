"""
LangGraph orchestrator for multi-repo analysis AND feature building.

Two modes:
  - analyze: classic flow (parse → search → subagents → plan)
  - build:   analysis flow + approval_gate → generate_code_changes
             → git_operations → create_pull_requests

Upgraded with:
- LLM-powered generate_plan using structured output (Item 7)
- Supervisor-style architecture with OrchestratorDecision (Item 3)
- Parallel subagent dispatch via Send API (Item 5)
- Persistent checkpointing with SqliteSaver (Item 9)
- Cross-repo rollback compensation (Item 10)
"""

import logging
import operator
import re
import sys
from datetime import datetime

# Configure root logger once so all [MILESTONE] logs reach the terminal.
# Only add the handler if none exist yet (guards against double-init on
# Streamlit hot-reload, which re-executes the module).
_root = logging.getLogger()
if not _root.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    _root.addHandler(_handler)
_root.setLevel(logging.INFO)
from pathlib import Path
from typing import Optional, Any, TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from pydantic import BaseModel, Field

from config import settings
from .tools import ALL_TOOLS, CODEGEN_TOOLS, VERIFY_TOOLS, search_code, find_api_contracts, cross_repo_trace
from .subagent import Subagent, SubagentResponse, create_orbit_subagent, create_trinity_subagent
from .prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    FEATURE_PLANNER_PROMPT,
    ORBIT_CODEGEN_PROMPT,
    TRINITY_CODEGEN_PROMPT,
    PLAN_GENERATION_PROMPT,
    SUPERVISOR_DECISION_PROMPT,
)
from .guardrails import get_intent_guard
from .rate_limiter import get_rate_limiter
from .exceptions import SecurityViolation, RateLimitExceeded

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Output Models (Items 3, 7)
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    """A single step in the implementation plan."""
    model_config = {"arbitrary_types_allowed": True}
    step_number: int = Field(description="Step number in sequence")
    repo: Literal["orbit", "trinity", "both"] = Field(description="Which repo this step affects")
    description: str = Field(description="What to do in this step")
    files: list[str] = Field(default_factory=list, description="Files to modify/create")
    complexity: Literal["simple", "moderate", "complex"] = Field(default="moderate", description="Estimated complexity")
    depends_on: list[int] = Field(default_factory=list, description="Step numbers this depends on")


class ImplementationPlan(BaseModel):
    """Structured implementation plan from LLM (Item 7)."""
    model_config = {"arbitrary_types_allowed": True}
    title: str = Field(description="Short title for the feature (max 60 chars)")
    summary: str = Field(description="1-2 sentence summary of what will be implemented")
    steps: list[PlanStep] = Field(default_factory=list, description="Ordered implementation steps")
    risks: list[str] = Field(default_factory=list, description="Potential risks or concerns")
    testing_strategy: str = Field(default="", description="How to test the feature")
    success_criteria: list[str] = Field(default_factory=list, description="How we know the feature is complete")
    estimated_effort: str = Field(default="", description="Rough effort estimate (e.g., '2-3 hours', '1 day')")


class OrchestratorDecision(BaseModel):
    """Structured output for supervisor-style orchestrator decisions (Item 3)."""
    model_config = {"arbitrary_types_allowed": True}
    next_action: Literal[
        "fetch_context",     # Need more information from codebase
        "delegate_orbit",    # Dispatch to Orbit subagent
        "delegate_trinity",  # Dispatch to Trinity subagent
        "delegate_both",     # Dispatch to both subagents in parallel
        "generate_plan",     # Ready to generate implementation plan
        "build",             # Proceed to build phase
        "done",              # Analysis complete
    ] = Field(description="What action to take next")
    reasoning: str = Field(description="Why this action is the right choice")
    query: str = Field(default="", description="Query to pass to the next action (if applicable)")

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict):
    """State for the analysis + optional build workflow."""
    feature_description: str
    search_results: list[dict]
    cross_repo_links: list[dict]
    orbit_analysis: Optional[SubagentResponse]
    trinity_analysis: Optional[SubagentResponse]
    impact_map: dict
    pr_draft: dict
    messages: Annotated[list[Any], operator.add]  # Reducer for parallel writes (Item 5)
    needs_clarification: bool
    clarification_question: str
    current_step: str
    error: Optional[str]

    # Feature Builder fields (only populated in build mode)
    build_mode: bool               # True ↔ generate code + commit + PR
    dry_run: bool                  # True ↔ generate but don't write/push
    approved: bool                 # human gate: True = proceed with build
    feature_slug: str              # URL-safe slug derived from description
    branch_names: dict             # {repo: branch_name}
    code_gen_results: list[dict]   # per-file results from CodeGenerator
    commit_shas: dict              # {repo: sha}
    pr_results: list[dict]         # serialised PRResult objects
    plan_text: str                 # the plan to embed in PR body
    
    # Structured plan (Item 7)
    implementation_plan: Optional[ImplementationPlan]
    
    # Supervisor fields (Item 3)
    iteration_count: int
    max_orchestrator_iterations: int
    
    # Rollback tracking (Item 10)
    succeeded_repos: list[str]
    compensation_results: list[dict]  # Results of rollback compensation
    
    # Checkpointing (Item 9)
    thread_id: str


def _get_checkpointer() -> Optional[SqliteSaver]:
    """
    Initialize the SqliteSaver for persistent checkpointing (Item 9).

    Uses check_same_thread=False so the connection can be accessed from the
    worker threads that LangGraph spawns for parallel Send() fan-out nodes.

    Returns None if checkpointing fails (e.g., permission issues).
    """
    try:
        import sqlite3
        db_path = Path(settings.checkpoint_db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is required because LangGraph runs parallel
        # nodes (Send fan-out) in separate threads that all share this saver.
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # Serialized mode ensures thread-safe writes without locking the GIL.
        conn.isolation_level = None  # autocommit — SqliteSaver manages transactions
        saver = SqliteSaver(conn)
        return saver
    except Exception as e:
        logger.warning(f"Failed to initialize checkpointer: {e}. Continuing without checkpointing.")
        return None


def create_orchestrator(checkpointer: Optional[SqliteSaver] = None):
    """Create the LangGraph orchestrator."""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # Create subagents
    orbit_agent = create_orbit_subagent()
    trinity_agent = create_trinity_subagent()
    
    # Define nodes
    def parse_request(state: AnalysisState) -> AnalysisState:
        """Parse and validate the feature request."""
        logger.info("=" * 60)
        logger.info("[MILESTONE] STEP 1/8 — parse_request")
        logger.info(f"  build_mode : {state.get('build_mode', False)}")
        logger.info(f"  dry_run    : {state.get('dry_run', True)}")
        logger.info(f"  feature    : {state['feature_description'][:120]}")
        state["current_step"] = "parse_request"
        
        description = state["feature_description"]
        
        # Check if description is clear enough
        if len(description.strip()) < 10:
            state["needs_clarification"] = True
            state["clarification_question"] = "Could you provide more details about the feature you want to implement?"
            logger.warning("  → needs_clarification=True (too short). Routing to END.")
            return state
        
        state["needs_clarification"] = False
        logger.info("  → OK. Routing to supervisor_decide.")
        return state
    
    def broad_search(state: AnalysisState) -> AnalysisState:
        """Perform broad search across both repos."""
        logger.info("Running broad search...")
        state["current_step"] = "broad_search"
        
        description = state["feature_description"]
        
        try:
            # Search for relevant code
            results = search_code.invoke({
                "query": description,
                "limit": 20,
            })
            state["search_results"] = [{"raw": results}]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            state["error"] = str(e)
        
        return state
    
    def identify_cross_links(state: AnalysisState) -> AnalysisState:
        """Identify cross-repository links."""
        logger.info("Identifying cross-repo links...")
        state["current_step"] = "identify_cross_links"
        
        try:
            # Get API contracts
            contracts = find_api_contracts.invoke({})
            state["cross_repo_links"] = [{"contracts": contracts}]
            
        except Exception as e:
            logger.error(f"Cross-link identification failed: {e}")
            state["cross_repo_links"] = []
        
        return state
    
    def dispatch_orbit(state: AnalysisState) -> dict:
        """Dispatch analysis to Orbit subagent.

        Returns only the fields this node writes so LangGraph can safely merge
        parallel writes from dispatch_orbit and dispatch_trinity.
        """
        logger.info("-" * 60)
        logger.info("[MILESTONE] STEP — dispatch_orbit (frontend analysis)")
        logger.info(f"  iteration  : {state.get('iteration_count', 0)}")
        try:
            context = {
                "search_results": state.get("search_results", []),
                "cross_repo_links": state.get("cross_repo_links", []),
            }
            response = orbit_agent.analyze(
                query=state["feature_description"],
                context=context,
            )
            logger.info(f"  → Orbit done. confidence={response.confidence}  files={len(response.impacted_files)}")
        except Exception as e:
            logger.error(f"  → Orbit analysis FAILED: {e}")
            response = SubagentResponse(
                analysis=f"Analysis failed: {str(e)}",
                confidence="low",
            )
        return {"orbit_analysis": response}

    def dispatch_trinity(state: AnalysisState) -> dict:
        """Dispatch analysis to Trinity subagent.

        Returns only the fields this node writes so LangGraph can safely merge
        parallel writes from dispatch_orbit and dispatch_trinity.
        """
        logger.info("-" * 60)
        logger.info("[MILESTONE] STEP — dispatch_trinity (backend analysis)")
        logger.info(f"  iteration  : {state.get('iteration_count', 0)}")
        try:
            context = {
                "search_results": state.get("search_results", []),
                "cross_repo_links": state.get("cross_repo_links", []),
            }
            response = trinity_agent.analyze(
                query=state["feature_description"],
                context=context,
            )
            logger.info(f"  → Trinity done. confidence={response.confidence}  files={len(response.impacted_files)}")
        except Exception as e:
            logger.error(f"  → Trinity analysis FAILED: {e}")
            response = SubagentResponse(
                analysis=f"Analysis failed: {str(e)}",
                confidence="low",
            )
        return {"trinity_analysis": response}
    
    def merge_results(state: AnalysisState) -> AnalysisState:
        """Merge results from both subagents."""
        logger.info("-" * 60)
        logger.info("[MILESTONE] STEP — merge_results")
        orbit_done = bool(state.get("orbit_analysis"))
        trinity_done = bool(state.get("trinity_analysis"))
        logger.info(f"  orbit_done={orbit_done}  trinity_done={trinity_done}")
        state["current_step"] = "merge_results"
        
        orbit = state.get("orbit_analysis")
        trinity = state.get("trinity_analysis")
        
        # Build unified impact map
        impact_map = {
            "orbit": {
                "files": orbit.impacted_files if orbit else [],
                "functions": orbit.impacted_functions if orbit else [],
                "confidence": orbit.confidence if orbit else "unknown",
            },
            "trinity": {
                "files": trinity.impacted_files if trinity else [],
                "functions": trinity.impacted_functions if trinity else [],
                "confidence": trinity.confidence if trinity else "unknown",
            },
            "cross_repo": state.get("cross_repo_links", []),
        }
        
        state["impact_map"] = impact_map
        return state
    
    def generate_plan(state: AnalysisState) -> AnalysisState:
        """
        Generate the final plan and PR draft using LLM with structured output (Item 7).
        
        Replaces hardcoded dict assembly with an LLM call that reasons about
        the best implementation approach.
        """
        logger.info("=" * 60)
        logger.info("[MILESTONE] STEP — generate_plan")
        logger.info(f"  build_mode : {state.get('build_mode', False)}")
        logger.info(f"  orbit      : {bool(state.get('orbit_analysis'))}  trinity : {bool(state.get('trinity_analysis'))}")
        state["current_step"] = "generate_plan"
        
        orbit = state.get("orbit_analysis")
        trinity = state.get("trinity_analysis")
        cross_links = state.get("cross_repo_links", [])
        
        # Build context for the LLM
        context_parts = [f"Feature request: {state['feature_description']}"]
        
        if trinity:
            context_parts.append(f"""
Trinity (Backend) Analysis:
- Confidence: {trinity.confidence}
- Impacted files: {', '.join(trinity.impacted_files[:10]) or 'none identified'}
- Impacted functions: {', '.join(trinity.impacted_functions[:10]) or 'none identified'}
- Suggested changes: {chr(10).join(f'  - {c}' for c in trinity.suggested_changes[:5]) or 'none'}
- Analysis: {trinity.analysis[:1000] if trinity.analysis else 'none'}
""")
        
        if orbit:
            context_parts.append(f"""
Orbit (Frontend) Analysis:
- Confidence: {orbit.confidence}
- Impacted files: {', '.join(orbit.impacted_files[:10]) or 'none identified'}
- Impacted functions: {', '.join(orbit.impacted_functions[:10]) or 'none identified'}
- Suggested changes: {chr(10).join(f'  - {c}' for c in orbit.suggested_changes[:5]) or 'none'}
- Analysis: {orbit.analysis[:1000] if orbit.analysis else 'none'}
""")
        
        if cross_links:
            context_parts.append(f"Cross-repo API links: {cross_links[:5]}")
        
        context = "\n\n".join(context_parts)
        
        # Use LLM with structured output to generate the plan
        llm_structured = llm.with_structured_output(ImplementationPlan)
        
        try:
            plan: ImplementationPlan = llm_structured.invoke([
                SystemMessage(content=PLAN_GENERATION_PROMPT),
                HumanMessage(content=context),
            ])
            
            state["implementation_plan"] = plan
            
            # Build plan_text for PR body from structured plan
            plan_lines = [f"# {plan.title}", "", plan.summary, "", "## Implementation Steps", ""]
            for step in plan.steps:
                deps = f" (depends on: {', '.join(str(d) for d in step.depends_on)})" if step.depends_on else ""
                plan_lines.append(f"{step.step_number}. [{step.repo.upper()}] {step.description}{deps}")
                if step.files:
                    for f in step.files[:5]:
                        plan_lines.append(f"   - `{f}`")
            
            if plan.risks:
                plan_lines.extend(["", "## Risks", ""])
                for risk in plan.risks:
                    plan_lines.append(f"- {risk}")
            
            if plan.testing_strategy:
                plan_lines.extend(["", "## Testing Strategy", "", plan.testing_strategy])
            
            state["plan_text"] = "\n".join(plan_lines)
            
            # Build PR draft from structured plan
            pr_draft = {
                "title": plan.title,
                "summary": [plan.summary],
                "orbit_changes": [s.description for s in plan.steps if s.repo in ("orbit", "both")],
                "trinity_changes": [s.description for s in plan.steps if s.repo in ("trinity", "both")],
                "testing": [plan.testing_strategy] if plan.testing_strategy else [],
                "risks": plan.risks,
            }
            
        except Exception as e:
            logger.error(f"LLM plan generation failed: {e}, falling back to basic plan")
            
            # Fallback to basic plan generation
            pr_draft = {
                "title": f"Feature: {state['feature_description'][:50]}",
                "summary": [],
                "orbit_changes": [],
                "trinity_changes": [],
                "testing": [],
                "risks": [],
            }
            
            if orbit:
                pr_draft["summary"].append(f"**Frontend (Orbit):** {orbit.confidence} confidence")
                pr_draft["orbit_changes"] = orbit.suggested_changes
            
            if trinity:
                pr_draft["summary"].append(f"**Backend (Trinity-v2):** {trinity.confidence} confidence")
                pr_draft["trinity_changes"] = trinity.suggested_changes
            
            if orbit and orbit.impacted_files:
                pr_draft["testing"].append(f"Test {len(orbit.impacted_files)} Orbit files")
            if trinity and trinity.impacted_files:
                pr_draft["testing"].append(f"Test {len(trinity.impacted_files)} Trinity files")
            
            # Build basic plan text
            plan_lines = [f"Feature: {state['feature_description']}", ""]
            if orbit:
                plan_lines.append(f"Orbit changes: {orbit.analysis[:500]}")
            if trinity:
                plan_lines.append(f"Trinity changes: {trinity.analysis[:500]}")
            state["plan_text"] = "\n".join(plan_lines)
        
        state["pr_draft"] = pr_draft
        plan = state.get("implementation_plan")
        if plan:
            logger.info(f"  → Plan generated: '{plan.title}'  steps={len(plan.steps)}")
        else:
            logger.warning("  → Plan generation fell back to basic (no structured plan)")
        logger.info(f"  → Routing next: {'approval_gate (build)' if state.get('build_mode') else 'END (analyze only)'}")
        return state

    # ------------------------------------------------------------------ #
    # Feature Builder nodes                                               #
    # ------------------------------------------------------------------ #

    def approval_gate(state: AnalysisState) -> AnalysisState:
        """
        Check whether the human has approved the build.

        In API/programmatic mode the caller sets approved=True before
        invoking the graph.  The UI sets it via a Streamlit button.
        If not approved we land at END with a message.
        """
        logger.info("=" * 60)
        logger.info("[MILESTONE] STEP — approval_gate")
        logger.info(f"  approved   : {state.get('approved', False)}")
        logger.info(f"  dry_run    : {state.get('dry_run', True)}")
        state["current_step"] = "approval_gate"
        if not state.get("approved"):
            state["error"] = (
                "Build not approved. Review the analysis plan and set "
                "approved=True to proceed with code generation."
            )
            logger.warning("  → NOT approved. Routing to END.")
        else:
            logger.info("  → Approved. Routing to generate_code_changes.")
        return state

    def generate_code_changes(state: AnalysisState) -> AnalysisState:
        """
        Dispatch codegen subagents to Orbit and Trinity.

        Each subagent receives the full analysis context and the
        codegen tools (edit_file, create_file, delete_file).
        
        Tracks succeeded_repos for rollback compensation (Item 10).
        """
        logger.info("=" * 60)
        logger.info("[MILESTONE] STEP — generate_code_changes")
        logger.info(f"  dry_run    : {state.get('dry_run', True)}")
        logger.info(f"  orbit_conf : {state.get('orbit_analysis').confidence if state.get('orbit_analysis') else 'n/a'}")
        logger.info(f"  trinity_cf : {state.get('trinity_analysis').confidence if state.get('trinity_analysis') else 'n/a'}")
        state["current_step"] = "generate_code_changes"

        from codegen.change_tracker import get_tracker
        tracker = get_tracker()
        tracker.reset()  # Clean slate for this run

        feature_description = state["feature_description"]
        orbit_analysis = state.get("orbit_analysis")
        trinity_analysis = state.get("trinity_analysis")
        pr_draft = state.get("pr_draft", {})

        plan_lines = [f"Feature: {feature_description}", ""]
        if orbit_analysis:
            plan_lines.append(f"Orbit changes: {orbit_analysis.analysis[:500]}")
        if trinity_analysis:
            plan_lines.append(f"Trinity changes: {trinity_analysis.analysis[:500]}")
        plan_text = "\n".join(plan_lines)
        state["plan_text"] = plan_text

        all_results: list[dict] = []
        succeeded_repos: list[str] = []  # Track for rollback (Item 10)
        has_failure = False

        dry_run = state.get("dry_run", True)

        # --- Trinity first (stabilise API contract) ---
        trinity_path = settings.trinity_repo_path
        if trinity_path and trinity_analysis and trinity_analysis.confidence != "low":
            logger.info("Running Trinity codegen subagent…")
            # Inject dry_run into tool context via environment variable so
            # edit_file / create_file know whether to actually write files.
            import os
            os.environ["KB_DRY_RUN"] = "1" if dry_run else "0"
            trinity_codegen = Subagent(
                repo="trinity",
                system_prompt=TRINITY_CODEGEN_PROMPT,
                tools=CODEGEN_TOOLS + VERIFY_TOOLS + ALL_TOOLS,
                max_iterations=15,
            )
            try:
                resp = trinity_codegen.analyze(
                    query=(
                        f"Implement the following feature in Trinity-v2 (backend):\n\n"
                        f"{feature_description}\n\n"
                        f"Analysis: {trinity_analysis.analysis}\n\n"
                        f"Suggested changes: {', '.join(trinity_analysis.suggested_changes)}"
                    ),
                    context={"plan": plan_text, "dry_run": dry_run},
                )
                all_results.append({"repo": "trinity", "analysis": resp.analysis})
                succeeded_repos.append("trinity")  # Track success (Item 10)
            except Exception as e:
                logger.error(f"Trinity codegen failed: {e}")
                all_results.append({"repo": "trinity", "error": str(e)})
                has_failure = True

        # --- Orbit second (consumes any new API) ---
        orbit_path = settings.orbit_repo_path
        if orbit_path and orbit_analysis and orbit_analysis.confidence != "low":
            logger.info("Running Orbit codegen subagent…")
            orbit_codegen = Subagent(
                repo="orbit",
                system_prompt=ORBIT_CODEGEN_PROMPT,
                tools=CODEGEN_TOOLS + VERIFY_TOOLS + ALL_TOOLS,
                max_iterations=15,
            )
            try:
                resp = orbit_codegen.analyze(
                    query=(
                        f"Implement the following feature in Orbit (frontend):\n\n"
                        f"{feature_description}\n\n"
                        f"Analysis: {orbit_analysis.analysis}\n\n"
                        f"Suggested changes: {', '.join(orbit_analysis.suggested_changes)}"
                    ),
                    context={"plan": plan_text, "dry_run": dry_run},
                )
                all_results.append({"repo": "orbit", "analysis": resp.analysis})
                succeeded_repos.append("orbit")  # Track success (Item 10)
            except Exception as e:
                logger.error(f"Orbit codegen failed: {e}")
                all_results.append({"repo": "orbit", "error": str(e)})
                has_failure = True

        state["code_gen_results"] = all_results
        state["succeeded_repos"] = succeeded_repos  # Store for compensation (Item 10)

        if not tracker.has_changes():
            logger.warning("  → No file changes were recorded by codegen subagents")

        # Set error flag if any failures occurred (for routing to compensate)
        if has_failure and succeeded_repos:
            # Partial failure: some repos succeeded, need compensation
            state["error"] = f"Partial codegen failure. Succeeded: {succeeded_repos}"
            logger.error(f"  → Partial failure. succeeded={succeeded_repos}. Routing to compensate.")
        elif has_failure:
            state["error"] = "All codegen attempts failed"
            logger.error("  → All codegen failed. Routing to END (abort).")
        else:
            changed = tracker.repos_with_changes()
            logger.info(f"  → Codegen complete. succeeded={succeeded_repos}  changed_repos={list(changed)}")
            logger.info("  → Routing to git_operations.")

        return state

    def compensate(state: AnalysisState) -> AnalysisState:
        """
        Rollback compensation node (Item 10).
        
        If codegen fails for one repo after another succeeded, this node
        rolls back the tracked changes in succeeded repos. Branch deletion
        is not attempted since branches may not exist yet (git_operations
        happens after generate_code_changes).
        """
        logger.info("=" * 60)
        logger.warning("[MILESTONE] STEP — compensate (rollback)")
        logger.warning(f"  rolling back repos: {state.get('succeeded_repos', [])}")
        state["current_step"] = "compensate"
        
        from codegen.change_tracker import get_tracker
        
        tracker = get_tracker()
        succeeded_repos = state.get("succeeded_repos", [])
        
        compensation_results: list[dict] = []
        
        for repo in succeeded_repos:
            try:
                # Rollback tracked changes for this repo
                logger.info(f"Rolling back changes for {repo}")
                tracker.rollback(repo)
                
                compensation_results.append({
                    "repo": repo,
                    "action": "changes_rolled_back",
                    "success": True,
                })
                
            except Exception as e:
                logger.error(f"Compensation failed for {repo}: {e}")
                compensation_results.append({
                    "repo": repo,
                    "action": "rollback_failed",
                    "error": str(e),
                    "success": False,
                })
        
        state["compensation_results"] = compensation_results
        state["error"] = f"Codegen failed. Rolled back: {succeeded_repos}."
        
        return state

    def git_operations(state: AnalysisState) -> AnalysisState:
        """
        Create feature branches, write files, and commit for each repo
        that has tracked changes.
        """
        logger.info("=" * 60)
        logger.info("[MILESTONE] STEP — git_operations")
        logger.info(f"  dry_run    : {state.get('dry_run', True)}")
        state["current_step"] = "git_operations"

        from codegen.change_tracker import get_tracker
        from codegen.git_ops import GitOps, GitOpsError

        tracker = get_tracker()
        dry_run = state.get("dry_run", True)

        feature_slug = _slugify(state.get("feature_description", "feature"))
        state["feature_slug"] = feature_slug

        branch_names: dict[str, str] = {}
        commit_shas: dict[str, str] = {}

        # For git operations we need the actual repo root, not a subdirectory.
        # orbit_repo_path may point to apps/trinity (the app) — walk up to find .git root.
        def _find_git_root(path) -> Optional[Path]:
            from pathlib import Path as P
            p = P(path).resolve()
            for candidate in [p, *p.parents]:
                if (candidate / ".git").exists():
                    return candidate
            return p  # Fallback: use as-is

        repo_path_map = {
            "orbit": _find_git_root(settings.orbit_repo_path) if settings.orbit_repo_path else None,
            "trinity": _find_git_root(settings.trinity_repo_path) if settings.trinity_repo_path else None,
        }
        logger.info(f"Git repo roots: {repo_path_map}")

        for repo_name in tracker.repos_with_changes():
            repo_path = repo_path_map.get(repo_name)
            if not repo_path:
                logger.warning(f"No path for repo '{repo_name}'; skipping git ops")
                continue

            changes = tracker.get_changes(repo_name)

            git = GitOps(repo_path=repo_path, repo_name=repo_name, dry_run=dry_run)
            branch = git.make_branch_name(feature_slug)

            try:
                # allow_dirty=True: untracked files (e.g. secrets/, .DS_Store) shouldn't block builds
                git.create_feature_branch(branch, base="main", allow_dirty=True)
                branch_names[repo_name] = branch

                # Separate out deletions and apply them
                to_delete = [c for c in changes if c.is_deleted]
                to_write = [c for c in changes if not c.is_deleted]

                commit_msg = (
                    f"feat: {state['feature_description'][:72]}\n\n"
                    f"Generated by Knowledge Base Feature Builder.\n"
                    f"Files: {', '.join(c.file_path for c in changes)}"
                )

                # Handle deletions first
                if to_delete and not dry_run:
                    for c in to_delete:
                        c.abs_path.unlink(missing_ok=True)
                    import git as gitpkg
                    git_repo = gitpkg.Repo(repo_path)
                    git_repo.index.remove([str(c.abs_path.relative_to(repo_path)) for c in to_delete])

                sha = git.apply_and_commit(to_write, commit_msg)
                if sha:
                    commit_shas[repo_name] = sha

                if not dry_run:
                    git.push_branch(branch)

            except Exception as e:
                logger.error(f"Git ops failed for {repo_name}: {e}")
                tracker.rollback(repo_name)
                state.setdefault("error", "")
                state["error"] = (state["error"] or "") + f"\nGit error ({repo_name}): {e}"

        state["branch_names"] = branch_names
        state["commit_shas"] = commit_shas
        logger.info(f"  → git_operations done. branches={branch_names}  shas={commit_shas}")
        logger.info("  → Routing to create_pull_requests.")
        return state

    def create_pull_requests(state: AnalysisState) -> AnalysisState:
        """
        Open GitHub PRs for each repo that has a feature branch.
        
        Uses LLM-generated PR content (Item 1).
        """
        logger.info("=" * 60)
        logger.info("[MILESTONE] STEP — create_pull_requests")
        state["current_step"] = "create_pull_requests"

        from codegen.change_tracker import get_tracker
        from codegen.pr_creator import PRCreator

        tracker = get_tracker()
        dry_run = state.get("dry_run", True)
        branch_names = state.get("branch_names", {})
        logger.info(f"  dry_run    : {dry_run}")
        logger.info(f"  branches   : {branch_names}")

        if not branch_names:
            state["pr_results"] = []
            return state

        creator = PRCreator(dry_run=dry_run, llm=llm)
        try:
            results = creator.create_prs(
                tracker=tracker,
                feature_title=state["feature_description"][:60],
                feature_description=state["feature_description"],
                branch_names=branch_names,
                plan_text=state.get("plan_text"),
                orbit_analysis=state.get("orbit_analysis"),
                trinity_analysis=state.get("trinity_analysis"),
            )
            state["pr_results"] = [
                {
                    "repo": r.repo_slug,
                    "branch": r.branch,
                    "pr_url": r.pr_url,
                    "pr_number": r.pr_number,
                    "dry_run": r.dry_run,
                    "success": r.success,
                    "error": r.error,
                    "was_updated": r.was_updated,  # Item 10: idempotency flag
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"  → PR creation FAILED: {e}")
            state["pr_results"] = [{"error": str(e)}]

        pr_results = state.get("pr_results", [])
        for pr in pr_results:
            repo = pr.get("repo", "?")
            if pr.get("success"):
                logger.info(f"  → PR created: [{repo}] {pr.get('pr_url', '(dry run)')}")
            elif pr.get("error"):
                logger.error(f"  → PR failed:  [{repo}] {pr['error']}")
        logger.info("[MILESTONE] PIPELINE COMPLETE → END")
        logger.info("=" * 60)
        return state

    # ------------------------------------------------------------------ #
    # Supervisor Orchestrator Nodes (Item 3)                               #
    # ------------------------------------------------------------------ #

    def supervisor_decide(state: AnalysisState) -> AnalysisState:
        """
        Supervisor decision node (Item 3).
        
        Uses LLM with structured output to decide the next action based on
        accumulated context. This enables cyclic workflows where the 
        orchestrator can request more context or dispatch subagents as needed.
        """
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_orchestrator_iterations", 8)
        orbit_done = bool(state.get("orbit_analysis"))
        trinity_done = bool(state.get("trinity_analysis"))
        logger.info("-" * 60)
        logger.info(f"[MILESTONE] STEP — supervisor_decide  (iter {iteration + 1}/{max_iterations})")
        logger.info(f"  build_mode : {state.get('build_mode', False)}")
        logger.info(f"  orbit_done : {orbit_done}  trinity_done : {trinity_done}")
        state["current_step"] = "supervisor_decide"
        
        # Check iteration limit
        if iteration >= max_iterations:
            logger.warning(f"  → Max iterations ({max_iterations}) reached, forcing done")
            state["iteration_count"] = iteration
            return state
        
        state["iteration_count"] = iteration + 1
        
        # Build context for the supervisor prompt
        orbit_analysis = state.get("orbit_analysis")
        trinity_analysis = state.get("trinity_analysis")
        
        build_mode = state.get("build_mode", False)
        prompt = SUPERVISOR_DECISION_PROMPT.format(
            feature_description=state["feature_description"],
            build_mode_label="BUILD (generate code + PRs after planning)" if build_mode else "ANALYZE ONLY",
            search_results=str(state.get("search_results", []))[:2000],
            cross_repo_links=str(state.get("cross_repo_links", []))[:1000],
            orbit_status="COMPLETE" if orbit_analysis else "NOT YET RUN",
            orbit_analysis=orbit_analysis.analysis[:500] if orbit_analysis else "Not yet analyzed",
            trinity_status="COMPLETE" if trinity_analysis else "NOT YET RUN",
            trinity_analysis=trinity_analysis.analysis[:500] if trinity_analysis else "Not yet analyzed",
            iteration_count=iteration + 1,
            max_iterations=max_iterations,
        )
        
        llm_structured = llm.with_structured_output(OrchestratorDecision)
        
        try:
            decision: OrchestratorDecision = llm_structured.invoke([
                SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            
            next_action = decision.next_action

            # Override bad LLM decisions: don't re-delegate a subagent that
            # already completed. If only one is done and we have iterations
            # left, ask for the missing one. If both are done (or we're nearly
            # out of iterations), go straight to generate_plan.
            if next_action in ("delegate_orbit", "delegate_trinity", "delegate_both"):
                orbit_done = bool(state.get("orbit_analysis"))
                trinity_done = bool(state.get("trinity_analysis"))
                remaining = max_iterations - (iteration + 1)

                if orbit_done and trinity_done:
                    next_action = "generate_plan"
                    logger.info("  → LLM wanted to re-delegate but both done → overriding to generate_plan")
                elif next_action == "delegate_orbit" and orbit_done:
                    if trinity_done or remaining <= 1:
                        next_action = "generate_plan"
                        logger.info("  → LLM re-delegated orbit (done) → overriding to generate_plan")
                    else:
                        next_action = "delegate_trinity"
                        logger.info("  → LLM re-delegated orbit (done) → switching to delegate_trinity")
                elif next_action == "delegate_trinity" and trinity_done:
                    if orbit_done or remaining <= 1:
                        next_action = "generate_plan"
                        logger.info("  → LLM re-delegated trinity (done) → overriding to generate_plan")
                    else:
                        next_action = "delegate_orbit"
                        logger.info("  → LLM re-delegated trinity (done) → switching to delegate_orbit")

            # Store the (possibly corrected) decision in messages for downstream nodes
            state["messages"] = state.get("messages", []) + [
                {"role": "supervisor", "decision": next_action, "reasoning": decision.reasoning, "query": decision.query}
            ]

            logger.info(f"  → DECISION: {next_action}")
            logger.info(f"    reason  : {decision.reasoning[:120]}")
            if decision.query:
                logger.info(f"    query   : {decision.query[:80]}")
            
        except Exception as e:
            logger.error(f"  → Supervisor decision FAILED: {e}. Defaulting to delegate_both.")
            # Default to delegate_both on error
            state["messages"] = state.get("messages", []) + [
                {"role": "supervisor", "decision": "delegate_both", "reasoning": f"Error: {e}", "query": state["feature_description"]}
            ]
        
        return state

    def context_fetcher(state: AnalysisState) -> AnalysisState:
        """
        Fetch additional context from the codebase (Item 3).
        
        Called when supervisor decides more information is needed before
        dispatching to subagents. Uses search tools to gather context.
        """
        logger.info("-" * 60)
        logger.info("[MILESTONE] STEP — context_fetcher")
        state["current_step"] = "context_fetcher"
        
        # Get the query from the last supervisor decision
        messages = state.get("messages", [])
        query = state["feature_description"]  # Default
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "supervisor":
                query = msg.get("query", query)
                break
        
        logger.info(f"  query      : {query[:100]}")
        try:
            # Run search
            results = search_code.invoke({"query": query, "limit": 15})
            state["search_results"] = state.get("search_results", []) + [{"query": query, "results": results}]
            logger.info("  → search_code complete. Results appended.")
            
            # Also check for API contracts if not already done
            if not state.get("cross_repo_links"):
                contracts = find_api_contracts.invoke({})
                state["cross_repo_links"] = [{"contracts": contracts}]
                logger.info("  → find_api_contracts complete.")
            
        except Exception as e:
            logger.error(f"  → Context fetch FAILED: {e}")
            state["messages"] = state.get("messages", []) + [
                {"role": "context_fetcher", "error": str(e)}
            ]
        
        return state

    def parallel_subagent_dispatch(state: AnalysisState) -> list[Send]:
        """
        Fan-out to both Orbit and Trinity subagents in parallel (Item 5).
        
        Returns a list of Send commands that LangGraph executes concurrently.
        The subagent nodes write back into AnalysisState and LangGraph
        merges the results using the reducer on the messages field.
        """
        logger.info("Dispatching to both subagents in parallel...")
        
        return [
            Send("dispatch_orbit", state),
            Send("dispatch_trinity", state),
        ]

    def supervisor_route_or_fanout(state: AnalysisState) -> Any:
        """
        Combined routing function that handles both regular routing and
        parallel fan-out (Item 3 + Item 5).

        Returns either a string (node name) or a list of Send objects for
        parallel execution.
        """
        messages = state.get("messages", [])
        orbit_done = bool(state.get("orbit_analysis"))
        trinity_done = bool(state.get("trinity_analysis"))
        either_done = orbit_done or trinity_done
        build_mode = state.get("build_mode", False)

        # Check iteration limit first
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_orchestrator_iterations", 8)
        if iteration >= max_iterations:
            # Go to plan if we have ANY analysis; bail only if completely empty
            if either_done:
                logger.warning(
                    f"  → Max iterations reached. orbit_done={orbit_done} trinity_done={trinity_done}. "
                    "Forcing generate_plan."
                )
                return "generate_plan"
            return END

        # Get the last supervisor decision
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "supervisor":
                decision = msg.get("decision", "done")

                if decision == "fetch_context":
                    return "context_fetcher"

                elif decision == "delegate_orbit":
                    if orbit_done:
                        # Orbit already ran — if we have enough to plan, do it;
                        # otherwise ask supervisor again with better context.
                        if trinity_done or iteration >= max_iterations - 1:
                            logger.info("  → delegate_orbit but orbit already ran + enough context → generate_plan")
                            return "generate_plan"
                        logger.info("  → delegate_orbit but orbit already ran → re-entering supervisor")
                        return "supervisor_decide"
                    return "dispatch_orbit"

                elif decision == "delegate_trinity":
                    if trinity_done:
                        # Trinity already ran — same override logic as above.
                        if orbit_done or iteration >= max_iterations - 1:
                            logger.info("  → delegate_trinity but trinity already ran + enough context → generate_plan")
                            return "generate_plan"
                        logger.info("  → delegate_trinity but trinity already ran → re-entering supervisor")
                        return "supervisor_decide"
                    return "dispatch_trinity"

                elif decision == "delegate_both":
                    if orbit_done and trinity_done:
                        logger.info("  → delegate_both but both already ran → generate_plan")
                        return "generate_plan"
                    if orbit_done:
                        return "dispatch_trinity"
                    if trinity_done:
                        return "dispatch_orbit"
                    # Parallel fan-out (Item 5)
                    logger.info("  → Parallel fan-out to both subagents")
                    return [
                        Send("dispatch_orbit", state),
                        Send("dispatch_trinity", state),
                    ]

                elif decision == "generate_plan":
                    return "generate_plan"

                elif decision == "build":
                    return "approval_gate"

                elif decision == "done":
                    # In build mode, never stop early — go to plan if we have anything
                    if build_mode and either_done:
                        logger.info("  → 'done' in build mode → overriding to generate_plan")
                        return "generate_plan"
                    return END

                else:
                    return END

        # No decision yet — kick off parallel dispatch for the first iteration
        return [
            Send("dispatch_orbit", state),
            Send("dispatch_trinity", state),
        ]

    # ------------------------------------------------------------------ #
    # Routing helpers                                                      #
    # ------------------------------------------------------------------ #

    def should_clarify(state: AnalysisState) -> str:
        if state.get("needs_clarification"):
            return "clarify"
        return "continue"

    def supervisor_router(state: AnalysisState) -> str:
        """
        Route based on supervisor's decision (Item 3).
        
        Reads the last decision from messages and routes accordingly.
        """
        messages = state.get("messages", [])
        
        # Check iteration limit first
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_orchestrator_iterations", 8)
        if iteration >= max_iterations:
            # Both analyses done? Go to plan. Otherwise, force completion.
            if state.get("orbit_analysis") and state.get("trinity_analysis"):
                return "generate_plan"
            return "done"
        
        # Get the last supervisor decision
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "supervisor":
                decision = msg.get("decision", "done")
                
                if decision == "fetch_context":
                    return "fetch_context"
                elif decision == "delegate_orbit":
                    return "dispatch_orbit"
                elif decision == "delegate_trinity":
                    return "dispatch_trinity"
                elif decision == "delegate_both":
                    return "dispatch_both"
                elif decision == "generate_plan":
                    return "generate_plan"
                elif decision == "build":
                    return "build"
                else:  # "done" or unknown
                    return "done"
        
        # No decision found, default to delegate_both for initial run
        return "dispatch_both"

    def should_build(state: AnalysisState) -> str:
        if state.get("build_mode"):
            return "build"
        return "done"

    def is_approved(state: AnalysisState) -> str:
        if state.get("error"):
            return "abort"
        return "proceed"

    # ------------------------------------------------------------------ #
    # Build the graph                                                      #
    # ------------------------------------------------------------------ #
    workflow = StateGraph(AnalysisState)

    # Supervisor nodes (Item 3)
    workflow.add_node("parse_request", parse_request)
    workflow.add_node("supervisor_decide", supervisor_decide)
    workflow.add_node("context_fetcher", context_fetcher)
    
    # Analysis nodes
    workflow.add_node("dispatch_orbit", dispatch_orbit)
    workflow.add_node("dispatch_trinity", dispatch_trinity)
    workflow.add_node("merge_results", merge_results)
    workflow.add_node("generate_plan", generate_plan)

    # Feature Builder nodes
    workflow.add_node("approval_gate", approval_gate)
    workflow.add_node("generate_code_changes", generate_code_changes)
    workflow.add_node("compensate", compensate)  # Rollback compensation (Item 10)
    workflow.add_node("git_operations", git_operations)
    workflow.add_node("create_pull_requests", create_pull_requests)

    # ------------------------------------------------------------------ #
    # Edges — Supervisor Loop (Item 3)                                     #
    # ------------------------------------------------------------------ #
    
    # Entry point
    workflow.set_entry_point("parse_request")
    workflow.add_conditional_edges(
        "parse_request",
        should_clarify,
        {"clarify": END, "continue": "supervisor_decide"},
    )
    
    # Supervisor decision routing with parallel fan-out support (Items 3 + 5).
    # No mapping dict — the router returns node name strings, Send objects, or END
    # directly, so LangGraph uses the return value verbatim.
    workflow.add_conditional_edges(
        "supervisor_decide",
        supervisor_route_or_fanout,
        [
            "context_fetcher",
            "dispatch_orbit",
            "dispatch_trinity",
            "supervisor_decide",
            "generate_plan",
            "approval_gate",
            END,
        ],
    )
    
    # Context fetcher cycles back to supervisor
    workflow.add_edge("context_fetcher", "supervisor_decide")
    
    # Subagent dispatch → merge → supervisor (cyclic)
    # For single dispatch, go directly to merge then back to supervisor
    workflow.add_edge("dispatch_orbit", "merge_results")
    workflow.add_edge("dispatch_trinity", "merge_results")
    workflow.add_edge("merge_results", "supervisor_decide")

    # ------------------------------------------------------------------ #
    # Edges — Plan and Build Phase                                         #
    # ------------------------------------------------------------------ #
    
    # After plan generation, decide: analyse only or full build
    workflow.add_conditional_edges(
        "generate_plan",
        should_build,
        {"done": END, "build": "approval_gate"},
    )

    # Edges — build phase
    workflow.add_conditional_edges(
        "approval_gate",
        is_approved,
        {"abort": END, "proceed": "generate_code_changes"},
    )
    
    # Codegen routing with compensation (Item 10)
    def codegen_router(state: AnalysisState) -> str:
        """Route based on codegen success/failure."""
        error = state.get("error")
        succeeded = state.get("succeeded_repos", [])
        
        if error and succeeded:
            # Partial failure: some repos succeeded, need compensation
            return "compensate"
        elif error:
            # Total failure: nothing to roll back
            return "abort"
        else:
            # Success: proceed to git operations
            return "continue"
    
    workflow.add_conditional_edges(
        "generate_code_changes",
        codegen_router,
        {
            "continue": "git_operations",
            "compensate": "compensate",
            "abort": END,
        },
    )
    
    # Compensation leads to END
    workflow.add_edge("compensate", END)
    
    workflow.add_edge("git_operations", "create_pull_requests")
    workflow.add_edge("create_pull_requests", END)

    # Compile with optional checkpointer (Item 9)
    if checkpointer:
        logger.info("Compiling graph with persistent checkpointing enabled")
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return slug.strip("-")[:50]


class Orchestrator:
    """High-level orchestrator interface with persistent checkpointing (Item 9)."""

    def __init__(self, enable_checkpointing: bool = True):
        """
        Initialize the orchestrator.
        
        Parameters
        ----------
        enable_checkpointing : bool
            If True, enables persistent checkpointing to SQLite.
            Set to False for testing or when checkpointing isn't needed.
        """
        self.checkpointer = _get_checkpointer() if enable_checkpointing else None
        self.graph = create_orchestrator(checkpointer=self.checkpointer)

    def _base_state(self, feature_description: str) -> AnalysisState:
        # Generate thread_id for checkpointing (Item 9)
        feature_slug = _slugify(feature_description)
        thread_id = f"kb-{feature_slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        return {
            "feature_description": feature_description,
            "search_results": [],
            "cross_repo_links": [],
            "orbit_analysis": None,
            "trinity_analysis": None,
            "impact_map": {},
            "pr_draft": {},
            "messages": [],
            "needs_clarification": False,
            "clarification_question": "",
            "current_step": "",
            "error": None,
            # Feature Builder defaults
            "build_mode": False,
            "dry_run": True,
            "approved": False,
            "feature_slug": feature_slug,
            "branch_names": {},
            "code_gen_results": [],
            "commit_shas": {},
            "pr_results": [],
            "plan_text": "",
            # Structured plan (Item 7)
            "implementation_plan": None,
            # Supervisor fields (Item 3)
            "iteration_count": 0,
            "max_orchestrator_iterations": 8,
            # Rollback tracking (Item 10)
            "succeeded_repos": [],
            "compensation_results": [],
            # Checkpointing (Item 9)
            "thread_id": thread_id,
        }

    def analyze(self, feature_description: str) -> dict:
        """
        Analyse a feature request (read-only, no code generation).

        Returns analysis results including impact map and PR draft.
        The thread_id in the result can be used to resume later.
        
        Raises
        ------
        SecurityViolation
            If the query is blocked by IntentGuard.
        RateLimitExceeded
            If rate limits are exceeded.
        """
        # Layer 1: Intent Guard - check query before processing
        intent_guard = get_intent_guard()
        intent_guard.guard(feature_description)  # Raises SecurityViolation if blocked
        
        # Layer 5: Rate limiter - check session limits
        rate_limiter = get_rate_limiter()
        rate_limiter.check_llm_call()  # Will be incremented as LLM calls are made
        
        initial_state = self._base_state(feature_description)
        thread_id = initial_state["thread_id"]
        
        # Pass thread_id config for checkpointing (Item 9)
        config = {"configurable": {"thread_id": thread_id}} if self.checkpointer else None
        final_state = self.graph.invoke(initial_state, config=config)
        
        result = self._format_result(final_state)
        result["thread_id"] = thread_id  # Include for resume capability
        return result

    def build(
        self,
        feature_description: str,
        approved: bool = False,
        dry_run: bool = True,
    ) -> dict:
        """
        Full Feature Builder run: analyse → (if approved) generate code
        → commit to feature branch → open GitHub PR.

        Parameters
        ----------
        feature_description: str
            Natural language feature description.
        approved: bool
            Set to True to proceed with code generation.
            False (default) stops at the approval gate.
        dry_run: bool
            When True (default), generate and diff but don't write to disk
            or push to GitHub.  Set to False to actually commit and push.
        
        Raises
        ------
        SecurityViolation
            If the query is blocked by IntentGuard.
        RateLimitExceeded
            If rate limits are exceeded.
        """
        # Layer 1: Intent Guard - check query before processing
        intent_guard = get_intent_guard()
        intent_guard.guard(feature_description)  # Raises SecurityViolation if blocked
        
        # Layer 5: Rate limiter - check session limits
        rate_limiter = get_rate_limiter()
        rate_limiter.check_llm_call()  # Will be incremented as LLM calls are made
        rate_limiter.check_build_rate()  # Check hourly build limit
        
        initial_state = self._base_state(feature_description)
        initial_state["build_mode"] = True
        initial_state["approved"] = approved
        initial_state["dry_run"] = dry_run
        
        thread_id = initial_state["thread_id"]

        # Pass thread_id config for checkpointing (Item 9)
        config = {"configurable": {"thread_id": thread_id}} if self.checkpointer else None
        final_state = self.graph.invoke(initial_state, config=config)
        result = self._format_result(final_state)

        # Record the build for rate limiting (after successful start)
        rate_limiter.record_build()

        # Add builder-specific fields
        result["build_mode"] = True
        result["dry_run"] = dry_run
        result["thread_id"] = thread_id  # Include for resume capability
        result["branch_names"] = final_state.get("branch_names", {})
        result["commit_shas"] = final_state.get("commit_shas", {})
        result["pr_results"] = final_state.get("pr_results", [])
        result["code_gen_results"] = final_state.get("code_gen_results", [])

        # Attach change tracker summary for the UI diff viewer
        try:
            from codegen.change_tracker import get_tracker
            result["change_summary"] = get_tracker().get_summary()
        except Exception:
            result["change_summary"] = {}

        return result

    def resume(self, thread_id: str, approved: bool = True, dry_run: bool = True) -> dict:
        """
        Resume a workflow from its last checkpoint (Item 9).
        
        This is useful for resuming after the approval gate without re-running
        the analysis phase. The workflow continues from where it left off.
        
        Parameters
        ----------
        thread_id : str
            The thread_id returned from a previous analyze() or build() call.
        approved : bool
            Set to True to proceed past the approval gate.
        dry_run : bool
            When True (default), generate but don't write to disk or push.
            
        Returns
        -------
        dict
            Same format as build() result.
            
        Raises
        ------
        ValueError
            If checkpointing is not enabled or thread_id is not found.
        """
        if not self.checkpointer:
            raise ValueError("Cannot resume: checkpointing is not enabled")
        
        logger.info(f"Resuming workflow from checkpoint: {thread_id}")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get the current state from checkpoint
        try:
            checkpoint_state = self.graph.get_state(config)
            if not checkpoint_state or not checkpoint_state.values:
                raise ValueError(f"No checkpoint found for thread_id: {thread_id}")
        except Exception as e:
            raise ValueError(f"Failed to retrieve checkpoint: {e}")
        
        # Update the state with new approval/dry_run values
        # This allows resuming past the approval gate
        update_state = {
            "approved": approved,
            "dry_run": dry_run,
        }
        
        # Resume execution
        final_state = self.graph.invoke(None, config=config)
        result = self._format_result(final_state)
        
        # Add builder-specific fields
        result["build_mode"] = final_state.get("build_mode", False)
        result["dry_run"] = dry_run
        result["thread_id"] = thread_id
        result["resumed"] = True
        result["branch_names"] = final_state.get("branch_names", {})
        result["commit_shas"] = final_state.get("commit_shas", {})
        result["pr_results"] = final_state.get("pr_results", [])
        result["code_gen_results"] = final_state.get("code_gen_results", [])
        
        try:
            from codegen.change_tracker import get_tracker
            result["change_summary"] = get_tracker().get_summary()
        except Exception:
            result["change_summary"] = {}
        
        return result

    @staticmethod
    def _format_result(final_state: AnalysisState) -> dict:
        return {
            "needs_clarification": final_state.get("needs_clarification", False),
            "clarification_question": final_state.get("clarification_question", ""),
            "impact_map": final_state.get("impact_map", {}),
            "pr_draft": final_state.get("pr_draft", {}),
            "orbit_analysis": final_state.get("orbit_analysis"),
            "trinity_analysis": final_state.get("trinity_analysis"),
            "error": final_state.get("error"),
            "implementation_plan": final_state.get("implementation_plan"),
            "plan_text": final_state.get("plan_text", ""),
        }

    def chat_stream(
        self,
        message: str,
        history: list[BaseMessage],
    ):
        """
        Stream a chat response with tool usage.
        
        This method builds a message list with the chat system prompt,
        conversation history, and user message, then streams the LLM
        response with tool execution inline.
        
        Parameters
        ----------
        message : str
            The user's current message.
        history : list[BaseMessage]
            LangChain messages representing conversation history.
            
        Yields
        ------
        str
            Text chunks as they arrive from the LLM, including tool
            status messages.
        """
        from langchain_core.messages import AIMessageChunk
        from .prompts import CHAT_SYSTEM_PROMPT

        # Initialize LLM with tools
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
            streaming=True,
        )
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        # Build history summary for the system prompt
        history_summary = (
            "Conversation has context from prior messages." if history
            else "This is the start of a new conversation."
        )

        system_prompt = CHAT_SYSTEM_PROMPT.format(history_summary=history_summary)
        messages = [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=message),
        ]

        # Run up to max_iterations tool-call rounds
        max_iterations = 8
        for _ in range(max_iterations):
            # Accumulate all streaming chunks into one AIMessageChunk so that
            # tool call args and IDs are fully merged (fixes null-id / missing-arg bugs).
            accumulated: Optional[AIMessageChunk] = None
            for chunk in llm_with_tools.stream(messages):
                # Yield text tokens to the UI as they arrive
                if chunk.content:
                    yield chunk.content
                accumulated = chunk if accumulated is None else accumulated + chunk

            if accumulated is None:
                break

            # No tool calls — we are done
            tool_calls = getattr(accumulated, "tool_calls", [])
            if not tool_calls:
                break

            # Build a single well-formed AIMessage from the merged chunk
            ai_msg = AIMessage(
                content=accumulated.content or "",
                tool_calls=tool_calls,
            )
            messages.append(ai_msg)

            # Execute each tool call and append ToolMessages
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id") or ""

                yield f"\n\n> Searching {tool_name}...\n"

                tool_fn = next((t for t in ALL_TOOLS if t.name == tool_name), None)
                if tool_fn:
                    try:
                        tool_result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Error: {str(e)}"
                else:
                    tool_result = f"Tool '{tool_name}' not found"

                result_preview = str(tool_result)
                if len(result_preview) > 500:
                    result_preview = result_preview[:500] + "..."
                yield f"> {result_preview}\n\n"

                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                )
