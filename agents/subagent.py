"""
Per-repository subagents for specialized analysis.

Upgraded with:
- Structured output using llm.with_structured_output() instead of regex parsing
- Rich init-time prompts via SubagentInitializer (CLAUDE.md-style)
- Pattern sampling for codegen via PatternSampler injection
- Verify → Lint → Fix loop for codegen mode
"""

import logging
from typing import Optional, Any, Literal
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from pydantic import BaseModel, Field

from config import settings
from .tools import ALL_TOOLS, CODEGEN_TOOLS, VERIFY_TOOLS
from .prompts import ORBIT_SUBAGENT_PROMPT, TRINITY_SUBAGENT_PROMPT
from .rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Output Models (Item 2)
# ---------------------------------------------------------------------------

class APIContract(BaseModel):
    """Represents an affected API contract."""
    method: str = Field(description="HTTP method (GET, POST, PUT, DELETE, etc.)")
    path: str = Field(description="API endpoint path (e.g., /api/v1/users)")
    description: str = Field(default="", description="Brief description of the contract change")


class SubagentStructuredOutput(BaseModel):
    """
    Structured output schema for subagent final synthesis.

    This replaces the old regex-based _parse_response with a typed,
    validated Pydantic model that the LLM returns directly via
    with_structured_output().
    """
    model_config = {"arbitrary_types_allowed": True}
    reasoning: str = Field(
        description="Chain-of-thought reasoning explaining the analysis process and conclusions"
    )
    impacted_files: list[str] = Field(
        default_factory=list,
        description="List of file paths that will be impacted by this feature/change"
    )
    impacted_functions: list[str] = Field(
        default_factory=list,
        description="List of function/method names that will need changes"
    )
    suggested_changes: list[str] = Field(
        default_factory=list,
        description="Specific actionable changes to implement (one per item)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the analysis: high (clear path), medium (some unknowns), low (significant uncertainty)"
    )
    api_contracts: list[APIContract] = Field(
        default_factory=list,
        description="API contracts that will be affected or need to be created"
    )
    testing_notes: str = Field(
        default="",
        description="Recommendations for testing the changes"
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Potential risks or concerns with the proposed changes"
    )


class SubagentResponse(BaseModel):
    """Structured output from a subagent (public interface)."""
    impacted_files: list[str] = Field(default_factory=list, description="List of impacted file paths")
    impacted_functions: list[str] = Field(default_factory=list, description="List of impacted function/method names")
    suggested_changes: list[str] = Field(default_factory=list, description="List of suggested code changes")
    confidence: str = Field(default="medium", description="Confidence level: high, medium, low")
    analysis: str = Field(default="", description="Detailed analysis text")
    api_contracts: list[dict] = Field(default_factory=list, description="Affected API contracts")
    reasoning: str = Field(default="", description="Chain-of-thought reasoning from the subagent")
    testing_notes: str = Field(default="", description="Testing recommendations")
    risks: list[str] = Field(default_factory=list, description="Identified risks")


# ---------------------------------------------------------------------------
# SubagentInitializer (Item 4) - Rich CLAUDE.md-style system prompts
# ---------------------------------------------------------------------------

class SubagentInitializer:
    """
    Builds rich, context-aware system prompts for subagents by sampling
    real files from the repository at init time (Item 4).
    
    Creates CLAUDE.md-style prompts with:
    - Codebase memory (real file excerpts)
    - Conventions extracted from actual code
    - Tool strategy guidance
    - Output contract (JSON schema)
    - Quality gates
    """
    
    # Representative file patterns for each repo
    SAMPLE_PATTERNS = {
        "orbit": [
            ("**/components/**/*.tsx", "React component pattern"),
            ("**/hooks/**/*.ts", "Custom hook pattern"),
            ("**/app/**/page.tsx", "App Router page pattern"),
            ("**/api/**/*.ts", "API client pattern"),
        ],
        "trinity": [
            ("**/*Controller.java", "Controller pattern"),
            ("**/*Service.java", "Service pattern"),
            ("**/*Repository.java", "Repository pattern"),
            ("**/dto/**/*.java", "DTO pattern"),
        ],
    }
    
    @classmethod
    def build_system_prompt(cls, repo: str, repo_path: Path) -> str:
        """
        Build a rich system prompt by sampling real files from the repo.
        
        Args:
            repo: Repository name ("orbit" or "trinity")
            repo_path: Path to the repository root
            
        Returns:
            A comprehensive system prompt with real codebase context
        """
        # Start with base prompt
        if repo == "orbit":
            base_prompt = ORBIT_SUBAGENT_PROMPT
            identity = "You are an expert Orbit frontend engineer specializing in React, TypeScript, and Next.js App Router."
        elif repo == "trinity":
            base_prompt = TRINITY_SUBAGENT_PROMPT
            identity = "You are an expert Trinity-v2 backend engineer specializing in Spring Boot, Java 21, and REST APIs."
        else:
            base_prompt = f"You are an expert in the {repo} codebase."
            identity = base_prompt
        
        # Sample real files
        codebase_memory = cls._sample_codebase(repo, repo_path)
        
        # Load conventions if present
        conventions = cls._load_conventions(repo_path)
        
        # Build the full prompt
        sections = [
            f"# Identity\n\n{identity}",
            f"# Base Knowledge\n\n{base_prompt}",
        ]
        
        if codebase_memory:
            sections.append(f"# Codebase Memory\n\nReal examples from this repository:\n\n{codebase_memory}")
        
        if conventions:
            sections.append(f"# Conventions\n\n{conventions}")
        
        # Tool strategy
        tool_strategy = """# Tool Strategy

When analyzing:
1. **Start broad** — use search_code with general terms first
2. **Follow the graph** — use get_node_graph to trace dependencies
3. **Read before suggesting** — always use get_file_content before proposing changes
4. **Trace cross-repo** — use cross_repo_trace for API endpoints

When generating code (codegen mode):
1. **Read first** — always read the target file before editing
2. **Match patterns** — follow the style of similar files in this repo
3. **One file at a time** — make one edit_file/create_file call per file
4. **Verify after** — use verify_file and lint_file to check your work
"""
        sections.append(tool_strategy)
        
        # Output contract
        output_contract = """# Output Contract

Your final synthesis must include:
- **reasoning**: Step-by-step explanation of your analysis
- **impacted_files**: Full relative paths to affected files
- **impacted_functions**: Names of functions/methods to change
- **suggested_changes**: Specific, actionable change descriptions
- **confidence**: high (clear path) / medium (some unknowns) / low (significant uncertainty)
- **api_contracts**: Any API endpoints affected (method, path, description)
- **testing_notes**: How to test the changes
- **risks**: Potential issues or concerns
"""
        sections.append(output_contract)
        
        # Quality gates
        quality_gates = f"""# Quality Gates ({repo})

Before finalizing analysis, verify:
- [ ] All mentioned files actually exist (you read them)
- [ ] Function/method names are correct (from actual code)
- [ ] API contracts match between frontend and backend
- [ ] No breaking changes to existing interfaces
- [ ] Testing strategy is realistic
"""
        sections.append(quality_gates)
        
        return "\n\n---\n\n".join(sections)
    
    @classmethod
    def _sample_codebase(cls, repo: str, repo_path: Path) -> str:
        """Sample representative files from the codebase."""
        import glob as glob_module
        
        patterns = cls.SAMPLE_PATTERNS.get(repo, [])
        samples = []
        
        for pattern, description in patterns:
            # Find matching files
            full_pattern = str(repo_path / pattern)
            matches = glob_module.glob(full_pattern, recursive=True)
            
            if matches:
                # Take the first match
                file_path = Path(matches[0])
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    # Truncate to first 100 lines
                    lines = content.split("\n")[:100]
                    truncated = "\n".join(lines)
                    if len(lines) == 100:
                        truncated += "\n... (truncated)"
                    
                    rel_path = file_path.relative_to(repo_path)
                    samples.append(f"## {description}\n\nFile: `{rel_path}`\n\n```\n{truncated}\n```")
                except Exception as e:
                    logger.debug(f"Could not read sample file {file_path}: {e}")
        
        return "\n\n".join(samples) if samples else ""
    
    @classmethod
    def _load_conventions(cls, repo_path: Path) -> str:
        """Load conventions from CONVENTIONS.md, CLAUDE.md, or README.md."""
        convention_files = ["CONVENTIONS.md", "CLAUDE.md", ".claude/instructions.md"]
        
        for filename in convention_files:
            conv_path = repo_path / filename
            if conv_path.exists():
                try:
                    content = conv_path.read_text(encoding="utf-8", errors="replace")
                    # Truncate if too long
                    if len(content) > 5000:
                        content = content[:5000] + "\n\n... (truncated)"
                    return f"From `{filename}`:\n\n{content}"
                except Exception as e:
                    logger.debug(f"Could not read conventions from {conv_path}: {e}")
        
        # Fallback: try to extract from README
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8", errors="replace")
                # Look for a conventions section
                if "convention" in content.lower() or "style" in content.lower():
                    lines = content.split("\n")[:200]  # First 200 lines
                    return f"From README.md (excerpt):\n\n" + "\n".join(lines)
            except Exception:
                pass
        
        return ""


class Subagent:
    """
    Per-repository subagent for specialized analysis or code generation.

    Parameters
    ----------
    repo : str
        Repository name: "orbit" or "trinity".
    repo_path : Path | None
        Path to the repository root. Used by SubagentInitializer for rich prompts.
    system_prompt : str | None
        Override the default system prompt.  When None the repo-appropriate
        default (ORBIT/TRINITY_SUBAGENT_PROMPT) is used.
    tools : list | None
        Override the tool set.  When None ALL_TOOLS is used.
        Pass CODEGEN_TOOLS + ALL_TOOLS for code-generation subagents.
    max_iterations : int
        Maximum tool-call iterations per analyze() call.
    pattern_sampler : PatternSampler | None
        Optional pattern sampler for codegen mode (Item 8).
    is_codegen : bool
        Whether this subagent is in codegen mode (enables verify loop).
    """

    def __init__(
        self,
        repo: str,
        repo_path: Optional[Path] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        max_iterations: int = 10,
        pattern_sampler: Optional[Any] = None,  # PatternSampler type hint avoided for now
        is_codegen: bool = False,
    ):
        self.repo = repo
        self.repo_path = repo_path
        self.max_iterations = max_iterations
        self.pattern_sampler = pattern_sampler
        self.is_codegen = is_codegen

        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

        # Tool set
        _tools = tools if tools is not None else ALL_TOOLS
        self.tools = _tools
        self.llm_with_tools = self.llm.bind_tools(_tools)
        
        # LLM with structured output for final synthesis (Item 2)
        self.llm_structured = self.llm.with_structured_output(SubagentStructuredOutput)

        # System prompt - use SubagentInitializer if repo_path provided (Item 4)
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif repo_path:
            self.system_prompt = SubagentInitializer.build_system_prompt(repo, repo_path)
        elif repo == "orbit":
            self.system_prompt = ORBIT_SUBAGENT_PROMPT
        elif repo == "trinity":
            self.system_prompt = TRINITY_SUBAGENT_PROMPT
        else:
            self.system_prompt = f"You are an expert in the {repo} codebase."
    
    def analyze(
        self,
        query: str,
        context: Optional[dict] = None,
        max_iterations: Optional[int] = None,
    ) -> SubagentResponse:
        """
        Analyze a query within the repository context.
        
        Two-phase approach:
        1. Tool-calling loop - explore codebase with search_code, get_node_graph, etc.
        2. Final synthesis - use llm.with_structured_output() for typed, validated output
        
        For codegen mode (is_codegen=True), adds a third phase:
        3. Verify loop - re-read, lint, and fix written files (Item 6)
        
        Args:
            query: The analysis query
            context: Optional additional context
            max_iterations: Maximum tool call iterations
            
        Returns:
            SubagentResponse with analysis results
        """
        messages: list[Any] = [
            SystemMessage(content=self.system_prompt),
        ]
        
        # Inject pattern samples for codegen mode (Item 8)
        if self.is_codegen and self.pattern_sampler and context:
            planned_files = context.get("planned_files", [])
            if planned_files:
                samples = self.pattern_sampler.get_samples_for_files(self.repo_path, planned_files)
                if samples:
                    messages.append(HumanMessage(content=f"""
Style references from this codebase — match these patterns exactly:

{samples}

---
"""))
        
        messages.append(HumanMessage(content=self._build_prompt(query, context)))

        _max = max_iterations if max_iterations is not None else self.max_iterations
        iterations = 0
        
        # Get rate limiter for tracking
        rate_limiter = get_rate_limiter()

        # Phase 1: Tool-calling loop
        while iterations < _max:
            # Check rate limit before LLM call
            rate_limiter.check_llm_call()
            
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Track LLM call and tokens
            rate_limiter.increment_llm_calls()
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                total_tokens = getattr(response.usage_metadata, 'total_tokens', 0)
                if total_tokens:
                    rate_limiter.add_tokens(total_tokens)
            
            # Check for tool calls
            tool_calls = getattr(response, 'tool_calls', None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    
                    # Force repo filter for this subagent
                    if "repo" in tool_args or tool_name in ["search_code", "get_node_graph", "find_callers"]:
                        tool_args["repo"] = self.repo
                    
                    # Execute tool (search in the subagent's own tool set)
                    tool_fn = next((t for t in self.tools if t.name == tool_name), None)
                    if tool_fn:
                        try:
                            result = tool_fn.invoke(tool_args)
                            messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call_id,
                            ))
                        except Exception as e:
                            messages.append(ToolMessage(
                                content=f"Tool '{tool_name}' failed: {str(e)}",
                                tool_call_id=tool_call_id,
                            ))
                    else:
                        messages.append(ToolMessage(
                            content=f"Unknown tool: {tool_name}",
                            tool_call_id=tool_call_id,
                        ))

                # Count LLM rounds, not individual tool calls within a round
                iterations += 1
            else:
                # No more tool calls, move to synthesis
                break
        
        # Phase 2 (codegen only): Verify → Lint → Fix loop (Item 6)
        if self.is_codegen:
            messages = self._run_verify_loop(messages, max_verify_iterations=5)
        
        # Phase 3: Final synthesis with structured output (Item 2)
        return self._synthesize_response(messages)
    
    def _run_verify_loop(self, messages: list[Any], max_verify_iterations: int = 5) -> list[Any]:
        """
        Run the verify → lint → fix loop for codegen mode (Item 6).
        
        Injects a PHASE: VERIFY message and runs a second tool loop where
        the agent is instructed to re-read, lint, and fix any written files.
        """
        # Check if any codegen tools (edit_file, create_file) were called
        codegen_tool_names = {"edit_file", "create_file"}
        has_codegen_calls = any(
            getattr(msg, 'tool_calls', None) and 
            any(tc.get("name") in codegen_tool_names for tc in msg.tool_calls)
            for msg in messages
            if hasattr(msg, 'tool_calls')
        )
        
        if not has_codegen_calls:
            return messages
        
        # Inject PHASE: VERIFY prompt
        verify_prompt = """
PHASE: VERIFY

You have made code changes. Now verify each file you created or edited:

1. Use `verify_file` to re-read each written file from disk (not from memory)
2. Use `lint_file` to check for syntax/style errors
3. If errors are found, use `edit_file` to fix them

Do NOT skip this phase. Verify ALL files you touched.
When all files pass verification, respond with "VERIFICATION COMPLETE".
"""
        messages.append(HumanMessage(content=verify_prompt))
        
        # Get rate limiter for tracking verify loop calls
        rate_limiter = get_rate_limiter()
        
        for _ in range(max_verify_iterations):
            # Check rate limit before LLM call
            rate_limiter.check_llm_call()
            
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Track LLM call
            rate_limiter.increment_llm_calls()
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                total_tokens = getattr(response.usage_metadata, 'total_tokens', 0)
                if total_tokens:
                    rate_limiter.add_tokens(total_tokens)
            
            tool_calls = getattr(response, 'tool_calls', None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    
                    if "repo" in tool_args or tool_name in ["verify_file", "lint_file"]:
                        tool_args["repo"] = self.repo
                    
                    tool_fn = next((t for t in self.tools if t.name == tool_name), None)
                    if tool_fn:
                        try:
                            result = tool_fn.invoke(tool_args)
                            messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call_id,
                            ))
                        except Exception as e:
                            messages.append(ToolMessage(
                                content=f"Tool '{tool_name}' failed: {str(e)}",
                                tool_call_id=tool_call_id,
                            ))
                    else:
                        messages.append(ToolMessage(
                            content=f"Unknown tool: {tool_name}",
                            tool_call_id=tool_call_id,
                        ))
            else:
                # No more tool calls - verification complete
                break
        
        return messages
    
    def _synthesize_response(self, messages: list[Any]) -> SubagentResponse:
        """
        Use structured output to synthesize the final response (Item 2).
        
        This replaces the old regex-based _parse_response with a clean
        LLM call using with_structured_output().
        """
        # Build synthesis prompt with conversation context
        synthesis_prompt = """
Based on your analysis above, provide a structured summary.

Include:
- Your reasoning process (chain-of-thought)
- All impacted files (full relative paths)
- All impacted functions/methods  
- Specific suggested changes (actionable items)
- Your confidence level (high/medium/low)
- Any API contracts affected
- Testing recommendations
- Identified risks

Be thorough and specific. Include file paths and function names you discovered.
"""
        
        synthesis_messages = messages + [HumanMessage(content=synthesis_prompt)]
        
        try:
            # Track synthesis LLM call
            rate_limiter = get_rate_limiter()
            rate_limiter.check_llm_call()
            
            structured_output: SubagentStructuredOutput = self.llm_structured.invoke(synthesis_messages)
            
            # Track the call
            rate_limiter.increment_llm_calls()
            # Note: structured output may not have usage_metadata, so we skip token tracking here
            
            # Convert to SubagentResponse
            return SubagentResponse(
                impacted_files=structured_output.impacted_files,
                impacted_functions=structured_output.impacted_functions,
                suggested_changes=structured_output.suggested_changes,
                confidence=structured_output.confidence,
                analysis=structured_output.reasoning,  # Use reasoning as analysis
                api_contracts=[
                    {"method": c.method, "path": c.path, "description": c.description}
                    for c in structured_output.api_contracts
                ],
                reasoning=structured_output.reasoning,
                testing_notes=structured_output.testing_notes,
                risks=structured_output.risks,
            )
        except Exception as e:
            logger.error(f"Structured output synthesis failed: {e}")
            # Fallback: extract from last message content
            last_content = ""
            for msg in reversed(messages):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    last_content = msg.content
                    break
            
            return SubagentResponse(
                analysis=last_content,
                confidence="low",
                reasoning=f"Structured synthesis failed: {e}. Raw content preserved.",
            )
    
    def _build_prompt(self, query: str, context: Optional[dict]) -> str:
        """Build the analysis or implementation prompt depending on mode."""
        if self.is_codegen:
            # Extract the explicit file list from context so the LLM cannot
            # skip writes by claiming the feature is "already implemented".
            required_files_section = ""
            if context and context.get("required_files"):
                files = context["required_files"]
                file_lines = "\n".join(f"  - {f}" for f in files)
                required_files_section = f"""
REQUIRED FILES — you MUST call edit_file or create_file for EVERY one of these, \
even if the file already exists on disk (it may contain stale content from a \
previous failed run):
{file_lines}
"""

            prompt = f"""Implement the following in the {self.repo} repository:

{query}
{required_files_section}
Steps you MUST follow:
1. Use get_file_content or search_code to read every file you plan to change BEFORE editing it.
2. Call edit_file or create_file for EACH file that needs changes — one tool call per file.
3. IMPORTANT: Do NOT skip a file because it appears to already contain the right code. \
Always call edit_file/create_file to record the change in the tracker, even for minor adjustments.
4. After all edits, respond with:

IMPLEMENTATION COMPLETE
Files modified: <comma-separated list>
Summary: <1-2 sentence summary>
Testing: <key test scenarios>

Focus only on the {self.repo} repository."""
        else:
            prompt = f"""Analyze the following in the {self.repo} repository:

{query}

Please:
1. Use search_code to find relevant code
2. Use get_node_graph to understand dependencies
3. Identify all impacted files and functions
4. Provide confidence level (high/medium/low)
5. Suggest specific changes needed

Focus only on the {self.repo} repository."""

        if context:
            # Exclude internal keys that aren't useful as raw context for the LLM
            extra = {k: v for k, v in context.items() if k not in ("required_files",)}
            if extra:
                prompt += f"\n\nAdditional context:\n{extra}"

        return prompt


def create_orbit_subagent(repo_path: Optional[Path] = None) -> Subagent:
    """Create subagent for Orbit repository."""
    from config import settings as cfg
    path = repo_path or cfg.orbit_repo_path
    return Subagent("orbit", repo_path=Path(path) if path else None)


def create_trinity_subagent(repo_path: Optional[Path] = None) -> Subagent:
    """Create subagent for Trinity-v2 repository."""
    from config import settings as cfg
    path = repo_path or cfg.trinity_repo_path
    return Subagent("trinity", repo_path=Path(path) if path else None)
