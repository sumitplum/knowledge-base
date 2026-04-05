"""
Chat router for intent classification and per-intent handling.

This is the brain of the chat interface - it classifies user intent
and dispatches to the appropriate handler.
"""

import logging
from typing import Optional, Literal, Generator, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from .conversation_memory import ConversationMemory, ConversationTurn
from .prompts import INTENT_CLASSIFICATION_PROMPT, PLAN_IMPROVEMENT_PROMPT
from .orchestrator import Orchestrator, ImplementationPlan

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Structured output for intent classification."""
    
    intent: Literal["question", "plan_request", "improve_plan", "build_request", "clarification"] = Field(
        description="The classified intent category"
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )
    extracted_feature: Optional[str] = Field(
        default=None,
        description="Feature description extracted from the message (for plan/build intents)"
    )


class ChatRouter:
    """
    Intent classification and per-intent handler dispatch.
    
    The router classifies incoming messages and routes them to the
    appropriate handler based on intent.
    """
    
    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        """
        Initialize the chat router.
        
        Parameters
        ----------
        orchestrator : Orchestrator, optional
            The orchestrator instance to use for analysis/build operations.
            If not provided, one will be created on demand.
        """
        self._orchestrator = orchestrator
        self._llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast model for classification
            api_key=settings.openai_api_key,
            temperature=0,
        )
    
    @property
    def orchestrator(self) -> Orchestrator:
        """Lazy-load the orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(enable_checkpointing=True)
        return self._orchestrator
    
    def classify(
        self,
        message: str,
        memory: ConversationMemory,
    ) -> IntentClassification:
        """
        Classify the user's intent based on message and conversation history.
        
        Parameters
        ----------
        message : str
            The user's current message.
        memory : ConversationMemory
            The conversation memory containing history.
            
        Returns
        -------
        IntentClassification
            The classified intent with reasoning.
        """
        # Build history string from recent turns
        recent_turns = memory.get_last_n_turns(6)
        history_lines = []
        for turn in recent_turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            content_preview = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            history_lines.append(f"{prefix}: {content_preview}")
            if turn.has_plan():
                history_lines.append("  [Contains implementation plan]")
        
        history_str = "\n".join(history_lines) if history_lines else "No prior conversation."
        
        # Build the classification prompt
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            history=history_str,
            message=message,
        )
        
        # Use structured output for classification
        llm_structured = self._llm.with_structured_output(IntentClassification)
        
        try:
            classification: IntentClassification = llm_structured.invoke([
                SystemMessage(content="You are an intent classifier for a code intelligence chat interface."),
                HumanMessage(content=prompt),
            ])
            
            return classification
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Default to question on error
            return IntentClassification(
                intent="question",
                reasoning=f"Classification failed: {str(e)}, defaulting to question.",
                extracted_feature=None,
            )
    
    def handle(
        self,
        classification: IntentClassification,
        message: str,
        memory: ConversationMemory,
        dry_run: bool = True,
    ) -> Generator[Union[str, dict], None, None]:
        """
        Handle the classified intent and yield streaming response.
        
        Parameters
        ----------
        classification : IntentClassification
            The classified intent.
        message : str
            The user's original message.
        memory : ConversationMemory
            The conversation memory.
        dry_run : bool
            For build operations, whether to actually write files.
            
        Yields
        ------
        str
            Text chunks as they're generated.
        dict
            Structured data as the final item (plan, PR results, etc.)
        """
        intent = classification.intent
        
        if intent == "question":
            return self._handle_question(message, memory)
        elif intent == "plan_request":
            return self._handle_plan_request(message, classification, memory)
        elif intent == "improve_plan":
            return self._handle_improve_plan(message, memory)
        elif intent == "build_request":
            return self._handle_build_request(message, classification, memory, dry_run)
        elif intent == "clarification":
            return self._handle_clarification(classification)
        else:
            return self._handle_clarification(classification)
    
    def _handle_question(
        self,
        message: str,
        memory: ConversationMemory,
    ) -> Generator[Union[str, dict], None, None]:
        """Handle a question about the codebase."""
        history = memory.to_langchain_messages()
        
        # Use the orchestrator's chat_stream for Q&A
        full_response = ""
        for chunk in self.orchestrator.chat_stream(message, history):
            full_response += chunk
            yield chunk
        
        yield {"type": "question", "response": full_response}
    
    def _handle_plan_request(
        self,
        message: str,
        classification: IntentClassification,
        memory: ConversationMemory,
    ) -> Generator[Union[str, dict], None, None]:
        """Handle a request to plan a feature."""
        feature_description = classification.extracted_feature or message
        
        yield f"> Analyzing feature request: {feature_description[:100]}...\n\n"
        
        # Run the orchestrator analysis
        try:
            result = self.orchestrator.analyze(feature_description)
            
            # Format the plan for display
            plan = result.get("implementation_plan")
            plan_text = result.get("plan_text", "")
            
            if plan:
                yield f"## Implementation Plan: {plan.title}\n\n"
                yield f"{plan.summary}\n\n"
                yield "### Steps\n\n"
                
                for step in plan.steps:
                    deps = f" (depends on: {', '.join(str(d) for d in step.depends_on)})" if step.depends_on else ""
                    yield f"{step.step_number}. **[{step.repo.upper()}]** {step.description}{deps}\n"
                    if step.files:
                        for f in step.files[:3]:
                            yield f"   - `{f}`\n"
                    yield "\n"
                
                if plan.risks:
                    yield "### Risks\n\n"
                    for risk in plan.risks:
                        yield f"- {risk}\n"
                    yield "\n"
                
                if plan.testing_strategy:
                    yield f"### Testing Strategy\n\n{plan.testing_strategy}\n\n"
                
                if plan.estimated_effort:
                    yield f"**Estimated Effort:** {plan.estimated_effort}\n"
                
            elif plan_text:
                yield plan_text
            else:
                yield "Unable to generate a detailed plan. The analysis may not have found enough context.\n"
                if result.get("error"):
                    yield f"\nError: {result['error']}\n"
            
            yield {
                "type": "plan_request",
                "implementation_plan": plan,
                "plan_text": plan_text,
                "thread_id": result.get("thread_id"),
                "orbit_analysis": result.get("orbit_analysis"),
                "trinity_analysis": result.get("trinity_analysis"),
            }
            
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            yield f"\nError generating plan: {str(e)}\n"
            yield {"type": "plan_request", "error": str(e)}
    
    def _handle_improve_plan(
        self,
        message: str,
        memory: ConversationMemory,
    ) -> Generator[Union[str, dict], None, None]:
        """Handle a request to improve an existing plan."""
        last_plan_data = memory.get_last_plan()
        
        if not last_plan_data:
            yield "I don't see a previous plan to improve. Please describe the feature you want to plan first.\n"
            yield {"type": "improve_plan", "error": "No prior plan found"}
            return
        
        plan = last_plan_data.get("implementation_plan")
        if not plan:
            yield "I couldn't find the plan details. Please describe what changes you want.\n"
            yield {"type": "improve_plan", "error": "Plan data missing"}
            return
        
        yield "> Improving the plan based on your feedback...\n\n"
        
        # Format the original plan for the LLM
        if hasattr(plan, 'model_dump'):
            original_plan_str = str(plan.model_dump())
        elif isinstance(plan, dict):
            original_plan_str = str(plan)
        else:
            original_plan_str = str(plan)
        
        # Build the improvement prompt
        prompt = PLAN_IMPROVEMENT_PROMPT.format(
            original_plan=original_plan_str,
            user_request=message,
        )
        
        # Use LLM with structured output
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
            streaming=True,
        )
        llm_structured = llm.with_structured_output(ImplementationPlan)
        
        try:
            improved_plan: ImplementationPlan = llm_structured.invoke([
                SystemMessage(content="You are a senior software architect improving implementation plans."),
                HumanMessage(content=prompt),
            ])
            
            # Format the improved plan
            yield f"## Improved Plan: {improved_plan.title}\n\n"
            yield f"{improved_plan.summary}\n\n"
            yield "### Steps\n\n"
            
            for step in improved_plan.steps:
                deps = f" (depends on: {', '.join(str(d) for d in step.depends_on)})" if step.depends_on else ""
                yield f"{step.step_number}. **[{step.repo.upper()}]** {step.description}{deps}\n"
                if step.files:
                    for f in step.files[:3]:
                        yield f"   - `{f}`\n"
                yield "\n"
            
            if improved_plan.risks:
                yield "### Risks\n\n"
                for risk in improved_plan.risks:
                    yield f"- {risk}\n"
                yield "\n"
            
            if improved_plan.testing_strategy:
                yield f"### Testing Strategy\n\n{improved_plan.testing_strategy}\n\n"
            
            yield {
                "type": "improve_plan",
                "implementation_plan": improved_plan,
            }
            
        except Exception as e:
            logger.error(f"Plan improvement failed: {e}")
            yield f"\nError improving plan: {str(e)}\n"
            yield {"type": "improve_plan", "error": str(e)}
    
    def _handle_build_request(
        self,
        message: str,
        classification: IntentClassification,
        memory: ConversationMemory,
        dry_run: bool = True,
    ) -> Generator[Union[str, dict], None, None]:
        """Handle a request to build/implement a plan."""
        last_plan_data = memory.get_last_plan()

        # Resolve feature description: prefer prior plan title, then LLM-extracted
        # feature, then the raw user message.
        if last_plan_data:
            plan = last_plan_data.get("implementation_plan")
            thread_id = last_plan_data.get("thread_id")
            feature_desc = (
                plan.title if hasattr(plan, "title")
                else plan.get("title", "") if isinstance(plan, dict)
                else ""
            ) or classification.extracted_feature or message
        else:
            plan = None
            thread_id = None
            feature_desc = classification.extracted_feature or message
        
        yield "> Starting build process...\n"
        if not last_plan_data:
            yield f"> No prior plan found — building directly from: *{feature_desc[:120]}*\n"
        yield f"> Dry run mode: {'ON (no files will be written)' if dry_run else 'OFF (files will be written)'}\n\n"

        try:
            # Resume from checkpoint when a prior plan thread exists
            if thread_id:
                yield f"> Resuming from checkpoint: {thread_id}\n"
                result = self.orchestrator.resume(thread_id, approved=True, dry_run=dry_run)
            else:
                result = self.orchestrator.build(
                    feature_description=feature_desc,
                    approved=True,
                    dry_run=dry_run,
                )
            
            # Stream progress
            if result.get("code_gen_results"):
                yield "### Code Generation Results\n\n"
                for cg in result["code_gen_results"]:
                    repo = cg.get("repo", "unknown")
                    if cg.get("error"):
                        yield f"- **{repo}**: Error - {cg['error']}\n"
                    else:
                        yield f"- **{repo}**: Completed\n"
                yield "\n"
            
            if result.get("branch_names"):
                yield "### Branches Created\n\n"
                for repo, branch in result["branch_names"].items():
                    yield f"- **{repo}**: `{branch}`\n"
                yield "\n"
            
            if result.get("pr_results"):
                yield "### Pull Requests\n\n"
                for pr in result["pr_results"]:
                    repo = pr.get("repo", "unknown")
                    if pr.get("success"):
                        pr_url = pr.get("pr_url", "")
                        pr_num = pr.get("pr_number", "")
                        if dry_run:
                            yield f"- **{repo}**: PR would be created (dry run)\n"
                        else:
                            yield f"- **{repo}**: [PR #{pr_num}]({pr_url})\n"
                    else:
                        yield f"- **{repo}**: Failed - {pr.get('error', 'Unknown error')}\n"
                yield "\n"
            
            if result.get("change_summary"):
                yield "### Files Changed\n\n"
                summary = result["change_summary"]
                try:
                    for repo, repo_summary in summary.items():
                        logger.debug(f"change_summary[{repo!r}] type={type(repo_summary).__name__} val={repr(repo_summary)[:200]}")
                        # get_summary() returns {repo: {"files": [...], "files_modified": N, ...}}
                        if isinstance(repo_summary, dict):
                            files = repo_summary.get("files", [])
                            created = repo_summary.get("files_created", 0)
                            modified = repo_summary.get("files_modified", 0)
                        elif isinstance(repo_summary, list):
                            files = repo_summary
                            created = modified = 0
                        else:
                            files = []
                            created = modified = 0
                        if files:
                            yield f"**{repo}** ({modified} modified, {created} created):\n"
                            for f in list(files)[:10]:
                                path = f["path"] if isinstance(f, dict) else str(f)
                                yield f"- `{path}`\n"
                            if len(files) > 10:
                                yield f"- ... and {len(files) - 10} more\n"
                            yield "\n"
                except Exception as cs_err:
                    import traceback as _tb
                    logger.error(f"change_summary rendering failed: {cs_err}\n{_tb.format_exc()}")
                    yield f"> *(Could not render file list: {cs_err})*\n"
            
            if result.get("error"):
                yield f"\n**Warning:** {result['error']}\n"
            
            yield {
                "type": "build_request",
                "pr_results": result.get("pr_results", []),
                "branch_names": result.get("branch_names", {}),
                "change_summary": result.get("change_summary", {}),
                "dry_run": dry_run,
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Build failed: {e}\n{traceback.format_exc()}")
            yield f"\nBuild failed: {str(e)}\n"
            yield {"type": "build_request", "error": str(e)}
    
    def _handle_clarification(
        self,
        classification: IntentClassification,
    ) -> Generator[Union[str, dict], None, None]:
        """Handle unclear or incomplete messages."""
        yield "I'm not sure I understand what you're asking for. Could you provide more details?\n\n"

        if classification.reasoning:
            yield f"*{classification.reasoning}*\n\n"

        yield "You can:\n"
        yield "- **Ask a question** about the codebase (e.g., \"How does the batch upload work?\")\n"
        yield "- **Request a plan** for a feature (e.g., \"Plan a feature to add pagination to the batch list\")\n"
        yield "- **Improve a plan** after one is generated (e.g., \"Add more detail to step 3\")\n"
        yield "- **Build** a feature directly (e.g., \"Build it\" or \"Build a feature to add pagination\")\n"

        yield {"type": "clarification"}
