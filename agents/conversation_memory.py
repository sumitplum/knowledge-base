"""
Conversation memory for the chat interface.

Provides stateful conversation history with context-window management,
plan retrieval, and LangChain message conversion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    
    role: str  # "user" or "assistant"
    content: str  # Rendered text shown in UI
    intent: Optional[str] = None  # Classified intent for this turn
    structured_data: Optional[dict] = None  # Stores raw plan/PR/impact objects
    timestamp: datetime = field(default_factory=datetime.now)
    
    def has_plan(self) -> bool:
        """Check if this turn contains an implementation plan."""
        if not self.structured_data:
            return False
        return "implementation_plan" in self.structured_data
    
    def has_pr_results(self) -> bool:
        """Check if this turn contains PR results."""
        if not self.structured_data:
            return False
        return "pr_results" in self.structured_data


class ConversationMemory:
    """
    Stateful conversation history with context-window management.
    
    Stored in st.session_state.memory as the single source of truth.
    """
    
    def __init__(self):
        self._turns: list[ConversationTurn] = []
    
    def add(self, turn: ConversationTurn) -> None:
        """Append a turn to the conversation history."""
        self._turns.append(turn)
    
    def add_user_message(self, content: str, intent: Optional[str] = None) -> ConversationTurn:
        """Convenience method to add a user message."""
        turn = ConversationTurn(
            role="user",
            content=content,
            intent=intent,
        )
        self.add(turn)
        return turn
    
    def add_assistant_message(
        self,
        content: str,
        intent: Optional[str] = None,
        structured_data: Optional[dict] = None,
    ) -> ConversationTurn:
        """Convenience method to add an assistant message."""
        turn = ConversationTurn(
            role="assistant",
            content=content,
            intent=intent,
            structured_data=structured_data,
        )
        self.add(turn)
        return turn
    
    def get_last_plan(self) -> Optional[dict]:
        """
        Scan history backwards for the most recent implementation plan.
        
        Returns the structured_data dict containing the plan, or None.
        """
        for turn in reversed(self._turns):
            if turn.has_plan():
                return turn.structured_data
        return None
    
    def get_last_pr_results(self) -> Optional[list]:
        """
        Scan history backwards for the most recent PR results.
        
        Returns the pr_results list, or None.
        """
        for turn in reversed(self._turns):
            if turn.has_pr_results():
                return turn.structured_data.get("pr_results")
        return None
    
    def get_context_window(self, max_turns: int = 12) -> list[ConversationTurn]:
        """
        Return the most recent N turns.
        
        Always includes any turn that contains a plan so plan context
        is never lost mid-conversation.
        """
        if len(self._turns) <= max_turns:
            return list(self._turns)
        
        # Get the most recent turns
        recent = self._turns[-max_turns:]
        
        # Check if we already have a plan in recent turns
        has_plan_in_recent = any(t.has_plan() for t in recent)
        
        if has_plan_in_recent:
            return recent
        
        # Find the most recent plan turn and include it
        for i, turn in enumerate(reversed(self._turns)):
            if turn.has_plan():
                plan_index = len(self._turns) - 1 - i
                # Include the plan turn at the start, then recent turns
                return [self._turns[plan_index]] + recent
        
        return recent
    
    def to_langchain_messages(self, max_turns: int = 12) -> list[BaseMessage]:
        """
        Convert conversation history to LangChain message format.
        
        Used for LLM calls that need conversation context.
        """
        messages: list[BaseMessage] = []
        
        for turn in self.get_context_window(max_turns):
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
        
        return messages
    
    def get_last_n_turns(self, n: int = 6) -> list[ConversationTurn]:
        """Get the last N turns for intent classification."""
        return self._turns[-n:] if len(self._turns) >= n else list(self._turns)
    
    def get_history_summary(self) -> str:
        """
        Generate a brief 2-sentence summary of conversation state.
        
        Used as context injection for the chat system prompt.
        """
        if not self._turns:
            return "This is the start of a new conversation. No prior context."
        
        turn_count = len(self._turns)
        last_plan = self.get_last_plan()
        last_pr = self.get_last_pr_results()
        
        parts = [f"Conversation has {turn_count} turns so far."]
        
        if last_plan:
            plan = last_plan.get("implementation_plan", {})
            if hasattr(plan, "title"):
                parts.append(f"Most recent plan: \"{plan.title}\".")
            elif isinstance(plan, dict) and plan.get("title"):
                parts.append(f"Most recent plan: \"{plan.get('title')}\".")
            else:
                parts.append("An implementation plan was generated.")
        elif last_pr:
            parts.append(f"PRs were created: {len(last_pr)} result(s).")
        else:
            # Summarize recent intents
            recent_intents = [t.intent for t in self._turns[-3:] if t.intent]
            if recent_intents:
                parts.append(f"Recent activity: {', '.join(recent_intents)}.")
        
        return " ".join(parts)
    
    def clear(self) -> None:
        """Wipe history for 'New Chat'."""
        self._turns = []
    
    @property
    def turns(self) -> list[ConversationTurn]:
        """Access the raw turns list (read-only view)."""
        return list(self._turns)
    
    def __len__(self) -> int:
        return len(self._turns)
    
    def __bool__(self) -> bool:
        return len(self._turns) > 0
