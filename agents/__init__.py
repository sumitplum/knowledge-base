"""
Agents package initialization.
"""

from .tools import ALL_TOOLS, search_code, get_node_graph, find_callers, find_api_contracts, cross_repo_trace
from .subagent import Subagent, SubagentResponse, create_orbit_subagent, create_trinity_subagent
from .orchestrator import Orchestrator, create_orchestrator
from .conversation_memory import ConversationMemory, ConversationTurn
from .chat_router import ChatRouter
from .chat_router import IntentClassification as ChatIntentClassification

# Security Guardrails System exports
from .exceptions import (
    KBBaseError,
    SecurityViolation,
    SafetyViolation,
    RateLimitExceeded,
    AuditError,
)
from .audit import AuditLogger, get_audit_logger, AuditEventType
from .guardrails import IntentGuard, get_intent_guard, guard_query, IntentClassification
from .rate_limiter import SessionRateLimiter, get_rate_limiter, RateLimitConfig

__all__ = [
    # Tools
    "ALL_TOOLS",
    "search_code",
    "get_node_graph",
    "find_callers",
    "find_api_contracts",
    "cross_repo_trace",
    # Agents
    "Subagent",
    "SubagentResponse",
    "create_orbit_subagent",
    "create_trinity_subagent",
    "Orchestrator",
    "create_orchestrator",
    # Chat Interface
    "ConversationMemory",
    "ConversationTurn",
    "ChatRouter",
    "ChatIntentClassification",
    # Security: Exceptions
    "KBBaseError",
    "SecurityViolation",
    "SafetyViolation",
    "RateLimitExceeded",
    "AuditError",
    # Security: Audit
    "AuditLogger",
    "get_audit_logger",
    "AuditEventType",
    # Security: Intent Guard
    "IntentGuard",
    "get_intent_guard",
    "guard_query",
    "IntentClassification",
    # Security: Rate Limiter
    "SessionRateLimiter",
    "get_rate_limiter",
    "RateLimitConfig",
]
