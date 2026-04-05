"""
Chat Interface — Conversational code intelligence for Orbit & Trinity-v2.

A Cursor/Claude-style chat page that unifies Q&A, impact analysis, feature planning,
plan improvement, and PR creation into a single streaming chat interface.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from neo4j import GraphDatabase

from config import settings

st.set_page_config(
    page_title="Chat - Knowledge Base",
    page_icon="💬",
    layout="wide",
)


# ── Force-reload agents on every run ──────────────────────────────────────────
_stale = [k for k in sys.modules if k.startswith(("agents", "codegen"))]
for _m in _stale:
    sys.modules.pop(_m, None)


# ── Imports after reload ──────────────────────────────────────────────────────
from agents.conversation_memory import ConversationMemory
from agents.chat_router import ChatRouter


# ── Session state initialization ──────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "thread_id" not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.session_state.thread_id = f"kb-chat-{timestamp}"

if "dry_run" not in st.session_state:
    st.session_state.dry_run = True

if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

if "chat_router" not in st.session_state:
    st.session_state.chat_router = ChatRouter()


# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_resource
def get_neo4j_driver():
    """Get cached Neo4j driver."""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def check_infrastructure() -> dict:
    """Check if Neo4j and Qdrant are accessible."""
    status = {"neo4j": False, "qdrant": False}
    
    # Check Neo4j
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        status["neo4j"] = True
    except Exception:
        pass
    
    # Check Qdrant
    try:
        from vectors import VectorStore
        store = VectorStore()
        # Simple check - if we can instantiate, it's working
        status["qdrant"] = True
    except Exception:
        pass
    
    return status


def render_plan_card(plan_data: dict):
    """Render a plan as a card with action buttons."""
    plan = plan_data.get("implementation_plan")
    if not plan:
        return
    
    with st.container(border=True):
        st.markdown(f"### 📋 Implementation Plan: {plan.title}")
        st.markdown(plan.summary)
        
        with st.expander("View Steps", expanded=True):
            for step in plan.steps:
                deps = f" *(depends on: {', '.join(str(d) for d in step.depends_on)})*" if step.depends_on else ""
                st.markdown(f"**{step.step_number}.** [{step.repo.upper()}] {step.description}{deps}")
                if step.files:
                    for f in step.files[:3]:
                        st.markdown(f"   - `{f}`")
        
        if plan.risks:
            with st.expander("Risks"):
                for risk in plan.risks:
                    st.markdown(f"- {risk}")
        
        if plan.testing_strategy:
            with st.expander("Testing Strategy"):
                st.markdown(plan.testing_strategy)
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Build this", key=f"build_{id(plan_data)}", use_container_width=True):
                st.session_state.pending_action = "build it"
                st.rerun()
        
        with col2:
            if st.button("✏️ Improve plan", key=f"improve_{id(plan_data)}", use_container_width=True):
                st.session_state.pending_action = "improve: make it more detailed"
                st.rerun()


def render_pr_results(pr_results: list, dry_run: bool):
    """Render PR results with links and diff expanders."""
    if not pr_results:
        return
    
    with st.container(border=True):
        st.markdown("### 🔗 Pull Requests")
        
        for pr in pr_results:
            repo = pr.get("repo", "unknown")
            
            if pr.get("success"):
                pr_url = pr.get("pr_url", "")
                pr_num = pr.get("pr_number", "")
                
                if dry_run:
                    st.info(f"**{repo}**: PR would be created (dry run mode)")
                else:
                    st.success(f"**{repo}**: [PR #{pr_num}]({pr_url})")
            else:
                error = pr.get("error", "Unknown error")
                st.error(f"**{repo}**: Failed - {error}")


def render_message(turn):
    """Render a single conversation turn."""
    role = turn.role
    content = turn.content
    
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            
            # Render structured data if present
            if turn.structured_data:
                if turn.has_plan():
                    render_plan_card(turn.structured_data)
                
                if turn.has_pr_results():
                    pr_results = turn.structured_data.get("pr_results", [])
                    dry_run = turn.structured_data.get("dry_run", True)
                    render_pr_results(pr_results, dry_run)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Chat Controls")
    
    # New Chat button
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.memory.clear()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.session_state.thread_id = f"kb-chat-{timestamp}"
        st.session_state.pending_action = None
        st.rerun()
    
    st.divider()
    
    # Session info
    st.subheader("Session Info")
    st.text_input(
        "Thread ID",
        value=st.session_state.thread_id,
        disabled=True,
        help="Use this ID to resume the session later",
    )
    
    turn_count = len(st.session_state.memory)
    st.metric("Conversation Turns", turn_count)
    
    # Dry-run toggle
    st.divider()
    st.subheader("Build Settings")
    st.session_state.dry_run = st.toggle(
        "Dry Run Mode",
        value=st.session_state.dry_run,
        help="When enabled, build operations won't write files or create PRs",
    )
    
    if st.session_state.dry_run:
        st.info("✓ Safe mode - no files will be written")
    else:
        st.warning("⚠️ Live mode - files will be written and PRs created")
    
    # Infrastructure status
    st.divider()
    st.subheader("Infrastructure")
    
    with st.spinner("Checking..."):
        infra_status = check_infrastructure()
    
    neo4j_icon = "✅" if infra_status["neo4j"] else "❌"
    qdrant_icon = "✅" if infra_status["qdrant"] else "❌"
    
    st.markdown(f"**Neo4j:** {neo4j_icon}")
    st.markdown(f"**Qdrant:** {qdrant_icon}")
    
    if not all(infra_status.values()):
        st.warning("Some services are offline. Chat may have limited functionality.")
    
    # Resume session
    st.divider()
    st.subheader("Resume Session")
    
    resume_thread_id = st.text_input(
        "Thread ID to resume",
        placeholder="kb-chat-20260403-141200",
    )
    
    if st.button("Resume", disabled=not resume_thread_id, use_container_width=True):
        st.session_state.thread_id = resume_thread_id
        st.info(f"Resumed session: {resume_thread_id}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════════════════════════════════════
st.title("💬 Chat")
st.markdown("*Conversational code intelligence for Orbit & Trinity-v2*")

# Display conversation history
memory: ConversationMemory = st.session_state.memory

if len(memory) == 0:
    st.info(
        "👋 Welcome! I'm your code intelligence assistant. You can:\n\n"
        "- **Ask questions** about the codebase (e.g., 'How does batch upload work?')\n"
        "- **Plan features** (e.g., 'Plan a feature to add pagination to the batch list')\n"
        "- **Improve plans** after they're generated (e.g., 'Make step 3 more specific')\n"
        "- **Build features** from plans (e.g., 'Build it' or 'Create the PR')"
    )
else:
    for turn in memory.turns:
        render_message(turn)

# Handle pending action from buttons
if st.session_state.pending_action:
    pending_msg = st.session_state.pending_action
    st.session_state.pending_action = None
    
    # Add user message
    memory.add_user_message(pending_msg)
    
    with st.chat_message("user"):
        st.markdown(pending_msg)
    
    # Process the message
    router: ChatRouter = st.session_state.chat_router
    
    with st.chat_message("assistant"):
        # Classify intent
        with st.spinner("Thinking..."):
            classification = router.classify(pending_msg, memory)
        
        # Handle and stream response
        response_text = ""
        response_placeholder = st.empty()
        
        structured_data = None
        for chunk_or_data in router.handle(
            classification,
            pending_msg,
            memory,
            st.session_state.dry_run,
        ):
            if isinstance(chunk_or_data, str):
                response_text += chunk_or_data
                response_placeholder.markdown(response_text)
            elif isinstance(chunk_or_data, dict):
                structured_data = chunk_or_data
        
        # Add assistant response to memory
        memory.add_assistant_message(
            response_text,
            intent=classification.intent,
            structured_data=structured_data,
        )
        
        # Render structured data if present
        if structured_data:
            if structured_data.get("implementation_plan"):
                render_plan_card(structured_data)
            
            if structured_data.get("pr_results"):
                render_pr_results(
                    structured_data["pr_results"],
                    structured_data.get("dry_run", True),
                )
    
    st.rerun()

# Chat input
user_input = st.chat_input("Type a message...")

if user_input:
    # Add user message to memory
    memory.add_user_message(user_input)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process the message
    router: ChatRouter = st.session_state.chat_router
    
    with st.chat_message("assistant"):
        # Classify intent
        with st.spinner("Thinking..."):
            classification = router.classify(user_input, memory)
        
        # Handle and stream response
        response_text = ""
        response_placeholder = st.empty()
        
        structured_data = None
        for chunk_or_data in router.handle(
            classification,
            user_input,
            memory,
            st.session_state.dry_run,
        ):
            if isinstance(chunk_or_data, str):
                response_text += chunk_or_data
                response_placeholder.markdown(response_text)
            elif isinstance(chunk_or_data, dict):
                structured_data = chunk_or_data
        
        # Add assistant response to memory
        memory.add_assistant_message(
            response_text,
            intent=classification.intent,
            structured_data=structured_data,
        )
        
        # Render structured data if present
        if structured_data:
            if structured_data.get("implementation_plan"):
                render_plan_card(structured_data)
            
            if structured_data.get("pr_results"):
                render_pr_results(
                    structured_data["pr_results"],
                    structured_data.get("dry_run", True),
                )
    
    st.rerun()
