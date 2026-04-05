"""
Feature Planner & Builder — AI-powered feature planning and code generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from config import settings

st.set_page_config(page_title="Feature Planner & Builder", page_icon="🚀", layout="wide")


# ── Force-reload agents on every run so Streamlit's long-lived process ────────
# always uses the latest code from disk (not a cached module).
_stale = [k for k in sys.modules if k.startswith(("agents", "codegen"))]
for _m in _stale:
    sys.modules.pop(_m, None)


# ── Session state init ────────────────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "build_result" not in st.session_state:
    st.session_state.build_result = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_analysis(desc: str) -> dict:
    try:
        from agents import Orchestrator          # fresh import after reload above
        return Orchestrator().analyze(desc)
    except Exception as e:
        return {"error": str(e)}


def run_build(desc: str, dry_run: bool) -> dict:
    try:
        from agents import Orchestrator
        return Orchestrator().build(desc, approved=True, dry_run=dry_run)
    except Exception as e:
        return {"error": str(e)}


def _badge(confidence: str) -> str:
    color = {"high": "green", "medium": "orange", "low": "red"}.get(confidence, "gray")
    return f":{color}[{confidence.upper()}]"


# ═════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
st.title("Feature Planner & Builder")
st.markdown("*AI-powered feature planning and code generation across Orbit & Trinity-v2*")

# ── Step 1: input ─────────────────────────────────────────────────────────────
st.subheader("1. Describe your feature")

feature_description = st.text_area(
    "What do you want to build?",
    height=110,
    placeholder=(
        "Be specific — mention affected areas, constraints, and expected behaviour.\n\n"
        'E.g. "Add a GET /api/v1/health endpoint that returns {status: ok, timestamp} '
        'and show a status badge in the Orbit dashboard header."'
    ),
)

with st.expander("Example descriptions"):
    for ex in [
        "Add pagination (20 items/page) to the batch list view",
        "Add GET /api/v1/health returning {status: ok, timestamp}",
        "Employee search by name or department with 300 ms debounce",
        "Validate batch upload file size (max 10 MB) with a UI error message",
        "Show a banner notification when batch processing completes",
    ]:
        if st.button(ex, key=f"ex_{hash(ex)}"):
            # Write into widget via session state — rerun picks it up
            st.session_state["_prefill"] = ex
            st.rerun()

# Handle example prefill (Streamlit widget value can only be set via session state)
if "_prefill" in st.session_state:
    feature_description = st.session_state.pop("_prefill")

# ── Controls row ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 2, 1])

_has_text     = bool(feature_description.strip())
_analysed     = st.session_state.analysis_result is not None

with c1:
    do_analyse = st.button(
        "🔍 Analyse Feature",
        type="primary",
        disabled=not _has_text,
        use_container_width=True,
    )

with c2:
    # Build is clickable as soon as: there is text AND analyse has been run at least once
    do_build = st.button(
        "🚀 Build Feature",
        disabled=not (_has_text and _analysed),
        use_container_width=True,
        help=(
            "Requires Analyse to have run first.\n"
            "Generates code, creates a feature branch, and opens a Draft PR."
        ),
    )

with c3:
    dry_run = st.toggle("Dry-run", value=True,
                        help="ON = diffs only (safe). OFF = write files + push + open PR.")

if dry_run:
    st.caption("ℹ️ Dry-run ON — nothing written to disk or pushed.")
else:
    st.caption("⚠️ Dry-run OFF — will write files, commit, push, and open a Draft PR.")

# ── Status indicator showing whether Build is ready ───────────────────────────
if not _has_text:
    st.info("Enter a feature description above, then click **Analyse Feature**.")
elif not _analysed:
    st.info("Click **Analyse Feature** first — Build will unlock after analysis completes.")
else:
    st.success("✅ Analysis done — **Build Feature** is ready to click.")

# ═════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
if do_analyse and _has_text:
    st.session_state.build_result = None
    with st.status("Analysing…", expanded=True) as status:
        for step in [
            "Parsing request…",
            "Searching codebase (vector + graph)…",
            "Identifying cross-repo API links…",
            "Orbit subagent — frontend impact…",
            "Trinity subagent — backend impact…",
            "Merging results…",
            "Generating plan…",
        ]:
            st.write(step)
        result = run_analysis(feature_description)
        st.session_state.analysis_result = result
        if result.get("error"):
            status.update(label=f"Analysis error: {result['error'][:100]}", state="error")
        else:
            status.update(label="Analysis complete ✅", state="complete")

# ═════════════════════════════════════════════════════════════════════════════
# RUN BUILD
# ═════════════════════════════════════════════════════════════════════════════
if do_build and _has_text and _analysed:
    label = "Building (dry-run)…" if dry_run else "Building for real…"
    with st.status(label, expanded=True) as status:
        st.write("Analysis + planning…")
        st.write("Trinity codegen subagent…")
        st.write("Orbit codegen subagent…")
        st.write("Git branch + commit…")
        st.write("Pull request creation…")
        result = run_build(feature_description, dry_run=dry_run)
        st.session_state.build_result = result
        has_changes = bool(result.get("change_summary"))
        if result.get("error") and not has_changes:
            status.update(label=f"Build failed: {result['error'][:100]}", state="error")
        elif not has_changes:
            status.update(label="Build finished — no files generated (see details below)", state="complete")
        else:
            status.update(label="Dry-run complete ✅" if dry_run else "Build complete ✅", state="complete")

# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY ANALYSIS RESULTS
# ═════════════════════════════════════════════════════════════════════════════
result = st.session_state.analysis_result
if result:
    st.divider()
    if result.get("error"):
        st.error(f"**Analysis failed:** {result['error']}")
        st.info("Make sure Neo4j + Qdrant are running and code has been ingested.")
    elif result.get("needs_clarification"):
        st.warning(result.get("clarification_question", "Please provide more details."))
    else:
        st.subheader("2. Impact Map")
        impact = result.get("impact_map", {})
        col1, col2 = st.columns(2)

        def _repo_card(col, key, label, analysis_key):
            data = impact.get(key, {})
            analysis = result.get(analysis_key)
            conf = data.get("confidence", "unknown")
            with col:
                st.markdown(f"### {label}")
                st.markdown(f"Confidence: {_badge(conf)}")
                files = data.get("files", [])
                funcs = data.get("functions", [])
                if files:
                    st.markdown(f"**Files to modify** ({len(files)}):")
                    for f in files[:10]:
                        st.markdown(f"- `{f}`")
                if funcs:
                    with st.expander(f"Functions affected ({len(funcs)})"):
                        for f in funcs[:20]:
                            st.markdown(f"- `{f}`")
                if analysis:
                    raw = analysis.analysis if hasattr(analysis, "analysis") else str(analysis)
                    with st.expander("Full analysis"):
                        st.markdown(raw)

        _repo_card(col1, "orbit",   "🌐 Orbit (Frontend)",    "orbit_analysis")
        _repo_card(col2, "trinity", "⚙️ Trinity-v2 (Backend)", "trinity_analysis")

        st.divider()
        st.subheader("3. Implementation Plan")
        pr = result.get("pr_draft", {})
        st.markdown(f"### {pr.get('title', 'Feature')}")
        c1, c2 = st.columns(2)
        with c1:
            if pr.get("orbit_changes"):
                st.markdown("**Orbit changes:**")
                for ch in pr["orbit_changes"]:
                    st.markdown(f"- {ch}")
        with c2:
            if pr.get("trinity_changes"):
                st.markdown("**Trinity changes:**")
                for ch in pr["trinity_changes"]:
                    st.markdown(f"- {ch}")
        if pr.get("testing"):
            with st.expander("Testing"):
                for t in pr["testing"]: st.markdown(f"- {t}")
        if pr.get("risks"):
            with st.expander("⚠️ Risks"):
                for r in pr["risks"]: st.markdown(f"- {r}")

# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY BUILD RESULTS
# ═════════════════════════════════════════════════════════════════════════════
build = st.session_state.build_result
if build:
    st.divider()
    st.subheader("4. Build Results")

    change_summary = build.get("change_summary", {})
    pr_results     = build.get("pr_results", [])
    branch_names   = build.get("branch_names", {})
    code_gen       = build.get("code_gen_results", [])

    if build.get("error") and not change_summary:
        st.error(f"**Build failed:** {build['error']}")
    else:
        if dry_run:
            st.info("**Dry-run** — diffs generated. Toggle Dry-run OFF and click Build again to commit for real.")
        else:
            st.success("**Build complete!**")

        if build.get("error"):
            st.warning(f"Partial error: {build['error']}")

        if not change_summary:
            st.warning(
                "No file changes were recorded.\n\n"
                "**Likely reasons:**\n"
                "- Analysis confidence was `low` — the codegen subagent was skipped\n"
                "- The LLM finished without calling `edit_file` / `create_file`\n\n"
                "Try a more specific feature description, or check that Neo4j/Qdrant have data."
            )
            if code_gen:
                with st.expander("Codegen agent log"):
                    for item in code_gen:
                        if item.get("error"):
                            st.error(f"**{item['repo']}**: {item['error']}")
                        else:
                            st.markdown(f"**{item['repo']}:** {str(item.get('analysis',''))[:500]}")
        else:
            # PR links
            if pr_results:
                st.markdown("### Pull Requests")
                for pr in pr_results:
                    repo = pr.get("repo", "").split("/")[-1]
                    branch = pr.get("branch", "")
                    url = pr.get("pr_url", "")
                    if pr.get("error"):
                        st.error(f"**{repo}** PR failed: {pr['error']}")
                    elif dry_run:
                        st.markdown(f"**{repo}** — would open PR on `{branch}`")
                        st.code(url)
                    else:
                        st.markdown(f"**{repo}** — [PR #{pr.get('pr_number','?')} ↗]({url})")
            elif branch_names:
                st.markdown("### Feature Branches")
                for repo, branch in branch_names.items():
                    st.markdown(f"- **{repo}**: `{branch}`")

            # File change cards
            st.markdown("### Files Changed")
            for repo_name, summary in change_summary.items():
                label = "🌐 Orbit" if repo_name == "orbit" else "⚙️ Trinity-v2"
                mod = summary.get("files_modified", 0)
                cre = summary.get("files_created", 0)
                add = summary.get("lines_added", 0)
                rem = summary.get("lines_removed", 0)
                with st.expander(f"{label} — {mod} modified, {cre} new | +{add}/−{rem} lines", expanded=True):
                    for f in summary.get("files", []):
                        badge = "🆕" if f["is_new"] else "✏️"
                        st.markdown(
                            f"{badge} `{f['path']}` — **+{f['lines_added']}/−{f['lines_removed']}**  \n"
                            f"_{f['description']}_"
                        )

            # Diff viewer
            try:
                from codegen.change_tracker import get_tracker
                all_changes = get_tracker().get_changes()
                if all_changes:
                    st.markdown("### Unified Diffs")
                    for ch in all_changes:
                        diff = ch.unified_diff()
                        if diff.strip():
                            header = (
                                f"`{ch.repo}/{ch.file_path}` — "
                                f"{'🆕 New' if ch.is_new_file else '✏️ Modified'} "
                                f"+{ch.lines_added()}/−{ch.lines_removed()}"
                            )
                            with st.expander(header):
                                st.code(diff, language="diff")
            except Exception as e:
                st.warning(f"Could not load diffs: {e}")

            # Codegen log
            if code_gen:
                with st.expander("Codegen agent log"):
                    for item in code_gen:
                        if item.get("error"):
                            st.error(f"**{item['repo']}**: {item['error']}")
                        else:
                            st.markdown(f"**{item.get('repo','?')}:** {str(item.get('analysis',''))[:800]}")

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Feature Builder")
    st.markdown("""
**How it works:**
1. **Analyse** — maps impact across both repos
2. **Build** — generates code, creates a branch from `main`, commits, opens a Draft PR

PRs are always **Draft** — you review before merging.
The agent never touches `main` directly.
""")
    st.divider()
    st.markdown("**Config**")
    st.markdown(f"GitHub token: {'✅' if settings.github_token else '❌ set GITHUB_TOKEN in .env'}")
    st.markdown(f"Branch prefix: `{settings.feature_branch_prefix}`")
    st.markdown(f"Max files/repo: `{settings.max_files_per_repo}`")
    st.divider()
    st.markdown("**Infrastructure**")
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        d.verify_connectivity(); d.close()
        st.markdown("Neo4j: ✅")
    except Exception as e:
        st.markdown(f"Neo4j: ❌ `{str(e)[:50]}`")
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, check_compatibility=False)
        cols = [c.name for c in qc.get_collections().collections]
        st.markdown(f"Qdrant: ✅ `{', '.join(cols) or 'no collections'}`")
    except Exception as e:
        st.markdown(f"Qdrant: ❌ `{str(e)[:50]}`")
