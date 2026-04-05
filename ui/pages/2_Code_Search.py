"""
Code Search - Semantic search across repositories.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from neo4j import GraphDatabase

from config import settings
from vectors import HybridSearch, Embedder, VectorStore

st.set_page_config(page_title="Code Search", page_icon="🔍", layout="wide")
st.title("Code Search")
st.markdown("*Semantic search across Orbit & Trinity-v2*")


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


@st.cache_resource
def get_search():
    embedder = Embedder()
    store = VectorStore()
    return HybridSearch(store, embedder, get_driver())


# Sidebar filters
st.sidebar.header("Filters")

repo_filter = st.sidebar.selectbox(
    "Repository",
    ["Both", "orbit", "trinity"],
)

node_type_filter = st.sidebar.multiselect(
    "Node Types",
    ["Function", "Class", "Component", "Hook", "Type", "Interface"],
)

limit = st.sidebar.slider("Max Results", 5, 50, 20)

include_context = st.sidebar.checkbox("Include Graph Context", value=True)

# Search input
query = st.text_input(
    "Search query",
    placeholder="Describe what you're looking for...",
    help="Use natural language to describe the code you're looking for",
)

# Example queries
with st.expander("Example queries"):
    examples = [
        "batch file upload handler",
        "user authentication middleware",
        "React form validation",
        "API error handling",
        "database query for employees",
        "state management hooks",
    ]
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                query = example
                st.rerun()


if query:
    with st.spinner("Searching..."):
        try:
            search = get_search()
            
            repos = None if repo_filter == "Both" else [repo_filter]
            node_types = node_type_filter if node_type_filter else None
            
            results = search.search(
                query=query,
                repos=repos,
                node_types=node_types,
                limit=limit,
                include_graph_context=include_context,
            )
            
            if results:
                st.success(f"Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.container():
                        # Header
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {i}. {result.name}")
                        with col2:
                            st.markdown(f"**{result.node_type}**")
                        with col3:
                            score_color = "green" if result.score > 0.8 else "orange" if result.score > 0.6 else "red"
                            st.markdown(f":{score_color}[Score: {result.score:.3f}]")
                        
                        # File info
                        st.markdown(f"📁 `{result.file_path}:{result.start_line}` | 📦 {result.repo}")
                        
                        # Code preview
                        lang = "typescript" if result.repo == "orbit" else "java"
                        code_preview = result.content[:500]
                        if len(result.content) > 500:
                            code_preview += "\n... (truncated)"
                        
                        st.code(code_preview, language=lang)
                        
                        # Graph context
                        if include_context and result.graph_context:
                            with st.expander("Related code"):
                                for ctx in result.graph_context[:5]:
                                    node = ctx.get("node", {})
                                    rel = ctx.get("relationship", "related")
                                    direction = ctx.get("direction", "")
                                    arrow = "→" if direction == "outgoing" else "←"
                                    st.write(f"{arrow} **{rel}**: {node.get('name', 'unknown')}")
                        
                        # Actions
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("Find Similar", key=f"similar_{i}"):
                                st.session_state.find_similar = result.chunk_id
                        with col2:
                            if st.button("Show in Graph", key=f"graph_{i}"):
                                st.session_state.graph_node = result.name
                                st.switch_page("pages/1_Graph_Explorer.py")
                        with col3:
                            if st.button("Impact Analysis", key=f"impact_{i}"):
                                st.session_state.impact_node = result.name
                                st.switch_page("pages/3_Impact_Analysis.py")
                        
                        st.divider()
            else:
                st.warning("No results found. Try a different query or adjust filters.")
                
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.info("Make sure Qdrant is running and data has been indexed.")

else:
    st.info("Enter a search query to find code.")
    
    # Show stats
    try:
        store = VectorStore()
        from config import QDRANT_COLLECTIONS
        
        st.subheader("Indexed Code")
        cols = st.columns(len(QDRANT_COLLECTIONS))
        
        for i, (repo, collection) in enumerate(QDRANT_COLLECTIONS.items()):
            with cols[i]:
                try:
                    stats = store.get_collection_stats(collection)
                    st.metric(repo.title(), f"{stats.get('vectors_count', 0):,} chunks")
                except Exception:
                    st.metric(repo.title(), "Not indexed")
                    
    except Exception:
        pass

# Tips
st.sidebar.divider()
st.sidebar.subheader("Tips")
st.sidebar.markdown("""
- Use natural language descriptions
- Mention specific technologies (React, Spring)
- Describe functionality, not just names
- Filter by node type for precision
""")
