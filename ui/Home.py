"""
Knowledge Base MVP - Streamlit Home Page
Dashboard with stats and quick access to features.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from neo4j import GraphDatabase

from config import settings
from graph import SchemaManager
from vectors import VectorStore

st.set_page_config(
    page_title="Knowledge Base",
    page_icon="📚",
    layout="wide",
)

st.title("Knowledge Base MVP")
st.markdown("*Cross-repository code intelligence for Orbit & Trinity-v2*")

# Primary CTA: Open Chat
st.markdown("### 💬 Get Started with Chat")
st.markdown("The fastest way to interact with your codebase - ask questions, plan features, and generate code in a conversational interface.")

if st.button("🚀 Open Chat", type="primary", use_container_width=True):
    st.switch_page("pages/6_Chat.py")

st.divider()


@st.cache_resource
def get_neo4j_driver():
    """Get cached Neo4j driver."""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


@st.cache_resource
def get_vector_store():
    """Get cached vector store."""
    return VectorStore()


def get_graph_stats():
    """Get statistics from Neo4j."""
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Count nodes by label
            result = session.run("""
                MATCH (n)
                WITH labels(n) AS labels, count(*) AS count
                UNWIND labels AS label
                RETURN label, sum(count) AS total
                ORDER BY total DESC
            """)
            node_counts = {r["label"]: r["total"] for r in result}
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            
            # Count repos
            result = session.run("MATCH (r:Repo) RETURN count(r) as count")
            repo_count = result.single()["count"]
            
            return {
                "nodes": node_counts,
                "relationships": rel_count,
                "repos": repo_count,
                "connected": True,
            }
    except Exception as e:
        return {"error": str(e), "connected": False}


def get_vector_stats():
    """Get statistics from Qdrant."""
    try:
        store = get_vector_store()
        from config import QDRANT_COLLECTIONS
        
        stats = {}
        for repo, collection in QDRANT_COLLECTIONS.items():
            try:
                info = store.get_collection_stats(collection)
                stats[repo] = info
            except Exception:
                stats[repo] = {"vectors_count": 0, "status": "not found"}
        
        return {"collections": stats, "connected": True}
    except Exception as e:
        return {"error": str(e), "connected": False}


# Dashboard layout
col1, col2, col3 = st.columns(3)

# Graph stats
graph_stats = get_graph_stats()

with col1:
    st.subheader("Graph Database")
    if graph_stats.get("connected"):
        total_nodes = sum(graph_stats.get("nodes", {}).values())
        st.metric("Total Nodes", f"{total_nodes:,}")
        st.metric("Relationships", f"{graph_stats.get('relationships', 0):,}")
        st.metric("Repositories", graph_stats.get("repos", 0))
        
        with st.expander("Node breakdown"):
            for label, count in sorted(graph_stats.get("nodes", {}).items(), key=lambda x: -x[1]):
                st.write(f"- **{label}**: {count:,}")
    else:
        st.error(f"Neo4j not connected: {graph_stats.get('error', 'Unknown error')}")
        st.info("Start Neo4j with: `docker-compose up -d`")

# Vector stats
vector_stats = get_vector_stats()

with col2:
    st.subheader("Vector Store")
    if vector_stats.get("connected"):
        collections = vector_stats.get("collections", {})
        total_vectors = sum(c.get("vectors_count", 0) for c in collections.values())
        st.metric("Total Vectors", f"{total_vectors:,}")
        
        for repo, info in collections.items():
            count = info.get("vectors_count", 0)
            status = info.get("status", "unknown")
            st.write(f"- **{repo}**: {count:,} vectors ({status})")
    else:
        st.error(f"Qdrant not connected: {vector_stats.get('error', 'Unknown error')}")
        st.info("Start Qdrant with: `docker-compose up -d`")

# Quick actions
with col3:
    st.subheader("Quick Actions")
    
    if st.button("🔍 Search Code", use_container_width=True):
        st.switch_page("pages/2_Code_Search.py")
    
    if st.button("🌐 Explore Graph", use_container_width=True):
        st.switch_page("pages/1_Graph_Explorer.py")
    
    if st.button("📊 Impact Analysis", use_container_width=True):
        st.switch_page("pages/3_Impact_Analysis.py")
    
    if st.button("🔗 Cross-Repo Map", use_container_width=True):
        st.switch_page("pages/4_Cross_Repo_Map.py")
    
    if st.button("📝 Feature Planner", use_container_width=True):
        st.switch_page("pages/5_Feature_Planner.py")

# Quick search
st.divider()
st.subheader("Quick Search")

search_query = st.text_input(
    "Search code",
    placeholder="e.g., batch upload handler, user authentication...",
)

if search_query:
    st.info(f"Searching for: {search_query}")
    st.page_link("pages/2_Code_Search.py", label=f"View full results →")

# Repository info
st.divider()
st.subheader("Repositories")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Orbit (Frontend)
    - **Framework**: Next.js 16 / React 19
    - **Language**: TypeScript
    - **Apps**: hr-dashboard, trinity
    - **Path**: `{}`
    """.format(settings.orbit_repo_path or "Not configured"))

with col2:
    st.markdown("""
    ### Trinity-v2 (Backend)
    - **Framework**: Spring Boot 3.5
    - **Language**: Java 21
    - **API**: REST under `/api/v1/`
    - **Path**: `{}`
    """.format(settings.trinity_repo_path or "Not configured"))

# Footer
st.divider()
st.caption("Knowledge Base MVP | Powered by Neo4j, Qdrant, and LangGraph")
