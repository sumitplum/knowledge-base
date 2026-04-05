"""
Impact Analysis - Understand change propagation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase

from config import settings
from graph.queries import QueryHelper

st.set_page_config(page_title="Impact Analysis", page_icon="📊", layout="wide")
st.title("Impact Analysis")
st.markdown("*Understand how changes propagate through the codebase*")


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


@st.cache_resource
def get_query_helper():
    return QueryHelper(get_driver())


# Sidebar
st.sidebar.header("Analysis Settings")

analysis_depth = st.sidebar.slider("Analysis Depth", 1, 3, 2)

show_cross_repo = st.sidebar.checkbox("Include Cross-Repo Impact", value=True)

# Input
col1, col2 = st.columns([3, 1])

with col1:
    target_node = st.text_input(
        "Node to analyze",
        placeholder="Enter function, class, or endpoint name...",
        value=st.session_state.get("impact_node", ""),
    )

with col2:
    search_type = st.selectbox(
        "Type",
        ["Function", "Class", "Component", "APIEndpoint"],
    )


def analyze_impact(node_name: str, depth: int):
    """Analyze impact of changes to a node."""
    helper = get_query_helper()
    driver = get_driver()
    
    with driver.session() as session:
        # Find the node
        result = session.run(f"""
            MATCH (n)
            WHERE n.name = $name
            RETURN n, labels(n) as labels
            LIMIT 1
        """, name=node_name)
        
        record = result.single()
        if not record:
            return None
        
        target = dict(record["n"])
        labels = record["labels"]
        
        # Find direct dependents (callers)
        callers_result = session.run("""
            MATCH (caller)-[r]->(n)
            WHERE n.name = $name
            RETURN caller, type(r) as rel_type, labels(caller) as labels
        """, name=node_name)
        
        callers = []
        for r in callers_result:
            callers.append({
                "node": dict(r["caller"]),
                "relationship": r["rel_type"],
                "labels": r["labels"],
            })
        
        # Find what this node depends on
        deps_result = session.run("""
            MATCH (n)-[r]->(dep)
            WHERE n.name = $name
            RETURN dep, type(r) as rel_type, labels(dep) as labels
        """, name=node_name)
        
        dependencies = []
        for r in deps_result:
            dependencies.append({
                "node": dict(r["dep"]),
                "relationship": r["rel_type"],
                "labels": r["labels"],
            })
        
        # For API endpoints, find cross-repo consumers
        cross_repo = []
        if "APIEndpoint" in labels:
            cross_result = session.run("""
                MATCH (consumer)-[:CONSUMES]->(endpoint:APIEndpoint)
                WHERE endpoint.name = $name OR endpoint.path CONTAINS $name
                RETURN consumer, consumer.repo as repo
            """, name=node_name)
            
            for r in cross_result:
                cross_repo.append({
                    "node": dict(r["consumer"]),
                    "repo": r["repo"],
                })
        
        return {
            "target": target,
            "labels": labels,
            "callers": callers,
            "dependencies": dependencies,
            "cross_repo": cross_repo,
        }


if target_node:
    with st.spinner("Analyzing impact..."):
        try:
            impact = analyze_impact(target_node, analysis_depth)
            
            if impact:
                # Summary
                st.subheader("Impact Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Direct Callers", len(impact["callers"]))
                with col2:
                    st.metric("Dependencies", len(impact["dependencies"]))
                with col3:
                    st.metric("Cross-Repo Impact", len(impact["cross_repo"]))
                
                # Target info
                st.subheader("Target Node")
                target = impact["target"]
                st.markdown(f"""
                - **Name**: {target.get('name', 'unknown')}
                - **Type**: {', '.join(impact['labels'])}
                - **File**: {target.get('file_path', 'N/A')}
                - **Repo**: {target.get('repo', 'N/A')}
                """)
                
                if target.get("signature"):
                    st.code(target.get("signature"))
                
                # Impact visualization
                st.subheader("Impact Graph")
                
                nodes = []
                edges = []
                
                # Center node
                center_id = "target"
                nodes.append(Node(
                    id=center_id,
                    label=target_node,
                    size=30,
                    color="#E91E63",
                ))
                
                # Callers (impacted)
                for i, caller in enumerate(impact["callers"][:20]):
                    node_id = f"caller_{i}"
                    name = caller["node"].get("name", "unknown")
                    nodes.append(Node(
                        id=node_id,
                        label=name[:15],
                        size=20,
                        color="#F44336",
                        title=f"IMPACTED: {name}",
                    ))
                    edges.append(Edge(
                        source=node_id,
                        target=center_id,
                        label=caller["relationship"],
                    ))
                
                # Dependencies
                for i, dep in enumerate(impact["dependencies"][:20]):
                    node_id = f"dep_{i}"
                    name = dep["node"].get("name", "unknown")
                    nodes.append(Node(
                        id=node_id,
                        label=name[:15],
                        size=15,
                        color="#2196F3",
                        title=f"Dependency: {name}",
                    ))
                    edges.append(Edge(
                        source=center_id,
                        target=node_id,
                        label=dep["relationship"],
                    ))
                
                # Cross-repo
                if show_cross_repo:
                    for i, cross in enumerate(impact["cross_repo"][:10]):
                        node_id = f"cross_{i}"
                        name = cross["node"].get("name", "unknown")
                        repo = cross.get("repo", "unknown")
                        nodes.append(Node(
                            id=node_id,
                            label=f"{name[:10]} ({repo})",
                            size=20,
                            color="#FF9800",
                            title=f"Cross-repo: {name} in {repo}",
                        ))
                        edges.append(Edge(
                            source=node_id,
                            target=center_id,
                            label="CONSUMES",
                            color="#FF9800",
                        ))
                
                config = Config(
                    width=1000,
                    height=500,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                )
                
                agraph(nodes=nodes, edges=edges, config=config)
                
                # Detailed lists
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Impacted Code (Callers)")
                    if impact["callers"]:
                        for caller in impact["callers"]:
                            node = caller["node"]
                            rel = caller["relationship"]
                            st.markdown(f"- **{node.get('name', 'unknown')}** ← {rel}")
                            st.caption(f"  {node.get('file_path', '')}")
                    else:
                        st.info("No direct callers found")
                
                with col2:
                    st.subheader("Dependencies")
                    if impact["dependencies"]:
                        for dep in impact["dependencies"]:
                            node = dep["node"]
                            rel = dep["relationship"]
                            st.markdown(f"- **{node.get('name', 'unknown')}** → {rel}")
                            st.caption(f"  {node.get('file_path', '')}")
                    else:
                        st.info("No dependencies found")
                
                # Cross-repo details
                if show_cross_repo and impact["cross_repo"]:
                    st.subheader("Cross-Repository Impact")
                    for cross in impact["cross_repo"]:
                        node = cross["node"]
                        repo = cross.get("repo", "unknown")
                        st.markdown(f"- **{node.get('name', 'unknown')}** in `{repo}`")
                        st.caption(f"  {node.get('file_path', '')}")
            else:
                st.warning(f"No node found with name '{target_node}'")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

else:
    st.info("Enter a node name to analyze its impact.")
    
    # Examples
    st.subheader("Try These Examples")
    
    examples = ["BatchController", "useAuth", "authenticatedFetch"]
    cols = st.columns(len(examples))
    
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, key=f"ex_{i}"):
                st.session_state.impact_node = example
                st.rerun()

# Legend
st.sidebar.divider()
st.sidebar.subheader("Legend")
st.sidebar.markdown("""
- 🔴 **Red**: Impacted (callers)
- 🔵 **Blue**: Dependencies
- 🟠 **Orange**: Cross-repo
- 🌸 **Pink**: Target node
""")
