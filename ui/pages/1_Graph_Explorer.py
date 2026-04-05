"""
Graph Explorer - Interactive visualization of code relationships.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase

from config import settings
from graph.queries import QueryHelper

st.set_page_config(page_title="Graph Explorer", page_icon="🌐", layout="wide")
st.title("Graph Explorer")
st.markdown("*Explore code relationships visually*")


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


@st.cache_resource
def get_query_helper():
    return QueryHelper(get_driver())


# Sidebar filters
st.sidebar.header("Filters")

repo_filter = st.sidebar.selectbox(
    "Repository",
    ["All", "orbit", "trinity"],
)

node_type_filter = st.sidebar.multiselect(
    "Node Types",
    ["Function", "Class", "Component", "Hook", "APIEndpoint", "Route", "Type"],
    default=["Function", "Class", "Component"],
)

depth = st.sidebar.slider("Traversal Depth", 1, 3, 1)

# Search input
search_term = st.text_input(
    "Search for a node",
    placeholder="Enter function, class, or component name...",
)

# Color mapping for node types
NODE_COLORS = {
    "Function": "#4CAF50",
    "Class": "#2196F3",
    "Component": "#9C27B0",
    "Hook": "#FF9800",
    "APIEndpoint": "#F44336",
    "Route": "#00BCD4",
    "Type": "#607D8B",
    "File": "#795548",
    "default": "#9E9E9E",
}


def build_graph(node_name: str, repo: str, depth: int):
    """Build graph data for visualization."""
    helper = get_query_helper()
    
    repo_param = None if repo == "All" else repo
    
    # Get neighbors
    neighbors = helper.get_node_neighbors(node_name, repo_param, depth, limit=100)
    
    if not neighbors:
        return [], []
    
    nodes_dict = {}
    edges = []
    
    # Add center node
    center_id = f"center_{node_name}"
    nodes_dict[center_id] = Node(
        id=center_id,
        label=node_name,
        size=30,
        color="#E91E63",
    )
    
    # Add neighbors
    for neighbor in neighbors:
        node_data = neighbor["node"]
        node_name_n = node_data.get("name", "unknown")
        node_id = f"{node_name_n}_{id(node_data)}"
        
        # Determine node type from labels (simplified)
        node_type = "default"
        for label in NODE_COLORS.keys():
            if label.lower() in str(node_data).lower():
                node_type = label
                break
        
        if node_id not in nodes_dict:
            nodes_dict[node_id] = Node(
                id=node_id,
                label=node_name_n[:20],
                size=20,
                color=NODE_COLORS.get(node_type, NODE_COLORS["default"]),
                title=f"{node_type}: {node_name_n}\n{node_data.get('file_path', '')}",
            )
        
        # Add edge
        rel_type = neighbor.get("relationship", "RELATED")
        direction = neighbor.get("direction", "outgoing")
        
        if direction == "outgoing":
            edges.append(Edge(
                source=center_id,
                target=node_id,
                label=rel_type,
            ))
        else:
            edges.append(Edge(
                source=node_id,
                target=center_id,
                label=rel_type,
            ))
    
    return list(nodes_dict.values()), edges


if search_term:
    try:
        repo = repo_filter if repo_filter != "All" else "All"
        nodes, edges = build_graph(search_term, repo, depth)
        
        if nodes:
            st.success(f"Found {len(nodes)} nodes and {len(edges)} relationships")
            
            # Graph configuration
            config = Config(
                width=1200,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
            )
            
            # Render graph
            selected = agraph(nodes=nodes, edges=edges, config=config)
            
            if selected:
                st.info(f"Selected: {selected}")
            
            # Show node details
            with st.expander("Node Details"):
                helper = get_query_helper()
                results = helper.search_nodes(search_term, limit=5)
                for r in results:
                    node = r["node"]
                    st.write(f"**{node.get('name', 'unknown')}**")
                    st.write(f"- File: {node.get('file_path', 'N/A')}")
                    st.write(f"- Lines: {node.get('start_line', '?')}-{node.get('end_line', '?')}")
                    if node.get("signature"):
                        st.code(node.get("signature"), language="typescript")
                    st.divider()
        else:
            st.warning(f"No results found for '{search_term}'")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure Neo4j is running and data has been ingested.")

else:
    st.info("Enter a node name to explore its relationships.")
    
    # Show sample nodes
    st.subheader("Sample Nodes to Explore")
    
    try:
        helper = get_query_helper()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Functions**")
            with get_driver().session() as session:
                result = session.run("""
                    MATCH (f:Function)
                    WHERE f.exported = true
                    RETURN f.name as name, f.repo as repo
                    LIMIT 10
                """)
                for record in result:
                    if st.button(f"{record['name']} ({record['repo']})", key=f"func_{record['name']}"):
                        st.session_state.search_term = record['name']
                        st.rerun()
        
        with col2:
            st.markdown("**API Endpoints**")
            with get_driver().session() as session:
                result = session.run("""
                    MATCH (e:APIEndpoint)
                    RETURN e.method as method, e.path as path
                    LIMIT 10
                """)
                for record in result:
                    endpoint = f"{record['method']} {record['path']}"
                    st.write(f"- {endpoint}")
                    
    except Exception as e:
        st.warning("Could not load sample nodes. Make sure data is ingested.")

# Legend
st.sidebar.divider()
st.sidebar.subheader("Legend")
for node_type, color in NODE_COLORS.items():
    if node_type != "default":
        st.sidebar.markdown(f"<span style='color:{color}'>●</span> {node_type}", unsafe_allow_html=True)
