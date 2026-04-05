"""
Cross-Repo Map - Visualize FE-BE API connections.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from neo4j import GraphDatabase

from config import settings
from graph.queries import QueryHelper

st.set_page_config(page_title="Cross-Repo Map", page_icon="🔗", layout="wide")
st.title("Cross-Repository Map")
st.markdown("*View connections between Orbit (FE) and Trinity-v2 (BE)*")


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def get_api_mappings():
    """Get all API endpoint to consumer mappings."""
    driver = get_driver()
    
    with driver.session() as session:
        # Get all endpoints with their consumers
        result = session.run("""
            MATCH (endpoint:APIEndpoint)
            OPTIONAL MATCH (endpoint)-[:EXPOSED_BY]->(handler)
            OPTIONAL MATCH (consumer)-[:CONSUMES]->(endpoint)
            RETURN endpoint, handler, collect(consumer) as consumers
            ORDER BY endpoint.path
        """)
        
        mappings = []
        for record in result:
            endpoint = dict(record["endpoint"])
            handler = dict(record["handler"]) if record["handler"] else None
            consumers = [dict(c) for c in record["consumers"]]
            
            mappings.append({
                "endpoint": endpoint,
                "handler": handler,
                "consumers": consumers,
                "has_consumers": len(consumers) > 0,
            })
        
        return mappings


def get_unmatched():
    """Get unmatched endpoints and API calls."""
    driver = get_driver()
    
    with driver.session() as session:
        # Unmatched endpoints (no consumers)
        unmatched_endpoints = session.run("""
            MATCH (endpoint:APIEndpoint)
            WHERE NOT exists((endpoint)<-[:CONSUMES]-())
            RETURN endpoint
            LIMIT 20
        """)
        
        endpoints = [dict(r["endpoint"]) for r in unmatched_endpoints]
        
        return {
            "unmatched_endpoints": endpoints,
        }


# Sidebar filters
st.sidebar.header("Filters")

api_prefix = st.sidebar.text_input(
    "API Path Filter",
    placeholder="/api/v1/batch",
)

show_unmatched = st.sidebar.checkbox("Show Unmatched Only", value=False)


# Main content
try:
    mappings = get_api_mappings()
    
    if api_prefix:
        mappings = [m for m in mappings if api_prefix.lower() in m["endpoint"].get("path", "").lower()]
    
    if show_unmatched:
        mappings = [m for m in mappings if not m["has_consumers"]]
    
    # Summary
    total = len(mappings)
    matched = sum(1 for m in mappings if m["has_consumers"])
    unmatched = total - matched
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Endpoints", total)
    with col2:
        st.metric("Matched", matched)
    with col3:
        st.metric("Unmatched", unmatched, delta=f"-{unmatched}" if unmatched > 0 else None, delta_color="inverse")
    
    st.divider()
    
    # Two-column view
    st.subheader("API Mappings")
    
    for mapping in mappings:
        endpoint = mapping["endpoint"]
        handler = mapping["handler"]
        consumers = mapping["consumers"]
        
        method = endpoint.get("http_method") or endpoint.get("method") or "?"
        path = endpoint.get("path", "unknown")
        
        # Color based on match status
        status_color = "green" if consumers else "red"
        
        with st.container():
            # Header
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### :{status_color}[{method}] `{path}`")
                
                st.markdown("**Backend (Trinity-v2)**")
                
                if handler:
                    st.markdown(f"- Handler: `{handler.get('name', 'unknown')}`")
                    st.markdown(f"- File: `{handler.get('file_path', 'N/A')}`")
                else:
                    st.markdown("- Handler: *Unknown*")
                
                # Controller info
                controller = endpoint.get("controller", "")
                if controller:
                    st.markdown(f"- Controller: `{controller}`")
            
            with col2:
                st.markdown("**Frontend (Orbit)**")
                
                if consumers:
                    for consumer in consumers[:5]:
                        name = consumer.get("name", "unknown")
                        file_path = consumer.get("file_path", "")
                        st.markdown(f"- `{name}`")
                        st.caption(f"  {file_path}")
                    
                    if len(consumers) > 5:
                        st.caption(f"  ... and {len(consumers) - 5} more")
                else:
                    st.warning("No frontend consumers found")
            
            st.divider()
    
    if not mappings:
        st.info("No API mappings found. Run ingestion first.")
    
    # Unmatched section
    if not show_unmatched:
        with st.expander("View Unmatched Endpoints"):
            unmatched = get_unmatched()
            
            st.markdown("### Endpoints without consumers")
            for ep in unmatched.get("unmatched_endpoints", []):
                method = ep.get("http_method") or ep.get("method") or "?"
                path = ep.get("path", "unknown")
                st.markdown(f"- :red[{method}] `{path}`")
            
            if not unmatched.get("unmatched_endpoints"):
                st.success("All endpoints have consumers!")

except Exception as e:
    st.error(f"Failed to load mappings: {str(e)}")
    st.info("Make sure Neo4j is running and data has been ingested.")


# Help
st.sidebar.divider()
st.sidebar.subheader("Help")
st.sidebar.markdown("""
This view shows how frontend code connects to backend APIs.

**Colors:**
- 🟢 Green: Endpoint has FE consumers
- 🔴 Red: No consumers found

**Actions:**
- Filter by API path
- Show only unmatched endpoints
- Click to explore in Graph
""")
