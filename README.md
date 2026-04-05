# Knowledge Base MVP

A Python-based Knowledge Base system that ingests codebases (Orbit + Trinity-v2), stores structural data in Neo4j + vector embeddings in Qdrant, provides LangGraph-powered agents for cross-repo analysis, and exposes a Streamlit UI.

## Quick Start

### 1. Setup Environment

Requires **Python 3.9+**. Use **3.10 or newer** if you want the latest `tree-sitter` wheels; on 3.9 the project pins `tree-sitter` 0.23.x.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and set your values:
# - OPENAI_API_KEY (required)
# - ORBIT_REPO_PATH / TRINITY_REPO_PATH — use absolute paths, or paths relative to the
#   knowledge-base folder (e.g. `../orbit` if Orbit sits next to `knowledge-base/` on disk).
#   `./orbit` only works if you cloned Orbit *inside* knowledge-base.
```

### 3. Start Infrastructure

```bash
# Start Neo4j and Qdrant
docker-compose up -d

# Wait for services to be healthy
docker-compose ps
```

### 4. Run Ingestion

```bash
# Ingest both repositories
python scripts/ingest.py --all

# Or ingest individually
python scripts/ingest.py --repo orbit
python scripts/ingest.py --repo trinity
```

### 5. Launch UI

```bash
streamlit run ui/Home.py
```

Visit http://localhost:8501

## Architecture

```
knowledge-base/
├── ingestion/          # Code parsing and extraction
│   ├── parser.py       # Tree-sitter AST parsing
│   ├── extractors/     # Language-specific extractors
│   └── pipeline.py     # Full ingestion orchestrator
├── graph/              # Neo4j operations
│   ├── schema.py       # Constraints and indexes
│   ├── loader.py       # Batch loading
│   └── queries.py      # Cypher query helpers
├── vectors/            # Qdrant operations
│   ├── embedder.py     # OpenAI embeddings
│   ├── store.py        # Collection management
│   └── search.py       # Hybrid retrieval
├── agents/             # LangGraph agents
│   ├── tools.py        # LangChain tools
│   ├── subagent.py     # Per-repo agents
│   └── orchestrator.py # State machine
└── ui/                 # Streamlit interface
    ├── Home.py         # Dashboard
    └── pages/          # Feature pages
```

## Features

- **Graph Explorer**: Interactive visualization of code relationships
- **Code Search**: Semantic search across both repositories  
- **Impact Analysis**: Understand change propagation
- **Cross-Repo Map**: View FE-BE API connections
- **Feature Planner**: AI-powered feature planning

## API Key Setup

Your OpenAI API key goes in the `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

This key is used for:
- Generating code embeddings (text-embedding-3-small)
- Powering the LangGraph agents (gpt-4o)
