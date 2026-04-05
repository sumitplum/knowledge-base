"""
Central configuration for Knowledge Base MVP.
Loads environment variables and provides typed settings.
"""

from pathlib import Path
from typing import Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


# Directory containing this file (knowledge-base project root). Used to resolve relative repo paths.
_PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings and agents")
    
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="knowledge_base_2024")
    
    # Qdrant
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    
    # Repository paths
    orbit_repo_path: Optional[Path] = Field(default=None, description="Path to Orbit monorepo")
    trinity_repo_path: Optional[Path] = Field(default=None, description="Path to Trinity-v2 repo")
    
    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    embedding_batch_size: int = Field(default=100)
    
    # LLM settings
    llm_model: str = Field(default="gpt-4o")
    
    # Logging
    log_level: str = Field(default="INFO")
    
    # Chunking settings
    max_chunk_tokens: int = Field(default=1500)

    # GitHub integration (Feature Builder)
    github_token: Optional[str] = Field(default=None, description="GitHub Personal Access Token with repo scope")
    github_org: str = Field(default="PlumHQ", description="GitHub organisation name")
    orbit_repo_github: str = Field(default="PlumHQ/orbit", description="GitHub repo slug for Orbit")
    trinity_repo_github: str = Field(default="PlumHQ/trinity-v2", description="GitHub repo slug for Trinity-v2")

    # Feature Builder settings
    feature_branch_prefix: str = Field(default="kb/feature", description="Prefix for auto-created branches")
    max_files_per_repo: int = Field(default=15, description="Safety: max files to modify per repo in one run")
    dry_run_default: bool = Field(default=True, description="Default to dry-run (no writes) for safety")

    # Checkpointing settings (Item 9)
    checkpoint_db_path: str = Field(
        default="~/.kb/checkpoints.db",
        description="SQLite database path for workflow checkpoints"
    )

    # Security settings (Security Guardrails System)
    max_llm_calls_per_session: int = Field(
        default=50,
        description="Maximum LLM calls allowed per session"
    )
    max_github_api_calls_per_session: int = Field(
        default=20,
        description="Maximum GitHub API calls allowed per session"
    )
    max_file_writes_per_session: int = Field(
        default=30,
        description="Maximum file writes allowed per session"
    )
    max_builds_per_hour: int = Field(
        default=5,
        description="Maximum build operations allowed per hour"
    )
    max_tokens_per_session: int = Field(
        default=200_000,
        description="Maximum token budget per session"
    )
    audit_log_path: str = Field(
        default="~/.kb/audit.jsonl",
        description="Path to the JSON-lines audit log file"
    )
    allowed_pr_base_branches: list = Field(
        default=["main", "develop"],
        description="Allowed base branches for pull requests"
    )
    intent_guard_enabled: bool = Field(
        default=True,
        description="Enable LLM-based intent classification guard (set False for testing)"
    )
    max_diff_lines_hard_cap: int = Field(
        default=2000,
        description="Hard cap on total diff lines per repo (enforced, not just warned)"
    )

    @field_validator("orbit_repo_path", "trinity_repo_path", mode="before")
    @classmethod
    def _empty_repo_path_to_none(cls, v: Union[str, Path, None]) -> Optional[Union[str, Path]]:
        if v is None or v == "":
            return None
        return v

    @field_validator("orbit_repo_path", "trinity_repo_path", mode="after")
    @classmethod
    def _resolve_repo_paths(cls, v: Optional[Path]) -> Optional[Path]:
        """Resolve relative paths against the knowledge-base project root (not shell CWD)."""
        if v is None:
            return None
        if not v.is_absolute():
            return (_PROJECT_ROOT / v).resolve()
        return v.resolve()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


# Collection names for Qdrant
QDRANT_COLLECTIONS = {
    "orbit": "orbit_code",
    "trinity": "trinity_code",
}

# Node labels for Neo4j
NODE_LABELS = [
    "Repo",
    "Module", 
    "File",
    "Function",
    "Class",
    "Component",
    "Hook",
    "Route",
    "APIEndpoint",
    "Type",
]

# Relationship types for Neo4j
RELATIONSHIP_TYPES = [
    "CONTAINS",
    "IMPORTS",
    "CALLS",
    "DEPENDS_ON",
    "RENDERS",
    "USES_HOOK",
    "HAS_METHOD",
    "EXPOSED_BY",
    "CONSUMES",
    "HANDLED_BY",
]
