"""
Feature Builder: code generation, git operations, and GitHub PR creation.
"""

from .change_tracker import ChangeTracker, FileChange, get_tracker
from .git_ops import GitOps
from .pr_creator import PRCreator
from .code_generator import CodeGenerator
from .safety import (
    SafetyViolation,
    check_branch_name,
    check_file_path,
    check_file_count,
    check_diff_size,
    run_all_checks,
)

__all__ = [
    "ChangeTracker",
    "FileChange",
    "get_tracker",
    "GitOps",
    "PRCreator",
    "CodeGenerator",
    "SafetyViolation",
    "check_branch_name",
    "check_file_path",
    "check_file_count",
    "check_diff_size",
    "run_all_checks",
]
