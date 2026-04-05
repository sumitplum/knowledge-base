"""
ChangeTracker — records all file edits made by the Feature Builder agent.

Each edit stores the original content alongside the new content so we can:
  - Generate unified diffs for the UI diff viewer
  - Commit only the actually modified files
  - Rollback by restoring original content on failure
"""

import difflib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """Represents a single file modification."""
    repo: str                       # "orbit" or "trinity"
    file_path: str                  # relative path within the repo
    abs_path: Path                  # absolute filesystem path
    original_content: Optional[str] # None for newly created files
    new_content: str
    description: str                # human-readable summary of the change
    is_new_file: bool = False
    is_deleted: bool = False

    def unified_diff(self, context_lines: int = 3) -> str:
        """Return a unified diff string."""
        before = (self.original_content or "").splitlines(keepends=True)
        after = self.new_content.splitlines(keepends=True)
        label_before = f"a/{self.file_path}" if not self.is_new_file else "/dev/null"
        label_after = f"b/{self.file_path}"
        diff = difflib.unified_diff(
            before,
            after,
            fromfile=label_before,
            tofile=label_after,
            n=context_lines,
        )
        return "".join(diff)

    def lines_added(self) -> int:
        diff = self.unified_diff()
        return sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))

    def lines_removed(self) -> int:
        diff = self.unified_diff()
        return sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))


class ChangeTracker:
    """
    Thread-safe registry of all file changes in a single Feature Builder run.

    Usage:
        tracker = get_tracker()
        tracker.record_edit(...)
        changes = tracker.get_changes("orbit")
        tracker.reset()
    """

    def __init__(self):
        self._lock = threading.Lock()
        # {repo: {file_path: FileChange}}
        self._changes: dict[str, dict[str, FileChange]] = {}

    def record_edit(
        self,
        repo: str,
        file_path: str,
        abs_path: Path,
        original_content: Optional[str],
        new_content: str,
        description: str,
        is_new_file: bool = False,
    ) -> FileChange:
        """Record a file edit. Can be called multiple times for the same file (last write wins)."""
        with self._lock:
            if repo not in self._changes:
                self._changes[repo] = {}

            # Keep the original content from the *first* edit of this file
            existing = self._changes[repo].get(file_path)
            original = existing.original_content if existing else original_content

            change = FileChange(
                repo=repo,
                file_path=file_path,
                abs_path=abs_path,
                original_content=original,
                new_content=new_content,
                description=description,
                is_new_file=is_new_file,
            )
            self._changes[repo][file_path] = change
            logger.debug(f"Tracked edit: {repo}/{file_path} ({'+' if is_new_file else '~'})")
            return change

    def get_changes(self, repo: Optional[str] = None) -> list[FileChange]:
        """Return all changes, optionally filtered by repo."""
        with self._lock:
            if repo:
                return list(self._changes.get(repo, {}).values())
            return [c for repo_changes in self._changes.values() for c in repo_changes.values()]

    def has_changes(self, repo: Optional[str] = None) -> bool:
        return len(self.get_changes(repo)) > 0

    def change_count(self, repo: Optional[str] = None) -> int:
        return len(self.get_changes(repo))

    def repos_with_changes(self) -> list[str]:
        with self._lock:
            return [r for r, ch in self._changes.items() if ch]

    def get_summary(self) -> dict:
        """Return a human-readable summary for the UI."""
        summary = {}
        for repo in self.repos_with_changes():
            changes = self.get_changes(repo)
            summary[repo] = {
                "files_modified": len([c for c in changes if not c.is_new_file]),
                "files_created": len([c for c in changes if c.is_new_file]),
                "lines_added": sum(c.lines_added() for c in changes),
                "lines_removed": sum(c.lines_removed() for c in changes),
                "files": [
                    {
                        "path": c.file_path,
                        "description": c.description,
                        "is_new": c.is_new_file,
                        "lines_added": c.lines_added(),
                        "lines_removed": c.lines_removed(),
                    }
                    for c in changes
                ],
            }
        return summary

    def rollback(self, repo: Optional[str] = None):
        """Restore original file contents on disk (undo all writes)."""
        changes = self.get_changes(repo)
        restored = 0
        for change in changes:
            try:
                if change.is_new_file:
                    if change.abs_path.exists():
                        change.abs_path.unlink()
                        logger.info(f"Rollback: deleted new file {change.abs_path}")
                elif change.original_content is not None:
                    change.abs_path.write_text(change.original_content, encoding="utf-8")
                    logger.info(f"Rollback: restored {change.abs_path}")
                restored += 1
            except Exception as e:
                logger.error(f"Rollback failed for {change.abs_path}: {e}")
        logger.info(f"Rollback complete: {restored}/{len(changes)} files restored")

    def reset(self, repo: Optional[str] = None):
        """Clear tracked changes (call after successful commit)."""
        with self._lock:
            if repo:
                self._changes.pop(repo, None)
            else:
                self._changes.clear()


# Process-level singleton — shared between tools and orchestrator nodes
_tracker_instance: Optional[ChangeTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> ChangeTracker:
    """Get or create the process-level ChangeTracker singleton."""
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = ChangeTracker()
        return _tracker_instance
