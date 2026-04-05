"""
CodeGenerator — LLM-driven file edit loop.

Given a feature plan and a list of (file_path, instructions) tuples, this
module reads each file, sends it to the LLM with targeted instructions, and
records the full updated file via ChangeTracker.

Design decisions:
  - Always sends the *complete* current file to the LLM (not a patch), so the
    model has full context and produces a coherent result.
  - Validates that the returned content is plausibly the same file type
    (non-empty, same extension etc.) before accepting it.
  - Limits to settings.max_files_per_repo per run for safety.
  - Rolls back on any hard failure.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from .change_tracker import ChangeTracker, get_tracker

logger = logging.getLogger(__name__)


class CodeGenerationError(Exception):
    pass


class CodeGenerator:
    """
    Drives LLM code generation for a single repository.

    Parameters
    ----------
    repo: str
        "orbit" or "trinity"
    repo_path: Path
        Absolute path to the local clone.
    tracker: ChangeTracker
        Where edits are recorded.  Defaults to the process singleton.
    dry_run: bool
        When True, generate content but do not write to disk.
    """

    SYSTEM_PROMPT = """You are an expert software engineer implementing a feature in an existing codebase.

You will be given:
1. The current content of a source file
2. Specific instructions for what to change in that file

Your task:
- Return the COMPLETE updated file content — every line, not just the changed parts.
- Do NOT include any explanation, markdown fences, or commentary.
- Preserve existing code style, indentation, and conventions.
- Do NOT add import statements that aren't needed.
- Do NOT remove existing functionality unless the instructions explicitly ask.
- If the file needs no changes, return the original content verbatim.

CRITICAL: Your entire response must be the raw file contents only.
"""

    def __init__(
        self,
        repo: str,
        repo_path: Path,
        tracker: Optional[ChangeTracker] = None,
        dry_run: bool = True,
    ):
        self.repo = repo
        self.repo_path = Path(repo_path).resolve()
        self.tracker = tracker or get_tracker()
        self.dry_run = dry_run
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )

    def generate_changes(
        self,
        file_instructions: list[dict],
        overall_context: str = "",
    ) -> list[dict]:
        """
        Iterate over file_instructions and generate code for each.

        Parameters
        ----------
        file_instructions: list of dicts with keys:
            - file_path: str (relative to repo root)
            - instructions: str (what to change)
            - is_new_file: bool (optional, default False)
            - description: str (optional, human summary)

        overall_context: str
            High-level feature description for richer LLM context.

        Returns list of result dicts:
            {file_path, success, error, lines_added, lines_removed}
        """
        results = []

        # Safety cap
        limited = file_instructions[: settings.max_files_per_repo]
        if len(limited) < len(file_instructions):
            logger.warning(
                f"Capped file edits for {self.repo} from "
                f"{len(file_instructions)} to {len(limited)} "
                f"(MAX_FILES_PER_REPO={settings.max_files_per_repo})"
            )

        for spec in limited:
            file_path = spec["file_path"].lstrip("/")
            instructions = spec["instructions"]
            is_new = spec.get("is_new_file", False)
            description = spec.get("description", instructions[:80])

            result = self._process_single_file(
                file_path=file_path,
                instructions=instructions,
                is_new_file=is_new,
                description=description,
                overall_context=overall_context,
            )
            results.append(result)

        return results

    def _process_single_file(
        self,
        file_path: str,
        instructions: str,
        is_new_file: bool,
        description: str,
        overall_context: str,
    ) -> dict:
        abs_path = self.repo_path / file_path
        original_content: Optional[str] = None

        if is_new_file:
            if abs_path.exists():
                is_new_file = False  # Treat as modification

        if not is_new_file:
            if not abs_path.exists():
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": f"File not found: {abs_path}",
                }
            try:
                original_content = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": f"Could not read file: {e}",
                }

        try:
            new_content = self._call_llm(
                file_path=file_path,
                original_content=original_content or "",
                instructions=instructions,
                is_new_file=is_new_file,
                overall_context=overall_context,
            )
        except Exception as e:
            logger.error(f"LLM call failed for {file_path}: {e}")
            return {
                "file_path": file_path,
                "success": False,
                "error": f"LLM error: {e}",
            }

        # Basic validation
        if not new_content.strip():
            return {
                "file_path": file_path,
                "success": False,
                "error": "LLM returned empty content",
            }

        # Safety checks
        try:
            from .safety import run_all_checks, SafetyViolation
            run_all_checks(
                repo=self.repo,
                file_path=file_path,
                abs_path=abs_path,
                repo_root=self.repo_path,
                content=new_content,
            )
        except Exception as se:
            return {
                "file_path": file_path,
                "success": False,
                "error": f"Safety check: {se}",
            }

        # Record and optionally write
        change = self.tracker.record_edit(
            repo=self.repo,
            file_path=file_path,
            abs_path=abs_path,
            original_content=original_content,
            new_content=new_content,
            description=description,
            is_new_file=is_new_file,
        )

        if not self.dry_run:
            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(new_content, encoding="utf-8")
                logger.info(f"Wrote {abs_path}")
            except Exception as e:
                self.tracker.rollback(self.repo)
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": f"Write failed: {e}",
                }
        else:
            logger.info(f"[DRY-RUN] Generated content for {file_path} ({change.lines_added()} additions)")

        return {
            "file_path": file_path,
            "success": True,
            "error": None,
            "lines_added": change.lines_added(),
            "lines_removed": change.lines_removed(),
        }

    def _call_llm(
        self,
        file_path: str,
        original_content: str,
        instructions: str,
        is_new_file: bool,
        overall_context: str,
    ) -> str:
        file_section = (
            f"<current_file path=\"{file_path}\">\n{original_content}\n</current_file>"
            if not is_new_file
            else f"<new_file path=\"{file_path}\">\n(This file does not exist yet — create it from scratch.)\n</new_file>"
        )

        context_section = (
            f"\n<feature_context>\n{overall_context}\n</feature_context>\n"
            if overall_context
            else ""
        )

        user_message = f"""{context_section}
{file_section}

<instructions>
{instructions}
</instructions>

Return the complete updated file content with no surrounding markdown or explanation."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = self._llm.invoke(messages)
        raw = response.content

        # Strip accidental markdown fences the model sometimes adds
        raw = self._strip_markdown_fences(raw, file_path)
        return raw

    @staticmethod
    def _strip_markdown_fences(content: str, file_path: str) -> str:
        """Remove leading/trailing ```lang ... ``` wrappers if present."""
        stripped = content.strip()
        # Match: ```[optional lang]\n...\n```
        fence_match = re.match(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$", stripped)
        if fence_match:
            return fence_match.group(1)
        # Partial: starts with ``` but doesn't end with it
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                return stripped[first_newline + 1:].rstrip("`").strip()
        return content
