"""
PatternSampler — injects style-reference samples into codegen context (Item 8).

Before the codegen subagent writes new files, PatternSampler finds similar
existing files in the repository and provides them as style references.
This reduces style drift and ensures generated code matches repo conventions.
"""

import logging
from pathlib import Path
from typing import Optional
import glob as glob_module
import os

logger = logging.getLogger(__name__)


class PatternSampler:
    """
    Finds and samples similar existing files to provide style references
    for code generation.
    
    Strategy:
    1. For each target file, find similar files by:
       - Same directory/package
       - Same suffix pattern (e.g., *Controller.java, *.hook.ts)
       - Same file extension
    2. Read and truncate the content of the N most similar files
    3. Return formatted samples for injection into subagent context
    """
    
    # Common suffix patterns to match
    SUFFIX_PATTERNS = {
        # Java patterns
        "Controller.java": "*Controller.java",
        "Service.java": "*Service.java",
        "Repository.java": "*Repository.java",
        "Dto.java": "*Dto.java",
        "Request.java": "*Request.java",
        "Response.java": "*Response.java",
        # TypeScript/React patterns
        ".hook.ts": "*.hook.ts",
        ".hook.tsx": "*.hook.tsx",
        ".component.tsx": "*.component.tsx",
        ".test.ts": "*.test.ts",
        ".test.tsx": "*.test.tsx",
        ".stories.tsx": "*.stories.tsx",
        "page.tsx": "**/page.tsx",
        "layout.tsx": "**/layout.tsx",
        # General
        ".config.ts": "*.config.ts",
        ".config.js": "*.config.js",
    }
    
    def __init__(self, max_samples: int = 3, max_lines_per_sample: int = 300):
        """
        Initialize the pattern sampler.
        
        Args:
            max_samples: Maximum number of sample files to return per target
            max_lines_per_sample: Maximum lines to include from each sample
        """
        self.max_samples = max_samples
        self.max_lines_per_sample = max_lines_per_sample
    
    def find_similar_files(
        self,
        repo_path: Path,
        target_file_path: str,
        n: Optional[int] = None,
    ) -> list[Path]:
        """
        Find N most similar existing files to the target file.
        
        Args:
            repo_path: Root path of the repository
            target_file_path: Relative path of the file being written
            n: Number of similar files to find (defaults to max_samples)
            
        Returns:
            List of Path objects for similar files
        """
        n = n or self.max_samples
        target = Path(target_file_path)
        target_name = target.name
        target_parent = target.parent
        target_ext = target.suffix
        
        candidates: list[tuple[int, Path]] = []  # (score, path)
        
        # Strategy 1: Same directory, same extension
        same_dir = repo_path / target_parent
        if same_dir.exists():
            for file in same_dir.glob(f"*{target_ext}"):
                if file.is_file() and file.name != target_name:
                    candidates.append((10, file))  # High score for same dir
        
        # Strategy 2: Match suffix pattern
        for suffix, pattern in self.SUFFIX_PATTERNS.items():
            if target_name.endswith(suffix):
                matches = glob_module.glob(str(repo_path / "**" / pattern), recursive=True)
                for match in matches:
                    p = Path(match)
                    if p.is_file() and p.name != target_name:
                        # Higher score if in same parent directory
                        score = 8 if p.parent == same_dir else 5
                        candidates.append((score, p))
        
        # Strategy 3: Same extension anywhere in repo (lower priority)
        if len(candidates) < n:
            matches = glob_module.glob(str(repo_path / "**" / f"*{target_ext}"), recursive=True)
            for match in matches[:50]:  # Limit search
                p = Path(match)
                if p.is_file() and p.name != target_name:
                    # Check if not already in candidates
                    if not any(c[1] == p for c in candidates):
                        candidates.append((1, p))
        
        # Sort by score (descending) and return top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:n]]
    
    def read_samples(self, similar_files: list[Path]) -> list[dict]:
        """
        Read content from similar files and truncate.
        
        Args:
            similar_files: List of file paths to read
            
        Returns:
            List of dicts with 'path' and 'content' keys
        """
        samples = []
        
        for file_path in similar_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")
                
                if len(lines) > self.max_lines_per_sample:
                    truncated = "\n".join(lines[:self.max_lines_per_sample])
                    truncated += f"\n... (truncated, {len(lines)} total lines)"
                else:
                    truncated = content
                
                samples.append({
                    "path": str(file_path),
                    "content": truncated,
                })
            except Exception as e:
                logger.debug(f"Could not read sample file {file_path}: {e}")
        
        return samples
    
    def get_samples_for_files(
        self,
        repo_path: Optional[Path],
        target_files: list[str],
    ) -> str:
        """
        Get formatted style reference samples for a list of target files.
        
        This is the main entry point called by Subagent in codegen mode.
        
        Args:
            repo_path: Root path of the repository
            target_files: List of relative file paths being written
            
        Returns:
            Formatted string with file samples for injection into context
        """
        if not repo_path or not target_files:
            return ""
        
        all_samples: list[dict] = []
        seen_paths: set[str] = set()
        
        for target in target_files:
            similar = self.find_similar_files(repo_path, target)
            for sample in self.read_samples(similar):
                if sample["path"] not in seen_paths:
                    seen_paths.add(sample["path"])
                    all_samples.append(sample)
        
        if not all_samples:
            return ""
        
        # Format samples for context injection
        sections = []
        for sample in all_samples[:self.max_samples * 2]:  # Cap total samples
            rel_path = sample["path"]
            try:
                rel_path = str(Path(sample["path"]).relative_to(repo_path))
            except ValueError:
                pass
            
            sections.append(f"""
### Style Reference: `{rel_path}`

```
{sample["content"]}
```
""")
        
        header = f"""
## Style References ({len(sections)} files)

Match the patterns, naming conventions, and code style from these existing files in the repository:
"""
        
        return header + "\n".join(sections)
    
    def check_similar_exists(
        self,
        repo_path: Path,
        target_file_path: str,
    ) -> Optional[str]:
        """
        Check if a file with similar name already exists.
        
        This is a guard for the create_file tool to warn before creating
        files that might duplicate existing ones.
        
        Args:
            repo_path: Root path of the repository
            target_file_path: Relative path of the file being created
            
        Returns:
            Warning message if similar file exists, None otherwise
        """
        target = Path(target_file_path)
        target_stem = target.stem.lower()  # filename without extension
        target_ext = target.suffix
        
        # Check for exact name match (different case or location)
        matches = glob_module.glob(
            str(repo_path / "**" / f"*{target_ext}"),
            recursive=True
        )
        
        similar_files = []
        for match in matches[:100]:  # Limit search
            p = Path(match)
            if p.stem.lower() == target_stem:
                similar_files.append(str(p.relative_to(repo_path)))
        
        if similar_files:
            return (
                f"Warning: File(s) with similar name already exist: "
                f"{', '.join(similar_files[:3])}. "
                f"Consider editing the existing file instead of creating a new one."
            )
        
        return None


# Singleton instance for convenience
_sampler: Optional[PatternSampler] = None


def get_pattern_sampler() -> PatternSampler:
    """Get or create the global PatternSampler instance."""
    global _sampler
    if _sampler is None:
        _sampler = PatternSampler()
    return _sampler
