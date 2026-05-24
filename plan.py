"""Pure data model for the agent's task queue.

This module is import-time only — no I/O, no rendering, no LLM calls.
It exports `Task` and `Plan` dataclasses plus a `TASK_STATUSES` tuple
documenting the valid status values. The `Plan` class is the canonical
mutator for task status; the rest of the codebase reads `Plan` snapshots
but does not edit task fields directly.
"""

from dataclasses import dataclass
from typing import Optional


TASK_STATUSES = ("pending", "in_progress", "done", "failed", "skipped")


def _norm_desc(s: str) -> str:
    """Normalize a task description for duplicate detection.

    Strategy: lowercase, collapse runs of whitespace, drop trailing
    punctuation. Designed to catch "Write the file" vs "write the file."
    vs "Write  the  file" — three phrasings the architect emits across
    consecutive turns — without conflating genuinely different tasks
    that happen to share a few words.
    """
    if not s:
        return ""
    out = " ".join(s.lower().split())
    while out and out[-1] in ".!?,:;":
        out = out[:-1]
    return out.strip()


_STOPWORDS_LEADING = frozenset({
    "a", "an", "the", "to", "now", "then", "first", "next", "finally",
    "please",
})


def _leading_verb(s: str) -> str:
    """Return the first content word (typically a verb) of a task
    description, skipping leading stop-words. Used as a coarse signal
    that two paraphrased tasks are doing the *same* operation:
    "Create..." and "Verify..." can share most of their nouns but
    aren't the same task.
    """
    if not s:
        return ""
    for tok in s.lower().split():
        clean = "".join(ch for ch in tok if ch.isalpha())
        if clean and clean not in _STOPWORDS_LEADING:
            return clean
    return ""


def _ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio of two strings. Thin wrapper kept here so
    the import is localized (callers don't need difflib in scope) and
    so the dedupe heuristic can be tuned in one place."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Task:
    id: int                # 1-based, stable across plan lifetime
    description: str       # short user-facing line
    status: str = "pending"
    # Tier 2.2: when set, this task is a sub-task of `parent_id`. The
    # renderer walks the tree and indents sub-tasks under their parent.
    # None means top-level (the default).
    parent_id: Optional[int] = None


@dataclass
class Plan:
    title: str
    tasks: list             # list[Task]
    current_task_id: Optional[int] = None

    def get_task(self, task_id: int) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def advance(self, task_id: int, status: str) -> None:
        """Transition a task to a new status. Invalid status or unknown id is a no-op.
        Already-finished tasks (done/skipped) cannot be regressed to in_progress."""
        if status not in TASK_STATUSES:
            return
        task = self.get_task(task_id)
        if task is None:
            return
        # Don't regress completed work
        if task.status in ("done", "skipped") and status == "in_progress":
            return
        task.status = status
        if status == "in_progress":
            self.current_task_id = task_id
        elif status in ("done", "failed", "skipped") and self.current_task_id == task_id:
            self.current_task_id = None

    def insert(
        self,
        after_id: int,
        description: str,
        parent_id: Optional[int] = None,
    ) -> Task:
        """Insert a new task after the given task id. If after_id is unknown,
        appends to the end. Returns the new task (or an existing duplicate
        when one is detected — see below). The new id is `max(existing) + 1`.

        When `parent_id` is provided, the task is marked as a sub-task of
        that parent (renderer indents it). The architect emits this field
        in `new_tasks` entries when a step discovers concrete follow-ups.
        Convention: if parent_id is omitted but after_id refers to an
        existing task, treat the new task as a SIBLING (not a child) —
        explicit parent_id is required to nest.

        Dedupe: if an OPEN task (pending or in_progress) with a similar
        description already exists under the same parent, this is a no-op
        and the existing task is returned. Three-tier check:
          1. Exact normalized match → duplicate.
          2. SequenceMatcher ratio ≥ HIGH_RATIO → duplicate regardless
             of verb (minor rewording only).
          3. SequenceMatcher ratio ≥ MID_RATIO AND leading non-stopword
             matches → duplicate (catches "Create ASCII Pyramid Script"
             vs "Create a functional ASCII pyramid script"). The verb
             check stops "Verify..." and "Generate..." from collapsing
             into a "Create..." sibling — different leading verb is
             treated as evidence the tasks are doing different work,
             even when the noun phrase is identical.
        """
        normalized = _norm_desc(description)
        new_verb = _leading_verb(description)
        for existing in self.tasks:
            if (
                existing.parent_id != parent_id
                or existing.status not in ("pending", "in_progress")
            ):
                continue
            existing_norm = _norm_desc(existing.description)
            if existing_norm == normalized:
                return existing
            if not existing_norm or not normalized:
                continue
            ratio = _ratio(existing_norm, normalized)
            if ratio >= 0.92:
                return existing  # very close — minor rewording only
            if (
                ratio >= 0.78
                and _leading_verb(existing.description) == new_verb
            ):
                return existing  # paraphrase under the same verb
        new_id = max((t.id for t in self.tasks), default=0) + 1
        new_task = Task(id=new_id, description=description, parent_id=parent_id)
        idx = next((i for i, t in enumerate(self.tasks) if t.id == after_id), None)
        if idx is None:
            self.tasks.append(new_task)
        else:
            self.tasks.insert(idx + 1, new_task)
        return new_task

    def children_of(self, task_id: int) -> list:
        """Return the list of tasks whose parent_id == task_id, in plan
        order. Used by the renderer to walk the tree."""
        return [t for t in self.tasks if t.parent_id == task_id]

    def roots(self) -> list:
        """Return the top-level (parent_id is None) tasks, in plan order.
        The renderer starts from these and recurses via children_of."""
        return [t for t in self.tasks if t.parent_id is None]

    def is_complete(self) -> bool:
        return all(t.status in ("done", "skipped") for t in self.tasks)

    def progress(self):
        """Return (done_or_skipped_count, total_count)."""
        done = sum(1 for t in self.tasks if t.status in ("done", "skipped"))
        return done, len(self.tasks)
