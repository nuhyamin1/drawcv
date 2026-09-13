"""History and Undo/Redo manager for DrawCV scenes."""

from __future__ import annotations
from contextlib import contextmanager
from typing import Generator

from drawcv.history.command import Command, CompoundCommand


class HistoryManager:
    """Manages reversible document history stacks, batching, and command suspension."""

    def __init__(self, max_undo: int = 100):
        self.max_undo = max_undo
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._suspended_count: int = 0
        self._batch_stack: list[CompoundCommand] = []

    @property
    def is_suspended(self) -> bool:
        """True if history recording is currently suspended."""
        return self._suspended_count > 0

    @contextmanager
    def suspended(self) -> Generator[None, None, None]:
        """Context manager to temporarily suspend history tracking."""
        self._suspended_count += 1
        try:
            yield
        finally:
            self._suspended_count -= 1

    @contextmanager
    def batch(self, name: str = "Batch") -> Generator[CompoundCommand, None, None]:
        """Context manager to group multiple operations into an atomic compound command."""
        compound = CompoundCommand(description=name)
        self._batch_stack.append(compound)
        try:
            yield compound
        except Exception:
            with self.suspended():
                compound.undo()
            raise
        else:
            self._batch_stack.pop()
            if not compound.is_noop():
                self.record(compound)
            return
        finally:
            if self._batch_stack and self._batch_stack[-1] is compound:
                self._batch_stack.pop()

    def record(self, command: Command) -> None:
        """Record an executed command into the undo history.
        
        Guarantees:
        - Suppressed if history is suspended.
        - Discards no-op commands.
        - Invalidates the redo stack upon recording any new non-noop action.
        - Groups into active batch if a batch context is open.
        - Respects max_undo capacity limits.
        """
        if self.is_suspended:
            return

        if command.is_noop():
            return

        if self._batch_stack:
            self._batch_stack[-1].add(command)
            return

        self._redo_stack.clear()
        self._undo_stack.append(command)
        if len(self._undo_stack) > self.max_undo:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Undo the most recent command, restoring prior state in-place.
        
        Returns:
            True if an action was undone; False if the undo stack was empty.
        """
        if not self._undo_stack:
            return False

        cmd = self._undo_stack[-1]
        with self.suspended():
            cmd.undo()
        self._undo_stack.pop()
        self._redo_stack.append(cmd)
        return True

    def redo(self) -> bool:
        """Re-apply the most recently undone command.
        
        Returns:
            True if an action was redone; False if the redo stack was empty.
        """
        if not self._redo_stack:
            return False

        cmd = self._redo_stack[-1]
        with self.suspended():
            cmd.redo()
        self._redo_stack.pop()
        self._undo_stack.append(cmd)
        return True

    @property
    def can_undo(self) -> bool:
        """Whether there are commands available to undo."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Whether there are commands available to redo."""
        return len(self._redo_stack) > 0

    @property
    def undo_count(self) -> int:
        """Number of undoable operations on stack."""
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        """Number of redoable operations on stack."""
        return len(self._redo_stack)

    @property
    def last_undo_description(self) -> str:
        """Description of the command that would be undone next."""
        return self._undo_stack[-1].description if self._undo_stack else ""

    @property
    def last_redo_description(self) -> str:
        """Description of the command that would be redone next."""
        return self._redo_stack[-1].description if self._redo_stack else ""

    def clear(self) -> None:
        """Clear both undo and redo stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()
