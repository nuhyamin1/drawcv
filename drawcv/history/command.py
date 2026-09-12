"""Command pattern interfaces and base classes for undo/redo."""

from __future__ import annotations
from abc import ABC, abstractmethod


class Command(ABC):
    """Abstract base class representing an undoable and redoable document action."""

    def __init__(self, description: str = ""):
        self.description = description

    @abstractmethod
    def execute(self) -> None:
        """Execute the command for the first time."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Revert the state modifications performed by this command."""
        pass

    def redo(self) -> None:
        """Re-apply the state modifications. Default implementation delegates to execute()."""
        self.execute()

    def is_noop(self) -> bool:
        """Determine if this command produced no actual state mutation."""
        return False


class CompoundCommand(Command):
    """A batch sequence of commands executed, undone, and redone atomically as a single unit."""

    def __init__(self, commands: list[Command] | None = None, description: str = "Batch"):
        super().__init__(description=description)
        self.commands: list[Command] = list(commands) if commands is not None else []

    def add(self, command: Command) -> None:
        """Append a command to this compound batch."""
        if not command.is_noop():
            self.commands.append(command)

    def execute(self) -> None:
        """Execute all commands forward."""
        for cmd in self.commands:
            cmd.execute()

    def undo(self) -> None:
        """Undo all commands in strict reverse order."""
        for cmd in reversed(self.commands):
            cmd.undo()

    def redo(self) -> None:
        """Re-apply all commands forward."""
        for cmd in self.commands:
            cmd.redo()

    def is_noop(self) -> bool:
        """A compound command is a no-op if it contains no commands or all its children are no-ops."""
        return len(self.commands) == 0 or all(cmd.is_noop() for cmd in self.commands)
