"""Conversation state management for the agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A single message in a conversation."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ConversationState:
    """State for a single conversation session."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages, optionally limited to the most recent."""
        if limit is None:
            return self.messages
        return self.messages[-limit:]

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.updated_at = datetime.now(timezone.utc).isoformat()


class ConversationStateManager:
    """Manager for conversation states."""

    def __init__(self, max_sessions: int = 1000):
        self._states: Dict[str, ConversationState] = {}
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> ConversationState:
        """Get an existing state or create a new one."""
        if session_id not in self._states:
            if len(self._states) >= self._max_sessions:
                oldest = min(self._states.values(), key=lambda s: s.updated_at)
                del self._states[oldest.session_id]
            self._states[session_id] = ConversationState(session_id=session_id)
        return self._states[session_id]

    def get(self, session_id: str) -> Optional[ConversationState]:
        """Get an existing state."""
        return self._states.get(session_id)

    def delete(self, session_id: str) -> None:
        """Delete a state."""
        self._states.pop(session_id, None)


_global_manager: Optional[ConversationStateManager] = None


def get_state_manager() -> ConversationStateManager:
    """Get the global conversation state manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ConversationStateManager()
    return _global_manager
