"""
A2A Executor - Bridges A2A protocol with the BuilderAgent logic.
Manages per-conversation agent instances and task lifecycle.
"""

import logging
import traceback

from a2a.server.agent_execution import AgentExecutor
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task

from agent.builder_agent import BuilderAgent

logger = logging.getLogger(__name__)


def get_message_text(message) -> str:
    """Extract text content from an A2A message."""
    if message and message.parts:
        for part in message.parts:
            if hasattr(part, "root"):
                inner = part.root
            else:
                inner = part
            if isinstance(inner, TextPart):
                return inner.text
            if hasattr(inner, "text"):
                return inner.text
    return ""


class Executor(AgentExecutor):
    """Manages BuilderAgent instances and handles A2A task execution."""

    def __init__(self):
        self._agents: dict[str, BuilderAgent] = {}

    def _get_or_create_agent(self, context_id: str) -> BuilderAgent:
        if context_id not in self._agents:
            self._agents[context_id] = BuilderAgent()
            logger.info(f"Created new BuilderAgent for context {context_id}")
        return self._agents[context_id]

    async def execute(self, context, event_queue: EventQueue):
        context_id = context.context_id or "default"
        message = context.message
        text = get_message_text(message)

        logger.info(f"[{context_id}] Received message: {text[:200]}...")

        task = context.current_task
        if task is None:
            task = new_task(message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Processing instruction..."),
            )

            agent = self._get_or_create_agent(context_id)
            response = await agent.process_message(text)

            logger.info(f"[{context_id}] Response: {response[:200]}...")

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response))],
                name="builder_response",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"[{context_id}] Error: {e}\n{traceback.format_exc()}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {str(e)}"),
            )

    async def cancel(self, context, event_queue: EventQueue):
        raise UnsupportedOperationError("Cancel not supported")
