"""
Purple Builder Agent - A2A Server Entry Point
Competitive agent for the build_what_i_mean benchmark.
"""

import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_agent_card(host: str, port: int, card_url: str | None = None) -> AgentCard:
    url = card_url or f"http://{host}:{port}/"

    skill = AgentSkill(
        id="block-builder",
        name="Block Structure Builder",
        description=(
            "Constructs block structures in a 9x9x9 grid from natural-language "
            "instructions. Capable of pragmatic reasoning about ambiguous instructions, "
            "speaker modeling, and strategic clarification questioning."
        ),
        tags=["building", "blocks", "spatial-reasoning", "pragmatics"],
        examples=[
            "Place a red block in each corner of the grid.",
            "Build a tower of 3 blue blocks in the center.",
        ],
    )

    return AgentCard(
        name="Purple Builder Agent",
        description=(
            "A competitive purple agent for the build_what_i_mean benchmark. "
            "Excels at interpreting natural-language building instructions, "
            "modeling speaker reliability, and strategic clarification."
        ),
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )


def main():
    parser = argparse.ArgumentParser(description="Purple Builder Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9018, help="Port to bind to")
    parser.add_argument("--card-url", default=None, help="Public URL for agent card")
    args = parser.parse_args()

    agent_card = build_agent_card(args.host, args.port, args.card_url)

    task_store = InMemoryTaskStore()
    executor = Executor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    logger.info(f"Starting Purple Builder Agent on {args.host}:{args.port}")
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
