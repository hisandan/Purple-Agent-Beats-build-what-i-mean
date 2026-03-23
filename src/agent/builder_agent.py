"""
BuilderAgent - Core agent logic for the build_what_i_mean benchmark.

Uses LLM-powered spatial reasoning with pragmatic speaker modeling
to interpret building instructions and construct block structures.
"""

import json
import logging
import os
import re

import litellm

from agent.grid import (
    VALID_X, VALID_Z, VALID_Y, VALID_COLORS,
    parse_blocks, blocks_to_string, normalize_block_set,
)
from agent.instruction_parser import parse_message, detect_ambiguity_type
from agent.speaker_model import SpeakerModel

logger = logging.getLogger(__name__)

# LLM Configuration
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4o")
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.0"))

SYSTEM_PROMPT = """You are an expert block-building agent operating on a 9×9 grid.

## COORDINATE SYSTEM
- Grid: 9×9 cells in the x-z plane
- Origin (0,0,0) is the center ("middle square" / "highlighted square")
- Valid X: [-400, -300, -200, -100, 0, 100, 200, 300, 400]
- Valid Z: [-400, -300, -200, -100, 0, 100, 200, 300, 400]
- Y (height): 50 (ground), 150, 250, 350, 450 (+100 per stacked block)

## DIRECTIONS (CRITICAL — get these right)
- "to the right" / "right" = +X (increasing X)
- "to the left" / "left" = -X (decreasing X)
- "in front of" / "front" = +Z (increasing Z). This is HORIZONTAL, not stacking.
- "behind" / "back" = -Z (decreasing Z). This is HORIZONTAL, not stacking.
- "on top of" = +Y (same X,Z position, increasing Y)

## CORNERS (all at ground level y=50)
- bottom-left = (-400, 50, 400)
- bottom-right = (400, 50, 400)
- top-left = (-400, 50, -400)
- top-right = (400, 50, -400)

## EDGES
- Left edge: x=-400 (varying z)
- Right edge: x=400 (varying z)
- Top/back edge: z=-400 (varying x)
- Bottom/front edge: z=400 (varying x)

## AVAILABLE COLORS
Red, Blue, Green, Yellow, Purple, Orange, White, Black, Brown, Pink, Grey, Cyan

## WORKED EXAMPLES

Example 1: "Place a red block in each corner. Then put a green block on top of each red block."
START: (empty)
→ Red at (-400,50,400), (400,50,400), (-400,50,-400), (400,50,-400)
→ Green at (-400,150,400), (400,150,400), (-400,150,-400), (400,150,-400)
Note: "on top" = same x,z, y+100.

Example 2: "Stack two red blocks directly in front of the green blocks."
START: Green,0,50,0; Green,0,150,0
→ Keep green. "in front of" = +z. Green at z=0, so red at z=100.
→ Red at (0,50,100), (0,150,100)

Example 3: "Build a row of three purple blocks, starting at the origin going left."
START: (empty)
→ Row starts AT x=0 (the origin), goes left (-x): (0,50,0), (-100,50,0), (-200,50,0)

Example 4: "Place a yellow block to the left of the red block, then place a blue block to the left of the yellow one."
START: Red,0,50,0
→ Yellow at (-100,50,0). Blue goes left of WHERE YELLOW WAS PLACED: (-200,50,0).
Note: "the yellow one" = the block you just placed, not the original.

Example 5: "Stack three yellow blocks to the left of the existing purple stack. Build a blue stack in front of the yellow one."
START: Purple,0,50,0; Purple,0,150,0; Purple,0,250,0
→ Yellow at (-100,50,0), (-100,150,0), (-100,250,0). "in front of yellow" = z+100.
→ Blue stack at (-100,50,100)... but how many? Count not specified — match adjacent height (3) or use context.

Example 6: "Place nine purple blocks along the grid's left edge."
START: (empty)
→ Left edge = x=-400. Nine blocks along z: (-400,50,-400), (-400,50,-300), ..., (-400,50,400)

## CRITICAL RULES
1. Output MUST include ALL blocks — existing (start) AND new blocks
2. Use exact coordinate values from valid lists
3. Ground level is y=50; stack by adding +100 per block above
4. Colors must be capitalized
5. Think step-by-step about spatial positions before giving coordinates
6. "in front of" = +Z (horizontal), NEVER stacking on top
7. "behind" = -Z (horizontal)
8. Chain references: "the X one" / "the block you just placed" = most recently placed position
9. "middle square" / "highlighted square" = origin (0,50,0). Row starting there starts AT x=0.
10. When a color is not specified for new blocks, use the same color as the most related existing blocks

## OUTPUT FORMAT
Respond with ONLY a JSON object:
{
  "reasoning": "your step-by-step spatial reasoning",
  "blocks": [
    {"color": "Red", "x": 0, "y": 50, "z": 0},
    ...
  ]
}

Include ALL blocks that should be on the grid after following the instruction."""


QUESTION_DECISION_PROMPT = """You are evaluating whether a building instruction is ambiguous enough to warrant asking a clarification question.

## CONTEXT
- Asking a question costs -5 points
- Building correctly earns +10, incorrectly costs -10
- So asking is worth it only if you'd likely guess wrong (>50% chance of error)

## SPEAKER CONTEXT
{speaker_context}

## INSTRUCTION
{instruction}

## EXISTING BLOCKS ON GRID
{start_blocks}

## YOUR TASK
Analyze:
1. Is the instruction fully specified (clear color, quantity, position)?
2. If ambiguous, what is ambiguous (color? number? position?)?
3. Given the speaker context, would you likely guess correctly?

Respond with ONLY a JSON object:
{
  "is_ambiguous": true/false,
  "ambiguity_type": "color_under" | "number_under" | "position_under" | "fully_spec",
  "confidence_without_asking": 0.0-1.0,
  "should_ask": true/false,
  "question": "the question to ask if should_ask is true"
}"""


class BuilderAgent:
    """
    Stateful agent that processes building instructions across a game session.
    Maintains speaker models and conversation history.
    """

    def __init__(self):
        self.speaker_model = SpeakerModel()
        self.conversation_history: list[dict] = []
        self.current_speaker: str | None = None
        self.current_instruction: str | None = None
        self.current_start_blocks: list[tuple[str, int, int, int]] = []
        self.current_ambiguity_type: str = "fully_spec"
        self.used_conservative: bool = False
        self.last_was_ask: bool = False
        self.pending_answer: str | None = None
        self.round_count: int = 0
        self.game_count: int = 0

    async def process_message(self, text: str) -> str:
        """Process a message from the green agent and return a response."""
        parsed = parse_message(text)

        # Handle new game transition
        if parsed.is_new_game:
            self.game_count += 1
            self.speaker_model.reset()
            self.conversation_history.clear()
            self.round_count = 0
            logger.info(f"New game started (game #{self.game_count})")
            return "[BUILD];"  # Acknowledge - green agent will send the real task next

        # Handle feedback from previous round
        if parsed.is_feedback:
            self._process_feedback(parsed)
            return "[BUILD];"  # Acknowledge - green agent will send next instruction

        # Handle answer to our question
        if parsed.is_answer:
            self.pending_answer = parsed.answer_text
            logger.info(f"Received answer: {self.pending_answer}")
            # Now build using the answer as additional context
            return await self._build_with_context()

        # Handle invalid format feedback
        if "Invalid response format" in text:
            # Retry with the current instruction
            if self.current_instruction:
                return await self._build_with_context()
            return "[BUILD];"

        # New instruction
        if parsed.instruction_text:
            self.round_count += 1
            self.current_speaker = parsed.speaker_name
            self.current_instruction = parsed.instruction_text
            self.current_start_blocks = parsed.start_blocks
            self.pending_answer = None
            self.last_was_ask = False

            # Detect ambiguity
            self.current_ambiguity_type = detect_ambiguity_type(
                parsed.instruction_text, parsed.start_blocks
            )

            logger.info(
                f"Round {self.round_count}: Speaker={self.current_speaker}, "
                f"Ambiguity={self.current_ambiguity_type}, "
                f"Instruction={parsed.instruction_text[:100]}"
            )

            # Decide whether to ask a question
            should_ask = self.speaker_model.should_ask_question(
                self.current_speaker, self.current_ambiguity_type
            )

            if should_ask and self.current_ambiguity_type != "fully_spec":
                question = await self._generate_question()
                if question:
                    self.last_was_ask = True
                    return f"[ASK];{question}"

            # Build directly
            return await self._build_with_context()

        # Fallback - try to build with whatever we have
        logger.warning(f"Unrecognized message format: {text[:200]}")
        return "[BUILD];"

    def _process_feedback(self, parsed):
        """Update speaker model based on feedback."""
        if self.current_speaker and parsed.feedback_correct is not None:
            self.speaker_model.record_result(
                speaker_name=self.current_speaker,
                was_correct=parsed.feedback_correct,
                was_ambiguous=self.current_ambiguity_type != "fully_spec",
                used_conservative=self.used_conservative,
            )
            logger.info(
                f"Feedback for {self.current_speaker}: "
                f"correct={parsed.feedback_correct}, "
                f"score={parsed.feedback_score}, "
                f"total={parsed.feedback_total}"
            )

    async def _generate_question(self) -> str | None:
        """Generate a strategic clarification question using LLM."""
        speaker = self.speaker_model.get_speaker(self.current_speaker)
        speaker_context = (
            f"Speaker '{self.current_speaker}' - "
            f"rounds seen: {speaker.total_rounds}, "
            f"reliability: {'unknown' if speaker.ambiguous_rounds < 2 else ('reliable' if speaker.is_likely_reliable else 'unreliable')}"
        )

        start_str = blocks_to_string(self.current_start_blocks) if self.current_start_blocks else "(empty grid)"

        prompt = QUESTION_DECISION_PROMPT.format(
            speaker_context=speaker_context,
            instruction=self.current_instruction,
            start_blocks=start_str,
        )

        try:
            response = await litellm.acompletion(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You analyze building instructions for ambiguity."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=500,
            )

            result = json.loads(response.choices[0].message.content)

            if result.get("should_ask") and result.get("question"):
                return result["question"]

        except Exception as e:
            logger.error(f"Error generating question: {e}")

        return None

    async def _build_with_context(self) -> str:
        """Build the structure using LLM reasoning."""
        if not self.current_instruction:
            return "[BUILD];"

        # Build the context for the LLM
        start_str = blocks_to_string(self.current_start_blocks) if self.current_start_blocks else "(no blocks)"

        # Add speaker model context for ambiguous instructions
        speaker_hint = ""
        if self.current_speaker and self.current_ambiguity_type != "fully_spec":
            pref = self.speaker_model.get_interpretation_preference(
                self.current_speaker, self.current_ambiguity_type
            )
            speaker = self.speaker_model.get_speaker(self.current_speaker)

            if speaker.ambiguous_rounds >= 2:
                if pref == "conservative":
                    speaker_hint = (
                        "\n\nSPEAKER CONTEXT: This speaker tends to be consistent/literal. "
                        "For ambiguous colors, prefer using the same color as existing blocks. "
                        "For ambiguous quantities, prefer the smaller/literal number."
                    )
                else:
                    speaker_hint = (
                        "\n\nSPEAKER CONTEXT: This speaker is unpredictable with ambiguous instructions. "
                        "For ambiguous colors, they might mean a different color than existing blocks. "
                        "For ambiguous quantities, they might mean more blocks than literally stated."
                    )
            self.used_conservative = (pref == "conservative")
        else:
            self.used_conservative = True  # default

        # Build the user prompt
        answer_context = ""
        if self.pending_answer:
            answer_context = f"\n\nCLARIFICATION ANSWER: I asked a question and received: \"{self.pending_answer}\"\nUse this information to resolve any ambiguity."

        user_prompt = f"""INSTRUCTION: {self.current_instruction}

EXISTING BLOCKS ON GRID (START_STRUCTURE): {start_str}
{speaker_hint}{answer_context}

Remember: Output ALL blocks that should be on the grid after following the instruction (including the start blocks if they should remain). Respond with ONLY the JSON object."""

        content = ""
        try:
            response = await litellm.acompletion(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            blocks = result.get("blocks", [])
            if not blocks:
                logger.warning("LLM returned no blocks")
                if self.current_start_blocks:
                    return "[BUILD];" + blocks_to_string(self.current_start_blocks)
                return "[BUILD];"

            # Convert to block tuples and validate
            validated_blocks = []
            for block in blocks:
                color = block.get("color", "").capitalize()
                x = int(block.get("x", 0))
                y = int(block.get("y", 50))
                z = int(block.get("z", 0))

                if color.upper() not in VALID_COLORS:
                    logger.warning(f"Invalid color: {color}")
                    continue
                if x not in VALID_X:
                    x = min(VALID_X, key=lambda v: abs(v - x))
                if z not in VALID_Z:
                    z = min(VALID_Z, key=lambda v: abs(v - z))
                if y not in VALID_Y:
                    y = min(VALID_Y, key=lambda v: abs(v - y))

                validated_blocks.append((color, x, y, z))

            # Remove duplicates preserving order
            seen = set()
            unique_blocks = []
            for block in validated_blocks:
                key = f"{block[0]},{block[1]},{block[2]},{block[3]}"
                if key not in seen:
                    seen.add(key)
                    unique_blocks.append(block)

            block_str = blocks_to_string(unique_blocks)
            return f"[BUILD];{block_str}"

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from LLM: {e}")
            return self._fallback_parse(content)

        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            if self.current_start_blocks:
                return "[BUILD];" + blocks_to_string(self.current_start_blocks)
            return "[BUILD];"

    def _fallback_parse(self, content: str) -> str:
        """Try to extract block coordinates from a non-JSON LLM response."""
        # Try to find coordinate patterns like Color,x,y,z
        pattern = r'(Red|Blue|Green|Yellow|Purple),\s*(-?\d+),\s*(\d+),\s*(-?\d+)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            blocks = []
            for color, x, y, z in matches:
                blocks.append((color.capitalize(), int(x), int(y), int(z)))
            return f"[BUILD];{blocks_to_string(blocks)}"

        # Last resort - return start blocks
        if self.current_start_blocks:
            return "[BUILD];" + blocks_to_string(self.current_start_blocks)
        return "[BUILD];"
