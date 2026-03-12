"""
Instruction parser - extracts structured information from green agent messages.
Parses the task description, speaker info, start structure, and instruction text.
"""

import re
import logging
from dataclasses import dataclass

from agent.grid import parse_blocks

logger = logging.getLogger(__name__)


@dataclass
class ParsedInstruction:
    """Structured representation of a green agent message."""
    raw_text: str
    grid_context: str
    speaker_name: str
    start_structure_raw: str
    start_blocks: list[tuple[str, int, int, int]]
    instruction_text: str
    is_new_game: bool
    is_feedback: bool
    is_answer: bool
    feedback_correct: bool | None
    feedback_score: int | None
    feedback_total: int | None
    answer_text: str | None


def parse_message(text: str) -> ParsedInstruction:
    """Parse a message from the green agent into structured form."""
    is_new_game = "A new task is starting" in text or "play the game again" in text
    is_feedback = text.strip().startswith("Feedback:")
    is_answer = text.strip().startswith("Answer:")
    is_invalid = "Invalid response format" in text

    feedback_correct = None
    feedback_score = None
    feedback_total = None
    answer_text = None

    if is_feedback:
        feedback_correct = "Correct!" in text or "correct!" in text.lower()
        score_match = re.search(r"Round score:\s*([+-]?\d+)", text)
        total_match = re.search(r"Total score:\s*([+-]?\d+)", text)
        if score_match:
            feedback_score = int(score_match.group(1))
        if total_match:
            feedback_total = int(total_match.group(1))

    if is_answer:
        # Extract the answer text after "Answer:"
        answer_text = text.split("Answer:", 1)[1].strip()
        # Remove the penalty note
        answer_text = re.sub(r"\s*\(-?\d+ points? for asking\)\s*", "", answer_text).strip()

    # Parse task instruction
    grid_context = ""
    speaker_name = ""
    start_structure_raw = ""
    instruction_text = ""

    if "[TASK_DESCRIPTION]" in text:
        # Extract grid context
        td_match = re.search(r"\[TASK_DESCRIPTION\](.*?)(?:\[SPEAKER\]|\[START_STRUCTURE\])", text, re.DOTALL)
        if td_match:
            grid_context = td_match.group(1).strip()

        # Extract speaker
        speaker_match = re.search(r"\[SPEAKER\]\s*(\w+)", text)
        if speaker_match:
            speaker_name = speaker_match.group(1).strip()

        # Extract start structure
        start_match = re.search(r"\[START_STRUCTURE\]\s*(.*?)(?:\n|$)", text)
        if start_match:
            start_structure_raw = start_match.group(1).strip()

        # Extract instruction text - everything after the last tag line
        lines = text.split("\n")
        instruction_lines = []
        past_tags = False
        for line in lines:
            if past_tags:
                instruction_lines.append(line)
            elif line.strip() and not any(
                tag in line for tag in ["[TASK_DESCRIPTION]", "[SPEAKER]", "[START_STRUCTURE]"]
            ):
                # If we already saw tags and this line has no tags, it's instruction
                if speaker_name or start_structure_raw:
                    instruction_lines.append(line)
            if "[START_STRUCTURE]" in line:
                past_tags = True

        instruction_text = "\n".join(instruction_lines).strip()

        # If instruction_text is empty, try to get the last meaningful line
        if not instruction_text:
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and not any(
                    tag in stripped
                    for tag in ["[TASK_DESCRIPTION]", "[SPEAKER]", "[START_STRUCTURE]"]
                ):
                    instruction_text = stripped
                    break

    start_blocks = parse_blocks(start_structure_raw)

    return ParsedInstruction(
        raw_text=text,
        grid_context=grid_context,
        speaker_name=speaker_name,
        start_structure_raw=start_structure_raw,
        start_blocks=start_blocks,
        instruction_text=instruction_text,
        is_new_game=is_new_game,
        is_feedback=is_feedback,
        is_answer=is_answer,
        feedback_correct=feedback_correct,
        feedback_score=feedback_score,
        feedback_total=feedback_total,
        answer_text=answer_text,
    )


def detect_ambiguity_type(instruction: str, start_blocks: list[tuple[str, int, int, int]]) -> str:
    """
    Detect if an instruction is ambiguous and what type.

    Returns: 'fully_spec', 'color_under', or 'number_under'
    """
    instruction_lower = instruction.lower()

    # Check for color underspecification
    # Color is underspecified when instruction mentions blocks without specifying color
    # and there are existing blocks of a specific color
    has_existing_blocks = len(start_blocks) > 0

    if has_existing_blocks:
        # Words that suggest adding blocks without specifying a color
        color_words = {"red", "blue", "green", "yellow", "purple"}
        mentioned_colors = {w for w in color_words if w in instruction_lower}

        # Check for quantifiers without explicit color
        has_quantity = bool(re.search(
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b',
            instruction_lower
        ))

        # If instruction mentions adding blocks but uses ambiguous color references
        if "block" in instruction_lower and not mentioned_colors and has_quantity:
            return "color_under"

        # Check for number underspecification
        # "add more blocks", "stack blocks", without specific count
        if "block" in instruction_lower and not has_quantity:
            return "number_under"

    return "fully_spec"
