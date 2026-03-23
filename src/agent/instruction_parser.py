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

    Uses per-clause analysis: splits instruction into clauses and checks
    each one for missing color or count between the placing verb and noun.
    """
    instruction_lower = instruction.lower()
    color_words = {"red", "blue", "green", "yellow", "purple",
                   "orange", "white", "black", "brown", "pink",
                   "grey", "gray", "cyan"}

    # Split into clauses
    clauses = re.split(r'\.\s+|\bthen\b|\band\s+then\b|,\s*then\b', instruction_lower)

    placing_verbs = r'(?:stack|place|build|add|put|extend|create|make)'
    block_nouns = r'(?:blocks?|stack|tower|row|line|column)'

    has_missing_color = False
    has_missing_count = False

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # Find placing verb → block noun patterns
        for m in re.finditer(
            r'\b' + placing_verbs + r'\b(.*?)\b' + block_nouns + r'\b',
            clause,
        ):
            between = m.group(1)  # text BETWEEN verb and noun

            # Check for color between verb and noun
            has_color_between = any(
                re.search(r'\b' + c + r'\b', between) for c in color_words
            )
            if not has_color_between:
                has_missing_color = True

            # Check for count between verb and noun
            has_count_between = bool(re.search(
                r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b',
                between,
            ))
            # "a" or "an" implies count=1
            has_article = bool(re.search(r'\ba\b|\ban\b', between))

            if not has_count_between and not has_article:
                has_missing_count = True

        # Also check "Build a [color] stack/tower" without count
        stack_m = re.search(
            r'\b(?:build|make|create)\s+a\s+(?:\w+\s+)?(?:stack|tower)\b',
            clause,
        )
        if stack_m:
            after = clause[stack_m.end():]
            if not re.search(r'^\s*(?:of\s+)?(\d+|one|two|three|four|five)\b', after):
                has_missing_count = True

    if has_missing_color:
        return "color_under"
    if has_missing_count:
        return "number_under"
    return "fully_spec"
