"""
Speaker reliability model.

Tracks per-speaker accuracy history and infers reliability.
A reliable speaker (Pia pattern) always uses the "b" variant for ambiguous
instructions - meaning same color as existing blocks for color-underspecified
tasks, and fewer blocks for number-underspecified tasks.

An unreliable speaker (Lisa pattern) uses mixed a/b variants unpredictably.

The agent learns this from feedback and adapts its interpretation strategy.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SpeakerHistory:
    """Track a speaker's instruction history and accuracy."""
    name: str
    total_rounds: int = 0
    correct_rounds: int = 0
    ambiguous_rounds: int = 0
    ambiguous_correct_conservative: int = 0  # correct when choosing conservative/same
    ambiguous_correct_liberal: int = 0  # correct when choosing liberal/different
    # Track consecutive patterns
    recent_results: list[bool] = field(default_factory=list)
    # Track what happened with ambiguous instructions
    ambiguous_outcomes: list[str] = field(default_factory=list)  # "conservative_correct", "liberal_correct", "conservative_wrong", "liberal_wrong"

    @property
    def accuracy(self) -> float:
        if self.total_rounds == 0:
            return 0.5
        return self.correct_rounds / self.total_rounds

    @property
    def reliability_score(self) -> float:
        """
        Score from 0 to 1 indicating how reliable/consistent this speaker is.
        High score = reliable (Pia-like, prefer conservative/same interpretation).
        Low score = unreliable (Lisa-like, unpredictable).
        """
        if self.ambiguous_rounds < 2:
            # Not enough data - use overall accuracy as proxy
            return 0.5

        # If conservative interpretations consistently work, speaker is reliable
        if self.ambiguous_correct_conservative + self.ambiguous_correct_liberal == 0:
            return 0.5

        conservative_rate = self.ambiguous_correct_conservative / (
            self.ambiguous_correct_conservative + self.ambiguous_correct_liberal + 0.01
        )
        return conservative_rate

    @property
    def is_likely_reliable(self) -> bool:
        """Whether this speaker appears to follow the reliable (Pia) pattern."""
        return self.reliability_score > 0.6

    @property
    def is_likely_unreliable(self) -> bool:
        """Whether this speaker appears to follow the unreliable (Lisa) pattern."""
        return self.reliability_score < 0.4 and self.ambiguous_rounds >= 3


class SpeakerModel:
    """
    Maintains models of speaker reliability across the evaluation.
    Resets between seeds (games) since speaker names are re-randomized.
    """

    def __init__(self):
        self._speakers: dict[str, SpeakerHistory] = {}
        self._current_speaker: str | None = None

    def reset(self):
        """Reset all speaker models (called between seeds/games)."""
        self._speakers.clear()
        self._current_speaker = None
        logger.info("Speaker models reset for new game")

    def get_speaker(self, name: str) -> SpeakerHistory:
        if name not in self._speakers:
            self._speakers[name] = SpeakerHistory(name=name)
        return self._speakers[name]

    def set_current_speaker(self, name: str):
        self._current_speaker = name

    def record_result(self, speaker_name: str, was_correct: bool, was_ambiguous: bool = False,
                      used_conservative: bool = False):
        """Record the outcome of a round for speaker modeling."""
        speaker = self.get_speaker(speaker_name)
        speaker.total_rounds += 1
        if was_correct:
            speaker.correct_rounds += 1
        speaker.recent_results.append(was_correct)
        if len(speaker.recent_results) > 10:
            speaker.recent_results.pop(0)

        if was_ambiguous:
            speaker.ambiguous_rounds += 1
            if was_correct and used_conservative:
                speaker.ambiguous_correct_conservative += 1
                speaker.ambiguous_outcomes.append("conservative_correct")
            elif was_correct and not used_conservative:
                speaker.ambiguous_correct_liberal += 1
                speaker.ambiguous_outcomes.append("liberal_correct")
            elif not was_correct and used_conservative:
                speaker.ambiguous_outcomes.append("conservative_wrong")
            else:
                speaker.ambiguous_outcomes.append("liberal_wrong")

        logger.info(
            f"Speaker '{speaker_name}': rounds={speaker.total_rounds}, "
            f"accuracy={speaker.accuracy:.2f}, "
            f"reliability={speaker.reliability_score:.2f}, "
            f"ambiguous_rounds={speaker.ambiguous_rounds}"
        )

    def should_ask_question(self, speaker_name: str, ambiguity_type: str) -> bool:
        """
        Decide whether asking a clarification question is worth the -5 cost.

        Strategy:
        - For fully specified instructions: never ask
        - For ambiguous instructions with a known reliable speaker: don't ask,
          use the conservative interpretation
        - For ambiguous instructions with unknown/unreliable speaker: ask if we
          have few data points (expected value of information > 5 points)
        - After enough observations of unreliable speaker, stop asking since
          we can't predict them anyway (and questions cost too much)
        """
        if ambiguity_type == "fully_spec":
            return False

        speaker = self.get_speaker(speaker_name)

        # Early rounds with this speaker - asking has high information value
        if speaker.ambiguous_rounds < 2 and speaker.total_rounds < 4:
            # Only ask for the very first ambiguous instruction per speaker
            # to efficiently learn their type
            return speaker.ambiguous_rounds == 0

        # If speaker is reliable, we know the pattern - no need to ask
        if speaker.is_likely_reliable:
            return False

        # If speaker is unreliable, asking is expensive and we can't predict
        # their pattern anyway. Better to guess.
        if speaker.is_likely_unreliable:
            return False

        # Uncertain about speaker - occasionally ask
        return speaker.ambiguous_rounds < 3

    def get_interpretation_preference(self, speaker_name: str, ambiguity_type: str) -> str:
        """
        Get preferred interpretation for ambiguous instructions.
        Returns 'conservative' or 'liberal'.

        For color_under: conservative = same color as existing blocks
        For number_under: conservative = fewer blocks (literal interpretation)
        """
        speaker = self.get_speaker(speaker_name)

        if speaker.is_likely_reliable:
            return "conservative"

        if speaker.is_likely_unreliable:
            # For unreliable speakers, look at recent pattern
            recent_outcomes = speaker.ambiguous_outcomes[-3:]
            conservative_wins = sum(1 for o in recent_outcomes if "conservative_correct" in o)
            liberal_wins = sum(1 for o in recent_outcomes if "liberal_correct" in o)
            if liberal_wins > conservative_wins:
                return "liberal"
            return "conservative"

        # Default to conservative (more common pattern overall)
        return "conservative"
