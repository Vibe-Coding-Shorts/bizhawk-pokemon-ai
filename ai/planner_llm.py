"""
planner_llm.py
==============
High-level goal planner that uses Claude (claude-opus-4-6) to decide which
strategic goal the RL agent should pursue next.

The planner is called periodically (e.g. every N environment steps or at the
end of each episode) rather than every frame, because LLM inference is slow
relative to the emulator frame rate.

Architecture
------------
  GameState (summary) ──► [LLM prompt] ──► goal string
  goal string ──► env.set_goal(goal) ──► reward function changes

The prompt is designed so Claude produces one of the 8 goal tokens:
  explore | battle | catch_pokemon | train_levels |
  heal | use_item | progress_story | idle

Claude also returns a brief reasoning string for debugging.

Usage
-----
    planner = LLMPlanner(api_key="sk-ant-...")
    goal, reasoning = await planner.get_goal(game_state, recent_actions)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Optional

import anthropic

from environment.memory_reader import GameState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid goal tokens (must match PokemonBlueEnv.GOALS)
# ---------------------------------------------------------------------------
VALID_GOALS = [
    "explore",
    "battle",
    "catch_pokemon",
    "train_levels",
    "heal",
    "use_item",
    "progress_story",
    "idle",
]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a strategic planning AI for a Pokémon Blue playthrough.
Your role is to analyse the current game state and decide the single best
high-level goal for the RL agent to pursue next.

You must respond with ONLY a valid JSON object in this exact format:
{
  "goal": "<one of the valid goals listed below>",
  "reasoning": "<1–2 sentence explanation>"
}

Valid goals and when to choose them:
- "explore"         : Move to new areas, discover the map, reach the next town.
                      Choose when few maps have been visited or no battle is active.
- "battle"          : Engage wild Pokémon or trainers for experience.
                      Choose when levels are low or money is low.
- "catch_pokemon"   : Attempt to catch a Pokémon to fill the Pokédex.
                      Choose when Pokédex is incomplete and party has room.
- "train_levels"    : Grind battles to raise party levels before a gym.
                      Choose when the next badge requires higher-level Pokémon.
- "heal"            : Reach the nearest Pokémon Center to restore HP.
                      Choose when player HP fraction < 30%%.
- "use_item"        : Use a held item (Potion, Repel, etc.) from the bag.
                      Choose when HP is low but a Center is far away.
- "progress_story"  : Follow the main story path, trigger key events.
                      Choose after acquiring a badge or reaching a new town.
- "idle"            : Do nothing (used when game is in a menu or cutscene).
                      Choose when text_on_screen is active.

Rules:
1. ONLY output valid JSON. No markdown, no extra text.
2. The "goal" field must be exactly one of the 8 tokens listed above.
3. Choose "heal" if HP fraction < 0.30 and not currently in battle.
4. Choose "idle" if a text box appears to be on screen.
5. Never recommend catching when party is full (6 Pokémon).
"""

USER_PROMPT_TEMPLATE = """\
Current game state:
{state_summary}

Recent agent actions (last {n_recent} steps):
{actions_summary}

Training progress:
- Total maps discovered: {maps_visited}
- Total tiles visited: {coords_visited}
- Episodes completed: {episodes}
- Current goal: {current_goal}

What goal should the agent pursue next?
"""


class LLMPlanner:
    """
    Uses Claude claude-opus-4-6 to select high-level goals for the RL agent.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key. Falls back to the ANTHROPIC_API_KEY env var.
    model : str
        Claude model ID (default: claude-opus-4-6 for strongest reasoning).
    cooldown_steps : int
        Minimum environment steps between planner calls (rate-limiting).
    max_recent_actions : int
        How many recent action indices to include in the prompt.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-6",
        cooldown_steps: int = 500,
        max_recent_actions: int = 20,
    ):
        self.model = model
        self.cooldown_steps = cooldown_steps
        self._last_call_step = -cooldown_steps  # allow first call immediately

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        self._recent_actions: deque[int] = deque(maxlen=max_recent_actions)
        self._current_goal = "explore"
        self._call_count = 0
        self._total_latency = 0.0

        # Context for prompt
        self._maps_visited = 0
        self._coords_visited = 0
        self._episodes = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def should_plan(self, current_step: int) -> bool:
        """Return True if enough steps have passed since the last plan."""
        return (current_step - self._last_call_step) >= self.cooldown_steps

    def record_action(self, action: int) -> None:
        """Log an action taken by the RL agent (for prompt context)."""
        self._recent_actions.append(action)

    def update_context(
        self,
        maps_visited: int = 0,
        coords_visited: int = 0,
        episodes: int = 0,
    ) -> None:
        """Update training statistics used in the prompt."""
        self._maps_visited = maps_visited
        self._coords_visited = coords_visited
        self._episodes = episodes

    def get_goal(
        self,
        state: GameState,
        current_step: int = 0,
        force: bool = False,
    ) -> tuple[str, str]:
        """
        Synchronously ask Claude for a goal recommendation.

        Returns
        -------
        (goal, reasoning)
            goal      – one of the 8 valid goal tokens
            reasoning – Claude's brief explanation
        """
        if not force and not self.should_plan(current_step):
            return self._current_goal, "Cooldown active – reusing previous goal."

        self._last_call_step = current_step

        prompt = self._build_prompt(state)
        t0 = time.perf_counter()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = self._extract_text(response)
            goal, reasoning = self._parse_response(raw_text)

        except anthropic.APIError as exc:
            logger.error("Claude API error: %s – keeping current goal.", exc)
            return self._current_goal, f"API error: {exc}"

        latency = time.perf_counter() - t0
        self._call_count += 1
        self._total_latency += latency

        logger.info(
            "LLM plan #%d | goal=%s | latency=%.2fs | reason: %s",
            self._call_count, goal, latency, reasoning,
        )

        self._current_goal = goal
        return goal, reasoning

    @property
    def current_goal(self) -> str:
        return self._current_goal

    @property
    def avg_latency(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._total_latency / self._call_count

    # ------------------------------------------------------------------ #
    # Prompt construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, state: GameState) -> str:
        action_names = ["Up", "Down", "Left", "Right", "A", "B", "Start", "NoOp"]
        actions_list = [action_names[a] for a in self._recent_actions]
        if actions_list:
            actions_summary = ", ".join(actions_list[-20:])
        else:
            actions_summary = "(no actions recorded yet)"

        return USER_PROMPT_TEMPLATE.format(
            state_summary=state.to_planner_summary(),
            n_recent=len(self._recent_actions),
            actions_summary=actions_summary,
            maps_visited=self._maps_visited,
            coords_visited=self._coords_visited,
            episodes=self._episodes,
            current_goal=self._current_goal,
        )

    # ------------------------------------------------------------------ #
    # Response parsing                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(response) -> str:
        """Pull the text block out of a Claude Messages response."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text.strip()
        return ""

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, str]:
        """
        Parse the JSON response from Claude.
        Falls back gracefully if the format is unexpected.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        try:
            data = json.loads(cleaned)
            goal = data.get("goal", "explore").lower().strip()
            reasoning = data.get("reasoning", "No reasoning provided.")

            if goal not in VALID_GOALS:
                logger.warning("Invalid goal '%s' from LLM, defaulting to 'explore'", goal)
                goal = "explore"

            return goal, reasoning

        except json.JSONDecodeError:
            # Try to extract a goal token from the raw text
            for token in VALID_GOALS:
                if token in raw.lower():
                    logger.warning("Fell back to regex goal extraction: %s", token)
                    return token, raw[:200]
            return "explore", f"Parse error – raw response: {raw[:200]}"


# ---------------------------------------------------------------------------
# Rule-based fallback planner (no API key required)
# ---------------------------------------------------------------------------

class RuleBasedPlanner:
    """
    Simple heuristic planner that mirrors the LLM planner's interface.
    Useful for offline testing or when an API key is unavailable.
    """

    def __init__(self):
        self._current_goal = "explore"
        self._call_count = 0

    @property
    def current_goal(self) -> str:
        return self._current_goal

    @property
    def avg_latency(self) -> float:
        return 0.0

    def should_plan(self, current_step: int) -> bool:
        return current_step % 200 == 0

    def record_action(self, action: int) -> None:
        pass

    def update_context(self, **kwargs) -> None:
        pass

    def get_goal(
        self,
        state: GameState,
        current_step: int = 0,
        force: bool = False,
    ) -> tuple[str, str]:
        self._call_count += 1

        # Heal if HP is critically low
        if state.hp_fraction < 0.25 and not state.is_in_battle:
            self._current_goal = "heal"
            return "heal", "HP critical – need healing."

        # Idle if text is on screen
        if getattr(state, "text_on_screen", 0):
            self._current_goal = "idle"
            return "idle", "Text box detected."

        # Battle if in battle already
        if state.is_in_battle:
            self._current_goal = "battle"
            return "battle", "Currently in battle."

        # Catch Pokémon if Pokédex is sparse and party not full
        if state.pokedex_caught < 10 and state.party_count < 6:
            self._current_goal = "catch_pokemon"
            return "catch_pokemon", "Pokédex needs more entries."

        # Train if levels are low
        if state.max_party_level < (state.badges * 10 + 5):
            self._current_goal = "train_levels"
            return "train_levels", "Levels too low for next badge."

        # Progress story after each badge
        if state.badges < 8:
            self._current_goal = "progress_story"
            return "progress_story", "Pursuing next gym badge."

        # Default: explore
        self._current_goal = "explore"
        return "explore", "Default – exploring the world."
