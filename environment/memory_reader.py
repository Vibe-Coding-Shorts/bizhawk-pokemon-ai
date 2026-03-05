"""
memory_reader.py
================
Typed game-state container used across the reward function, RL environment,
and LLM planner. Decoupled from the raw dict that comes out of the emulator
so that any future state changes only touch one file.

Pokémon Blue memory map reference
----------------------------------
Address   Size  Description
-------   ----  -----------
0xD362    1     Player X tile coordinate (column on current map)
0xD361    1     Player Y tile coordinate (row on current map)
0xD35E    1     Current map ID
0xD057    1     Battle type  (0=none, 1=wild, 2=trainer)
0xD163    1     Number of Pokémon in party (0–6)
0xD16C–D  2     Party slot 1 current HP (big-endian)
0xD18D–E  2     Party slot 1 max HP     (big-endian)
0xD18C    1     Party slot 1 level
0xD164–9  1ea   Party species IDs (6 bytes)
0xCFE6–7  2     Enemy current HP (big-endian, only valid during battle)
0xCFEA–B  2     Enemy max HP    (big-endian, only valid during battle)
0xD347–9  3     Player money (BCD, 6 decimal digits)
0xD356    1     Badge bits    (bit N = gym N+1 beaten)
0xD30A    1     Pokédex "seen" count
0xD2F7    1     Pokédex "caught" count
0xDA40    1     In-game clock hours
0xDA41    1     In-game clock minutes

Note: party slots 2–6 follow the same layout offset by 44 (0x2C) bytes each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Map ID → human-readable area name (partial list)
# ---------------------------------------------------------------------------
MAP_NAMES: dict[int, str] = {
    0:   "Pallet Town",
    1:   "Viridian City",
    2:   "Pewter City",
    3:   "Cerulean City",
    4:   "Lavender Town",
    5:   "Vermilion City",
    6:   "Celadon City",
    7:   "Fuchsia City",
    8:   "Cinnabar Island",
    9:   "Indigo Plateau",
    10:  "Saffron City",
    11:  "Unused",
    12:  "Route 1",
    13:  "Route 2",
    14:  "Route 3",
    15:  "Route 4",
    16:  "Route 5",
    17:  "Route 6",
    18:  "Route 7",
    19:  "Route 8",
    20:  "Route 9",
    21:  "Route 10",
    22:  "Route 11",
    23:  "Route 12",
    24:  "Route 13",
    25:  "Route 14",
    26:  "Route 15",
    27:  "Route 16",
    28:  "Route 17",
    29:  "Route 18",
    30:  "Route 19",
    31:  "Route 20",
    32:  "Route 21",
    33:  "Route 22",
    34:  "Route 23",
    35:  "Route 24",
    36:  "Route 25",
    37:  "Red's House 1F",
    38:  "Red's House 2F",
    39:  "Blue's House",
    40:  "Oak's Lab",
    # ... (abbreviated)
    200: "Pokémon Tower 1F",
    201: "Pokémon Tower 2F",
    210: "Silph Co 1F",
    220: "Victory Road 1F",
    230: "Mt. Moon 1F",
    240: "Cerulean Cave 1F",
}

POKEMON_NAMES: dict[int, str] = {
    1: "Bulbasaur", 4: "Charmander", 7: "Squirtle",
    25: "Pikachu", 39: "Jigglypuff", 52: "Meowth",
    54: "Psyduck", 63: "Abra", 74: "Geodude",
    92: "Gastly", 129: "Magikarp", 133: "Eevee",
    143: "Snorlax", 149: "Dragonite", 150: "Mewtwo",
    151: "Mew",
}

ITEM_NAMES: dict[int, str] = {
    1: "Master Ball", 2: "Ultra Ball", 3: "Great Ball", 4: "Poké Ball",
    5: "Town Map", 6: "Bicycle", 10: "Potion", 11: "Antidote",
    12: "Burn Heal", 13: "Ice Heal", 14: "Awakening", 15: "Parlyz Heal",
    16: "Full Restore", 17: "Max Potion", 18: "Hyper Potion", 19: "Super Potion",
    20: "Full Heal", 21: "Revive", 22: "Max Revive",
}


@dataclass
class GameState:
    """
    Canonical game-state object passed between environment, reward function,
    and LLM planner.
    """
    player_x: int = 0
    player_y: int = 0
    map_id: int = 0
    in_battle: int = 0           # 0 = overworld, 1 = wild, 2 = trainer
    player_hp: int = 0
    player_hp_max: int = 1
    enemy_hp_cur: int = 0
    enemy_hp_max: int = 0
    money: int = 0
    badges: int = 0
    pokedex_seen: int = 0
    pokedex_caught: int = 0
    party_levels: List[int] = field(default_factory=list)
    party_count: int = 0

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def map_name(self) -> str:
        return MAP_NAMES.get(self.map_id, f"Map {self.map_id}")

    @property
    def hp_fraction(self) -> float:
        """Current HP as fraction [0, 1]."""
        if self.player_hp_max <= 0:
            return 0.0
        return self.player_hp / self.player_hp_max

    @property
    def is_in_battle(self) -> bool:
        return self.in_battle > 0

    @property
    def is_wild_battle(self) -> bool:
        return self.in_battle == 1

    @property
    def is_trainer_battle(self) -> bool:
        return self.in_battle == 2

    @property
    def max_party_level(self) -> int:
        if not self.party_levels:
            return 0
        return max(self.party_levels)

    @property
    def avg_party_level(self) -> float:
        if not self.party_levels:
            return 0.0
        return sum(self.party_levels) / len(self.party_levels)

    @property
    def coord(self) -> tuple[int, int, int]:
        """(map_id, x, y) – unique position key."""
        return (self.map_id, self.player_x, self.player_y)

    def to_dict(self) -> dict:
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "map_id": self.map_id,
            "map_name": self.map_name,
            "in_battle": self.in_battle,
            "player_hp": self.player_hp,
            "player_hp_max": self.player_hp_max,
            "hp_fraction": round(self.hp_fraction, 3),
            "enemy_hp_cur": self.enemy_hp_cur,
            "enemy_hp_max": self.enemy_hp_max,
            "money": self.money,
            "badges": self.badges,
            "pokedex_seen": self.pokedex_seen,
            "pokedex_caught": self.pokedex_caught,
            "party_levels": self.party_levels,
            "party_count": self.party_count,
            "max_party_level": self.max_party_level,
        }

    def to_planner_summary(self) -> str:
        """
        Compact natural-language summary for the LLM planner system prompt.
        """
        battle_str = (
            "Not in battle"
            if not self.is_in_battle
            else f"In {'wild' if self.is_wild_battle else 'trainer'} battle "
                 f"(enemy HP {self.enemy_hp_cur}/{self.enemy_hp_max})"
        )
        levels_str = ", ".join(str(lv) for lv in self.party_levels) or "none"
        return (
            f"Location: {self.map_name} (map {self.map_id}), "
            f"tile ({self.player_x}, {self.player_y})\n"
            f"Battle status: {battle_str}\n"
            f"Player HP: {self.player_hp}/{self.player_hp_max} "
            f"({self.hp_fraction:.0%})\n"
            f"Party levels: [{levels_str}] ({self.party_count} Pokémon)\n"
            f"Badges: {self.badges}/8 | "
            f"Pokédex: {self.pokedex_seen} seen / {self.pokedex_caught} caught\n"
            f"Money: ¥{self.money:,}"
        )
