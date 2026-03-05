# Pokémon Blue AI – BizHawk + RL + LLM Planner

A research-grade prototype that trains an AI agent to play **Pokémon Blue** using:

- **BizHawk** emulator as the game engine
- **Lua script** as the emulator interface
- **TCP sockets** for real-time communication
- **Stable Baselines3 (PPO/DQN/A2C)** for low-level gameplay control
- **Claude claude-opus-4-6** as a high-level strategic planner

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        BizHawk                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  bizhawk_script.lua                                 │    │
│  │  • Reads memory (HP, position, battle state, etc.)  │    │
│  │  • Sends JSON game state → Python via TCP           │    │
│  │  • Receives action index ← Python via TCP           │    │
│  │  • Applies joypad input / save-state resets         │    │
│  └──────────────────────┬──────────────────────────────┘    │
└─────────────────────────│───────────────────────────────────┘
                          │  TCP  (localhost:65432)
                          │  JSON state ↓  / action index ↑
┌─────────────────────────▼───────────────────────────────────┐
│                     Python AI System                        │
│                                                             │
│  EmulatorBridge ──► PokemonBlueEnv (Gymnasium)              │
│        │                   │                                │
│        │            reward function                         │
│        │            observation builder                     │
│        │                   │                                │
│        └──────────────► RLAgent (PPO / DQN / A2C)          │
│                            │                                │
│                     [every N steps]                         │
│                            │                                │
│                    LLMPlanner (Claude)                      │
│                    • reads game state                       │
│                    • outputs goal token                     │
│                    • goal → reward shaping                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pokemon-ai/
├── emulator/
│   └── bizhawk_script.lua          # Lua script loaded inside BizHawk
├── environment/
│   ├── __init__.py
│   ├── communication.py            # TCP bridge (EmulatorBridge)
│   ├── memory_reader.py            # GameState dataclass + memory map
│   └── pokemon_blue_env.py         # Gymnasium environment (PokemonBlueEnv)
├── ai/
│   ├── __init__.py
│   ├── rl_agent.py                 # RLAgent wrapper (SB3 PPO/DQN/A2C)
│   └── planner_llm.py              # LLMPlanner (Claude) + RuleBasedPlanner
├── training/
│   ├── __init__.py
│   └── train.py                    # Main training entry point
├── visualization/
│   ├── __init__.py
│   ├── heatmap.py                  # Exploration heatmap
│   └── dashboard.py                # Training stats dashboard
├── models/                         # Saved model checkpoints
├── logs/                           # TensorBoard logs, CSVs, plots
├── savestates/                     # BizHawk save states (.State files)
├── config.py                       # Central configuration
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Install BizHawk

**Windows (recommended – best Lua support):**

1. Download BizHawk from https://tasvideos.org/BizHawk/ReleaseHistory
   Use version **2.9.1+** for best Lua/luasocket support.
2. Extract to `C:\BizHawk\` (or any path without spaces).
3. Run `EmuHawk.exe` once to create default config files.
4. Go to **Config → Cores → Game Boy / GBC → Gambatte** (recommended core).

**Linux (via Wine or native build):**
```bash
# Via Mono (Linux native)
sudo apt install mono-complete
# Download BizHawk Linux release from the same URL
```

### 2. Obtain Pokémon Blue ROM

You need a legally-obtained Pokémon Blue (USA) ROM file.

Load it in BizHawk: **File → Open ROM → select .gb file**.

### 3. Create a Save State at Game Start

1. In BizHawk, start a new game in Pokémon Blue.
2. Name your character, choose a starter Pokémon.
3. Once you regain control in Pallet Town: **File → Save State → Save Named State**
4. Save as `savestates/start.State` (relative to the project root).

The Python training script uses this state to reset episodes.

### 4. Install Python Dependencies

```bash
cd pokemon-ai
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Set Your Anthropic API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."   # Linux / macOS
set ANTHROPIC_API_KEY=sk-ant-api03-...         # Windows CMD
```

If no API key is set the system falls back to `RuleBasedPlanner` automatically.

### 6. Load the Lua Script in BizHawk

1. Open BizHawk with Pokémon Blue loaded.
2. **Tools → Lua Console**.
3. Click **Open Script** → navigate to `emulator/bizhawk_script.lua`.
4. Click **Run** (▶). You should see:
   ```
   [LUA] Pokémon Blue AI script starting…
   [LUA] Waiting for Python server on 127.0.0.1:65432
   ```

### 7. Start Training

```bash
# Basic training (PPO, 500k steps)
python -m training.train

# With options
python -m training.train \
  --algo PPO \
  --steps 1000000 \
  --llm-model claude-opus-4-6 \
  --planner-interval 500

# Resume from checkpoint
python -m training.train --checkpoint models/PPO_step_0100000

# Use SB3 native loop
python -m training.train --sb3
```

Once Python starts, the Lua console in BizHawk will show:
```
[LUA] Connected to Python AI server at 127.0.0.1:65432
```

---

## Component Details

### 1. BizHawk Lua Script (`emulator/bizhawk_script.lua`)

Reads memory directly from the Game Boy RAM and communicates with Python over TCP.

**Key memory addresses:**

| Address | Size | Description |
|---------|------|-------------|
| `0xD362` | 1B | Player X tile coordinate |
| `0xD361` | 1B | Player Y tile coordinate |
| `0xD35E` | 1B | Current map ID |
| `0xD057` | 1B | Battle type (0=none, 1=wild, 2=trainer) |
| `0xD163` | 1B | Party Pokémon count |
| `0xD16C–D` | 2B | Party slot 1 current HP (big-endian u16) |
| `0xD18D–E` | 2B | Party slot 1 max HP (big-endian u16) |
| `0xD18C` | 1B | Party slot 1 level |
| `0xCFE6–7` | 2B | Enemy HP current (battle only) |
| `0xD347–9` | 3B | Player money (BCD encoded) |
| `0xD356` | 1B | Badge bits (bit N = gym N+1 cleared) |

**Action mapping:**

| Index | Button | Use |
|-------|--------|-----|
| 0 | Up | Move north |
| 1 | Down | Move south |
| 2 | Left | Move west |
| 3 | Right | Move east |
| 4 | A | Confirm / interact / attack |
| 5 | B | Cancel / run |
| 6 | Start | Open menu |
| 7 | (none) | No-op |

### 2. Communication Protocol

```
Lua → Python :  {"frame":1234,"player_x":5,"player_y":8,"map_id":0,...}\n
Python → Lua :  4\n     (action index)
               "SAVE"\n  (save BizHawk state)
               "RESET"\n (reload episode savestate)
```

### 3. Gymnasium Environment (`environment/pokemon_blue_env.py`)

**Observation vector (19 dims, float32):**
```
[x/255, y/255, map_id/255, in_battle, hp_frac, enemy_hp_frac,
 money_norm, badges/8, dex_seen/151, dex_caught/151,
 avg_level/100,
 goal_one_hot × 8]
```

**Reward shaping:**

| Event | Reward | Goal bonus |
|-------|--------|------------|
| New tile visited | +0.02 | ×2 if goal=explore |
| New map discovered | +1.0 | – |
| Win battle | +2.0 | ×2 if goal=battle |
| Catch Pokémon | +3.0 | ×2 if goal=catch |
| Level up (per level) | +1.5 | ×2 if goal=train |
| Earn badge | +10.0 | – |
| Heal at Pokémon Center | +0.5 | ×2 if goal=heal |
| Faint | −5.0 | – |
| Lose battle | −3.0 | – |
| Stuck (20+ steps same tile) | −0.01/step | – |

### 4. LLM Planner (`ai/planner_llm.py`)

Uses Claude claude-opus-4-6 with adaptive thinking. Called every 500 steps (configurable).

**Available goals:**
```
explore | battle | catch_pokemon | train_levels |
heal | use_item | progress_story | idle
```

**Example Claude response:**
```json
{
  "goal": "explore",
  "reasoning": "HP is above 50%, party level is reasonable for early game,
                and there are many unexplored areas nearby."
}
```

The planner falls back to `RuleBasedPlanner` when no API key is set.

### 5. Visualisation

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# Generated images in logs/
#   exploration_heatmap.png
#   reward_curve.png
#   badge_pokedex.png
#   episode_length.png
#   goal_reward_dist.png
```

---

## Performance Tips

### Fast-Forward Mode

Enable in BizHawk: **Config → Speed/Skip → Unthrottle** (or hold Tab).

Add to Lua script for automatic fast-forward:
```lua
client.speedmode("maximum")
```

### Frame Skipping

In `emulator/bizhawk_script.lua`:
```lua
CONFIG.frame_skip = 4   -- only send state every 4 frames
CONFIG.action_hold = 8  -- hold each button for 8 frames
```

### Save State Resets

Episode resets are near-instant: `savestate.load(slot)` restores the game in ~1 frame. Zero overhead vs real-time environments.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `[LUA] Could not connect` | Start Python training script first, then run Lua |
| `Timed out waiting for emulator` | Check BizHawk is running with the Lua script active |
| `ANTHROPIC_API_KEY not set` | Set env var; falls back to rule-based planner |
| Low FPS | Increase `frame_skip`, enable fast-forward in BizHawk |
| Agent only presses A | Increase `ent_coef` in PPO, or add diversity reward |
| OOM on GPU | Use `--device cpu` or reduce `batch_size` |

---

## References

- BizHawk: https://tasvideos.org/BizHawk
- Stable Baselines3: https://stable-baselines3.readthedocs.io
- Gymnasium: https://gymnasium.farama.org
- Anthropic Claude API: https://docs.anthropic.com
- Pokémon Red/Blue RAM map: https://datacrystal.romhacking.net/wiki/Pokémon_Red/Blue:RAM_map

---

## License

MIT License – see `LICENSE` for details.
