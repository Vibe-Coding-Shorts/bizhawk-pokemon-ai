# Pokémon Blue AI — BizHawk 2.11 + RL + LLM Planner

An AI agent that learns to play **Pokémon Blue** using reinforcement learning
with a Claude LLM acting as a high-level strategic planner.

- **BizHawk 2.11** emulator (Game Boy, Gambatte core)
- **Lua script** reads RAM and handles input inside BizHawk
- **TCP socket** carries game state and actions between BizHawk and Python
- **Stable Baselines3** (PPO / DQN / A2C) for low-level control
- **Claude claude-opus-4-6** selects high-level goals (explore, battle, heal, …)

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  BizHawk 2.11  (EmuHawk.exe)                     │
│                                                  │
│  bizhawk_script.lua                              │
│   • reads Game Boy RAM every 4 frames            │
│   • comm.socketServerSend(json_state)  ──────┐   │
│   • comm.socketServerResponse()        ◄─────┘   │
│   • applies joypad / save-state resets           │
└───────────────────────────┬──────────────────────┘
                            │  TCP  127.0.0.1:65432
              JSON state ▼  │  ▲  "$N action" reply
┌───────────────────────────┴──────────────────────┐
│  Python AI System                                │
│                                                  │
│  EmulatorBridge                                  │
│   • TCP server, recv_loop sends "$N msg" replies │
│   • state queue / pending-action slot            │
│        │                                         │
│  PokemonBlueEnv  (Gymnasium)                     │
│   • observation builder (19-dim float32)         │
│   • reward shaping                               │
│        │                                         │
│  RLAgent  (SB3 PPO)                              │
│   • MLP policy                                   │
│   • predicts action 0–7                          │
│        │  every 500 steps                        │
│  LLMPlanner  (Claude claude-opus-4-6)            │
│   • reads GameState summary                      │
│   • returns goal token → reward multipliers      │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
bizhawk-pokemon-ai/
├── emulator/
│   └── bizhawk_script.lua          # Lua script loaded inside BizHawk
├── environment/
│   ├── communication.py            # TCP bridge (EmulatorBridge)
│   ├── memory_reader.py            # GameState dataclass
│   └── pokemon_blue_env.py         # Gymnasium environment
├── ai/
│   ├── rl_agent.py                 # SB3 PPO/DQN/A2C wrapper
│   └── planner_llm.py              # Claude LLM planner + rule-based fallback
├── training/
│   └── train.py                    # Entry point
├── visualization/
│   ├── heatmap.py
│   └── dashboard.py
├── models/                         # Saved checkpoints
├── logs/                           # TensorBoard, CSV, plots
├── savestates/                     # (not used – slot-based saves instead)
├── config.py
└── requirements.txt
```

---

## Setup

### 1. Download BizHawk 2.11

1. Go to <https://tasvideos.org/BizHawk/ReleaseHistory> and download
   **BizHawk 2.11** for Windows.
2. Extract to a path **without spaces**, e.g. `C:\BizHawk-2.11\`.
3. Run `EmuHawk.exe` once to generate config files.
4. **Config → Cores → Game Boy / GBC → Gambatte** (required for Pokémon Blue).

> **Linux / macOS** – BizHawk runs via Wine. Replace `EmuHawk.exe` with
> `wine EmuHawk.exe` in all commands below.

### 2. Get a Pokémon Blue ROM

You need a legally-obtained Pokémon Blue (USA) `.gb` ROM.

In BizHawk: **File → Open ROM** → select the `.gb` file.

### 3. Create the Starting Save State

The Lua script resets episodes by reloading **BizHawk save slot 1**.

1. Start a new Pokémon Blue game.
2. Name your character, pick a starter Pokémon, and walk out of the house
   into Pallet Town (so you have full control).
3. In BizHawk: **File → Save State → Save to slot 1** (or press `F1`).

> Every episode reset reloads slot 1 instantly.
> Re-save slot 1 any time you want to change the starting point.

### 4. Create the Startup Batch File

Create `start_bizhawk.bat` next to `EmuHawk.exe` (adjust path as needed):

```bat
@echo off
start "" "C:\BizHawk-2.11\EmuHawk.exe" --socket_ip=127.0.0.1 --socket_port=65432
```

The `--socket_ip` and `--socket_port` flags tell BizHawk to connect to the
Python TCP server automatically when it starts.
**Do not launch BizHawk normally** — these flags are required for the Lua
`comm` API to work.

### 5. Install Python Dependencies

Requires Python **3.10+**.

```bash
cd bizhawk-pokemon-ai
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 6. Set Your Anthropic API Key (optional)

```bash
# Windows CMD
set ANTHROPIC_API_KEY=sk-ant-api03-...

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-..."

# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

If no key is set the system falls back to the built-in rule-based planner
(no LLM calls, slightly simpler heuristics).

---

## Running

**Order matters: start Python first, then BizHawk.**

### Step 1 — Start Python training

```bash
# Basic (PPO, 500 k steps)
python -m training.train

# With options
python -m training.train --algo PPO --steps 1000000

# Resume from checkpoint
python -m training.train --checkpoint models/PPO_step_0100000

# No LLM (rule-based planner only)
python -m training.train --planner-interval 999999999
```

You should see:

```
INFO  EmulatorBridge listening on 127.0.0.1:65432
INFO  Waiting for BizHawk to connect… (run the Lua script in BizHawk)
```

### Step 2 — Launch BizHawk

Double-click `start_bizhawk.bat`
(or run `EmuHawk.exe --socket_ip=127.0.0.1 --socket_port=65432` directly).

BizHawk opens with Pokémon Blue already loaded (if you set a default ROM in
the config, otherwise open the ROM manually: **File → Open ROM**).

### Step 3 — Load the Lua Script

1. In BizHawk: **Tools → Lua Console**.
2. Click **Open Script** → navigate to `emulator/bizhawk_script.lua`.
3. Click **Run** (▶ button or **Script → Run**).

The Lua Console should show:

```
[LUA] Pokémon Blue AI script starting…
[LUA] Connecting to Python server on port 65432
[LUA] Start the Python training script first, then this script.
[LUA] Running main loop…
```

Python will log:

```
INFO  Emulator connected from ('127.0.0.1', ...)
INFO  BizHawk connected.
INFO  Training starts | algo=PPO | max_steps=500000
```

Training is now running. BizHawk plays the game automatically.

---

## Communication Protocol Detail

```
Lua → Python    plain JSON, newline-terminated
                {"frame":1234,"player_x":5,"player_y":3,"map_id":0,...}

Python → Lua    length-prefixed (required by BizHawk 2.6.2+):
                "$1 7"       → action 7 (NoOp)
                "$1 4"       → action 4 (A button)
                "$5 RESET"   → reload save slot 1
                "$4 SAVE"    → save to slot 1
```

BizHawk's `comm.socketServerResponse()` reads the `$N ` prefix, then reads
exactly N bytes, and returns that string to Lua.  Without the prefix the call
fails silently and returns an empty string.

### Socket flags vs Lua API

| What | Correct |
|------|---------|
| Connection setup | `--socket_ip=127.0.0.1 --socket_port=65432` CLI flags |
| Timeout | `comm.socketServerSetTimeout(3000)` in Lua |
| Send state | `comm.socketServerSend(json)` |
| Receive action | `comm.socketServerResponse()` |
| **Never call** | `comm.socketServerSetIp()` / `comm.socketServerSetPort()` |

Calling `socketServerSetIp/Port` from Lua after the CLI connection is
established disrupts the socket and causes `socketServerResponse()` to fail.

---

## Action Space

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

---

## Observation Space (19-dim float32)

```
player_x / 255          player_y / 255         map_id / 255
in_battle (0/1)         player_hp_fraction     enemy_hp_fraction
money / 99999           badges / 8
pokedex_seen / 151      pokedex_caught / 151
avg_party_level / 100
goal_one_hot[8]         (set by LLM planner)
```

---

## Reward Shaping

| Event | Reward | Goal bonus |
|-------|--------|------------|
| New tile visited | +0.02 | ×2 if goal=`explore` |
| New map discovered | +1.0 | – |
| Win battle | +2.0 | ×2 if goal=`battle` |
| Catch Pokémon | +3.0 | ×2 if goal=`catch_pokemon` |
| Level up (per level) | +1.5 | ×2 if goal=`train_levels` |
| Earn badge | +10.0 | – |
| Heal at Pokémon Center | +0.5 | ×2 if goal=`heal` |
| Faint | −5.0 | – |
| Lose battle | −3.0 | – |
| Stuck > 20 steps same tile | −0.01/step | – |
| Per-step idle penalty | −0.001 | – |

---

## LLM Planner Goals

Claude selects one goal every 500 steps (configurable via `--planner-interval`):

```
explore        battle         catch_pokemon   train_levels
heal           use_item       progress_story  idle
```

---

## Monitoring

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# Live logs
tail -f logs/training.log
```

Outputs in `logs/`:
- `training.log` — full log
- `episode_stats.csv` — per-episode metrics
- `exploration_heatmap.png` — tile visit heat map
- `tensorboard/` — SB3 training curves

---

## CLI Reference

```
python -m training.train [options]

  --algo      PPO | DQN | A2C            (default: PPO)
  --steps     INT                        (default: 500000)
  --checkpoint PATH                      resume from .zip checkpoint
  --host      STR                        (default: 127.0.0.1)
  --port      INT                        (default: 65432)
  --llm-model STR                        (default: claude-opus-4-6)
  --planner-interval INT                 steps between LLM calls (default: 500)
  --device    auto | cpu | cuda          (default: auto)
  --api-key   STR                        overrides ANTHROPIC_API_KEY env var
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `[LUA] Running main loop…` but no state arrives in Python | Make sure BizHawk was launched with `--socket_ip=127.0.0.1 --socket_port=65432` |
| `Timed out waiting for emulator after reset` | BizHawk must be running with the Lua script active; check slot 1 save exists |
| Lua console shows error `NLua … Invalid` | You called `socketServerSetIp/Port` from Lua — remove those calls |
| Python responses appear empty in Lua | Python is sending plain text; must use `$N payload` format |
| Very slow training | Enable fast-forward in BizHawk: **Config → Speed/Skip → Unthrottle** (or hold Tab) |
| Agent only presses A | Increase `ent_coef` in PPO config (entropy regularisation) |
| CUDA OOM | Add `--device cpu` or reduce `batch_size` in `config.py` |
| `ANTHROPIC_API_KEY not set` | Set env var; system falls back to rule-based planner automatically |

---

## Key RAM Addresses (Pokémon Blue USA)

| Address | Size | Description |
|---------|------|-------------|
| `0xD362` | 1 B | Player X tile |
| `0xD361` | 1 B | Player Y tile |
| `0xD35E` | 1 B | Map ID |
| `0xD057` | 1 B | Battle type (0=none 1=wild 2=trainer) |
| `0xD163` | 1 B | Party count |
| `0xD16C–D` | 2 B | Party slot 1 current HP (big-endian u16) |
| `0xD18D–E` | 2 B | Party slot 1 max HP |
| `0xD18C` | 1 B | Party slot 1 level |
| `0xCFE6–7` | 2 B | Enemy current HP |
| `0xD347–9` | 3 B | Money (BCD) |
| `0xD356` | 1 B | Badges (bit N = badge N+1) |
| `0xD30A` | 1 B | Pokédex seen count |
| `0xD2F7` | 1 B | Pokédex caught count |

Full RAM map: <https://datacrystal.romhacking.net/wiki/Pokémon_Red/Blue:RAM_map>

---

## References

- BizHawk Lua functions: <https://tasvideos.org/Bizhawk/LuaFunctions>
- BizHawk releases: <https://tasvideos.org/BizHawk/ReleaseHistory>
- Stable Baselines3: <https://stable-baselines3.readthedocs.io>
- Gymnasium: <https://gymnasium.farama.org>
- Anthropic Claude API: <https://docs.anthropic.com>

---

## License

MIT — see `LICENSE`.
