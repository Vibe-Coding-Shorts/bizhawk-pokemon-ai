-- =============================================================================
-- BizHawk Lua Script: Pokémon Blue AI Interface
-- =============================================================================
-- Runs inside BizHawk and acts as the bridge between the emulator and the
-- external Python AI system.
--
-- Communication: uses BizHawk's built-in comm API (no luasocket needed).
--   comm.socketServerSetPort(port)      -- set Python server port
--   comm.socketServerSetTimeout(ms)     -- connection/read timeout
--   comm.socketServerResponse(data)     -- send JSON state, receive action
--
-- Python runs a TCP server; BizHawk connects to it as a client.
--
-- Memory addresses for Pokémon Blue (Game Boy):
--   Player X:          0xD362
--   Player Y:          0xD361
--   Map ID:            0xD35E
--   In Battle:         0xD057  (0=none, 1=wild, 2=trainer)
--   Party Count:       0xD163
--   Party HP (slot 1 current): 0xD16C–0xD16D  (big-endian u16)
--   Party HP (slot 1 max):     0xD18D–0xD18E
--   Enemy HP (current):        0xCFE6–0xCFE7
--   Enemy HP (max):            0xCFEA–0xCFEB  (approx)
--   Player Money:      0xD347–0xD349  (BCD, 3 bytes)
--   Badge Count:       0xD356
--   Pokédex Seen:      0xD30A  (count)
--   Pokédex Caught:    0xD2F7  (count)
--   Game Clock (h):    0xDA40
--   Game Clock (m):    0xDA41
--   Party Levels:      0xD18C (slot1), 0xD1B8 (slot2), ...
-- =============================================================================

-- No require() needed – comm is a BizHawk built-in

-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------
local CONFIG = {
    host          = "127.0.0.1",
    port          = 65432,
    frame_skip    = 4,        -- send state every N frames
    action_hold   = 8,        -- hold each button for N frames
    fast_forward  = false,    -- toggle with hotkey
    debug         = true,
    savestate_slot = 1,
}

-- ---------------------------------------------------------------------------
-- Memory addresses (Pokémon Blue / Red share the same map)
-- ---------------------------------------------------------------------------
local ADDR = {
    player_x         = 0xD362,
    player_y         = 0xD361,
    map_id           = 0xD35E,
    in_battle        = 0xD057,
    party_count      = 0xD163,
    -- Party slot HP (current / max) – slot offsets of 44 bytes each
    party_hp_cur     = { 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248 },
    party_hp_max     = { 0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269 },
    party_level      = { 0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268 },
    party_species    = { 0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169 },
    enemy_hp_cur     = 0xCFE6,
    enemy_hp_max     = 0xCFEA,
    enemy_species    = 0xCFD8,
    money            = 0xD347,   -- 3 bytes BCD
    badges           = 0xD356,
    pokedex_seen     = 0xD30A,
    pokedex_caught   = 0xD2F7,
    clock_hours      = 0xDA40,
    clock_minutes    = 0xDA41,
    -- Warp / text flags
    text_on_screen   = 0xD730,
    overworld_loop   = 0xD72E,
    -- Items in bag (first 4 slots for quick snapshot)
    bag_item1        = 0xD31E,
    bag_item1_count  = 0xD31F,
    bag_item2        = 0xD320,
    bag_item2_count  = 0xD321,
}

-- ---------------------------------------------------------------------------
-- Button mapping  (BizHawk joypad key names)
-- ---------------------------------------------------------------------------
local BUTTONS = {
    "Up", "Down", "Left", "Right",
    "A", "B", "Start", "Select",
}

-- Action index → button name (matches Python action_space)
-- 0=Up 1=Down 2=Left 3=Right 4=A 5=B 6=Start 7=NoOp
local ACTION_MAP = {
    [0] = "Up",
    [1] = "Down",
    [2] = "Left",
    [3] = "Right",
    [4] = "A",
    [5] = "B",
    [6] = "Start",
    [7] = nil,   -- no-op
}

-- ---------------------------------------------------------------------------
-- Helper: read big-endian unsigned 16-bit value
-- ---------------------------------------------------------------------------
local function read_u16_be(addr)
    local hi = memory.read_u8(addr)
    local lo = memory.read_u8(addr + 1)
    return hi * 256 + lo
end

-- ---------------------------------------------------------------------------
-- Helper: decode 3-byte BCD money value
-- ---------------------------------------------------------------------------
local function read_money()
    local b1 = memory.read_u8(ADDR.money)
    local b2 = memory.read_u8(ADDR.money + 1)
    local b3 = memory.read_u8(ADDR.money + 2)
    -- Each byte encodes two BCD digits
    local function bcd(b) return (math.floor(b / 16) * 10) + (b % 16) end
    return bcd(b1) * 10000 + bcd(b2) * 100 + bcd(b3)
end

-- ---------------------------------------------------------------------------
-- Helper: count set bits (popcount) for badge/seen/caught bytes
-- ---------------------------------------------------------------------------
local function popcount_byte(b)
    local count = 0
    while b > 0 do
        count = count + (b % 2)
        b = math.floor(b / 2)
    end
    return count
end

-- Count total badges (8 bits across 1 byte)
local function count_badges()
    local b = memory.read_u8(ADDR.badges)
    return popcount_byte(b)
end

-- ---------------------------------------------------------------------------
-- Build game state table
-- ---------------------------------------------------------------------------
local function read_game_state()
    local party_count = memory.read_u8(ADDR.party_count)
    -- Clamp to valid range
    if party_count > 6 then party_count = 0 end

    -- Party HP / levels / species
    local party = {}
    for i = 1, math.max(party_count, 1) do
        local hp_cur = read_u16_be(ADDR.party_hp_cur[i])
        local hp_max = read_u16_be(ADDR.party_hp_max[i])
        local level  = memory.read_u8(ADDR.party_level[i])
        local species = memory.read_u8(ADDR.party_species[i])
        table.insert(party, {
            species = species,
            level   = level,
            hp_cur  = hp_cur,
            hp_max  = hp_max,
        })
    end

    -- Battle state
    local in_battle    = memory.read_u8(ADDR.in_battle)
    local enemy_hp_cur = 0
    local enemy_hp_max = 0
    local enemy_species = 0
    if in_battle > 0 then
        enemy_hp_cur  = read_u16_be(ADDR.enemy_hp_cur)
        enemy_hp_max  = read_u16_be(ADDR.enemy_hp_max)
        enemy_species = memory.read_u8(ADDR.enemy_species)
    end

    -- First bag items (quick snapshot)
    local bag = {
        { id = memory.read_u8(ADDR.bag_item1),  count = memory.read_u8(ADDR.bag_item1_count) },
        { id = memory.read_u8(ADDR.bag_item2),  count = memory.read_u8(ADDR.bag_item2_count) },
    }

    local state = {
        frame         = emu.framecount(),
        player_x      = memory.read_u8(ADDR.player_x),
        player_y      = memory.read_u8(ADDR.player_y),
        map_id        = memory.read_u8(ADDR.map_id),
        in_battle     = in_battle,
        party_count   = party_count,
        party         = party,
        enemy_hp_cur  = enemy_hp_cur,
        enemy_hp_max  = enemy_hp_max,
        enemy_species = enemy_species,
        money         = read_money(),
        badges        = count_badges(),
        pokedex_seen  = memory.read_u8(ADDR.pokedex_seen),
        pokedex_caught = memory.read_u8(ADDR.pokedex_caught),
        clock_hours   = memory.read_u8(ADDR.clock_hours),
        clock_minutes = memory.read_u8(ADDR.clock_minutes),
        text_on_screen = memory.read_u8(ADDR.text_on_screen),
        bag           = bag,
    }

    return state
end

-- ---------------------------------------------------------------------------
-- Serialise table to a compact JSON-like string (no external lib needed)
-- ---------------------------------------------------------------------------
local function to_json(val, depth)
    depth = depth or 0
    local t = type(val)

    if t == "nil"     then return "null"
    elseif t == "boolean" then return tostring(val)
    elseif t == "number"  then
        -- Handle integers vs floats cleanly
        if val == math.floor(val) then return tostring(math.floor(val))
        else return tostring(val) end
    elseif t == "string"  then
        -- Escape special chars
        val = val:gsub('\\', '\\\\'):gsub('"', '\\"')
                 :gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
        return '"' .. val .. '"'
    elseif t == "table" then
        -- Detect array vs object
        local is_array = (#val > 0)
        if is_array then
            local parts = {}
            for _, v in ipairs(val) do
                table.insert(parts, to_json(v, depth + 1))
            end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            local parts = {}
            for k, v in pairs(val) do
                table.insert(parts, '"' .. tostring(k) .. '":' .. to_json(v, depth + 1))
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    end
    return "null"
end

-- ---------------------------------------------------------------------------
-- Apply a button action for one frame
-- ---------------------------------------------------------------------------
local held_button   = nil
local hold_counter  = 0

local function apply_action(action_idx)
    local btn = ACTION_MAP[action_idx]
    held_button  = btn
    hold_counter = CONFIG.action_hold
end

local function step_input()
    local inputs = {}
    if hold_counter > 0 and held_button ~= nil then
        inputs[held_button] = true
        hold_counter = hold_counter - 1
    end
    joypad.set(inputs, 1)
end

-- ---------------------------------------------------------------------------
-- Save / Load state helpers
-- ---------------------------------------------------------------------------
local function save_state()
    savestate.saveslot(CONFIG.savestate_slot)
    if CONFIG.debug then
        console.log("[LUA] State saved to slot " .. CONFIG.savestate_slot)
    end
end

local function load_state()
    savestate.loadslot(CONFIG.savestate_slot)
    if CONFIG.debug then
        console.log("[LUA] State loaded from slot " .. CONFIG.savestate_slot)
    end
end

-- ---------------------------------------------------------------------------
-- Screen overlay (debug HUD)
-- ---------------------------------------------------------------------------
local function draw_hud(state)
    local x, y = 2, 2
    gui.text(x, y,      string.format("Map:%d  X:%d Y:%d", state.map_id, state.player_x, state.player_y), "white")
    gui.text(x, y + 10, string.format("Battle:%d  Money:%d", state.in_battle, state.money), "yellow")
    if state.party_count > 0 and state.party[1] then
        local p = state.party[1]
        gui.text(x, y + 20, string.format("Lv%d HP:%d/%d", p.level, p.hp_cur, p.hp_max), "lime")
    end
    gui.text(x, y + 30, string.format("Badges:%d  Seen:%d", state.badges, state.pokedex_seen), "cyan")
end

-- ---------------------------------------------------------------------------
-- BizHawk comm API setup
-- ---------------------------------------------------------------------------
-- BizHawk's comm socket must be enabled at launch via command line:
--   EmuHawk.exe --socket_ip=127.0.0.1 --socket_port=65432
--
-- DO NOT call comm.socketServerSetPort() here – that requires the socket
-- to already be initialised, which only happens via the CLI args above.
--
-- comm.socketServerSetTimeout(ms)   – how long to wait for a response
-- comm.socketServerResponse(data)   – send data, return response string
-- ---------------------------------------------------------------------------

-- Configure socket connection to Python server.
-- These calls set the IP/port so BizHawk 2.11+ does NOT require the
-- --socket_ip / --socket_port command-line flags at launch.
pcall(comm.socketServerSetIp, CONFIG.host)
pcall(comm.socketServerSetPort, CONFIG.port)

-- Wrap in pcall: BizHawk tries to connect immediately, which fails if Python
-- isn't running yet.  We catch the error so the script keeps running and
-- retries on every communicate() call (which is already wrapped in pcall).
local _ok, _err = pcall(comm.socketServerSetTimeout, 2000)
if not _ok then
    console.log("[LUA] Socket not ready (start Python server first): " .. tostring(_err))
end

-- Send game state JSON, receive action string from Python.
--
-- BizHawk's socket API (no-arg form):
--   comm.socketServerSend(data)    → send data to Python
--   comm.socketServerResponse()    → block until Python sends a line, return it
--
-- Protocol: Python sends an action first (including the initial no-op on
-- connect), then BizHawk reads it and sends the current game state.
local function communicate(state)
    -- 1. Read the action Python already sent (blocks up to the timeout)
    local ok_r, response = pcall(comm.socketServerResponse)
    if not ok_r or response == nil or response == "" then
        return nil   -- Python not ready yet
    end

    -- Strip whitespace / newline
    response = response:match("^%s*(.-)%s*$")

    -- Handle special commands BEFORE sending state back
    if response == "SAVE" then
        save_state()
        -- Send state so Python gets its next round-trip
        pcall(comm.socketServerSend, to_json(state) .. "\n")
        return nil
    elseif response == "LOAD" or response == "RESET" then
        load_state()
        -- After loading, send the (now-reset) state
        pcall(comm.socketServerSend, to_json(read_game_state()) .. "\n")
        return nil
    end

    -- 2. Send current game state to Python
    local ok_s, _ = pcall(comm.socketServerSend, to_json(state) .. "\n")
    if not ok_s then
        return nil
    end

    local action_idx = tonumber(response)
    if action_idx ~= nil then
        return math.floor(action_idx)
    end

    return nil  -- no-op
end

-- ---------------------------------------------------------------------------
-- Main loop  (loop-based: required in BizHawk 2.11 to keep the script alive)
-- ---------------------------------------------------------------------------
local frame_counter = 0

console.log("[LUA] Pokémon Blue AI script starting…")
console.log("[LUA] Connecting to Python server on port " .. CONFIG.port)
console.log("[LUA] Start the Python training script first, then this script.")
console.log("[LUA] Running main loop…")

while true do
    frame_counter = frame_counter + 1

    -- Apply any held button every frame for smooth input
    step_input()

    -- Only communicate every frame_skip frames
    if frame_counter % CONFIG.frame_skip == 0 then
        -- Read game state
        local state = read_game_state()

        -- Draw HUD overlay
        if CONFIG.debug then
            draw_hud(state)
        end

        -- Send state to Python, receive action
        local action = communicate(state)

        -- Apply action if received
        if action ~= nil then
            apply_action(action)
        end
    end

    emu.frameadvance()
end
