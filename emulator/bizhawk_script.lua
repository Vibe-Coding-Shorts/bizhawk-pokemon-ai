-- =============================================================================
-- BizHawk Lua Script: Pokémon Blue AI Interface
-- =============================================================================
-- Runs inside BizHawk and acts as the bridge between the emulator and the
-- external Python AI system. Communicates via TCP socket (luasocket).
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

local socket = require("socket")

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
    savestate.save(CONFIG.savestate_slot)
    if CONFIG.debug then
        console.log("[LUA] State saved to slot " .. CONFIG.savestate_slot)
    end
end

local function load_state()
    savestate.load(CONFIG.savestate_slot)
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
-- TCP socket server / client setup
-- ---------------------------------------------------------------------------
-- Strategy: BizHawk Lua acts as a TCP client connecting to the Python server.
-- Python runs a socket server on port 65432 and waits for connections.
-- Protocol:  Lua → Python: JSON state line  (terminated by \n)
--            Python → Lua: single decimal action index + \n
-- ---------------------------------------------------------------------------

local conn = nil
local connected = false

local function try_connect()
    local c = socket.tcp()
    c:settimeout(0.5)
    local ok, err = c:connect(CONFIG.host, CONFIG.port)
    if ok then
        c:settimeout(0)   -- non-blocking after connect
        conn = c
        connected = true
        console.log("[LUA] Connected to Python AI server at " .. CONFIG.host .. ":" .. CONFIG.port)
        return true
    else
        c:close()
        console.log("[LUA] Could not connect: " .. tostring(err) .. " – retrying…")
        return false
    end
end

local function disconnect()
    if conn then
        pcall(function() conn:close() end)
        conn = nil
    end
    connected = false
end

-- Send game state JSON and receive action
local recv_buf = ""

local function communicate(state)
    if not connected then return nil end

    -- Serialise and send state
    local json_str = to_json(state) .. "\n"
    local ok, err = conn:send(json_str)
    if not ok then
        console.log("[LUA] Send error: " .. tostring(err))
        disconnect()
        return nil
    end

    -- Wait briefly for response (up to ~60 ms in 10 ms polling steps)
    local action_line = nil
    for _ = 1, 6 do
        local chunk, err2 = conn:receive("*l")
        if chunk then
            action_line = chunk
            break
        elseif err2 ~= "timeout" then
            console.log("[LUA] Receive error: " .. tostring(err2))
            disconnect()
            return nil
        end
        socket.sleep(0.01)
    end

    if action_line then
        -- Parse special commands
        if action_line == "SAVE" then
            save_state()
            return nil
        elseif action_line == "LOAD" then
            load_state()
            return nil
        elseif action_line == "RESET" then
            load_state()
            return nil
        end
        local action_idx = tonumber(action_line)
        if action_idx ~= nil then
            return math.floor(action_idx)
        end
    end

    return nil  -- no-op this frame
end

-- ---------------------------------------------------------------------------
-- Main loop
-- ---------------------------------------------------------------------------
local frame_counter   = 0
local reconnect_timer = 0
local RECONNECT_INTERVAL = 120  -- frames between reconnect attempts

console.log("[LUA] Pokémon Blue AI script starting…")
console.log("[LUA] Waiting for Python server on " .. CONFIG.host .. ":" .. CONFIG.port)

-- Initial connection attempt
try_connect()

-- Register per-frame callback
event.onframestart(function()
    frame_counter = frame_counter + 1

    -- Step held input every frame
    step_input()

    -- Only communicate every frame_skip frames
    if frame_counter % CONFIG.frame_skip ~= 0 then
        return
    end

    -- Attempt reconnect if not connected
    if not connected then
        reconnect_timer = reconnect_timer + 1
        if reconnect_timer >= RECONNECT_INTERVAL then
            reconnect_timer = 0
            try_connect()
        end
        return
    end

    -- Read game state
    local state = read_game_state()

    -- Draw HUD overlay
    if CONFIG.debug then
        draw_hud(state)
    end

    -- Communicate with Python
    local action = communicate(state)

    -- Apply action if received
    if action ~= nil then
        apply_action(action)
    end
end)

-- Graceful shutdown on script stop
event.onexit(function()
    console.log("[LUA] Script exiting, closing connection.")
    disconnect()
end)

console.log("[LUA] Event handlers registered. Running…")
