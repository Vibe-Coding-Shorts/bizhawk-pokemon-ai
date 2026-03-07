-- =============================================================================
-- BizHawk Lua Script: Pokémon Blue AI Interface
-- Compatible with BizHawk 2.11 (Gambatte core, Game Boy)
-- =============================================================================
--
-- LAUNCH BIZHAWK WITH:
--   EmuHawk.exe --socket_ip=127.0.0.1 --socket_port=65432
--
-- The --socket_ip / --socket_port flags establish the TCP connection to the
-- Python server automatically. Do NOT call comm.socketServerSetIp/Port from
-- Lua – those calls reconnect the socket and break the CLI-established link.
--
-- BizHawk 2.11 socket comm API (comm library):
--   comm.socketServerSetTimeout(ms)  – set read timeout (call once at startup)
--   comm.socketServerSend(msg)       – send plain-text to Python server
--   comm.socketServerResponse()      – read Python's reply; Python MUST format
--                                      replies as "$<len> <msg>" (since 2.6.2)
--
-- Protocol (per communicate() call):
--   Lua  → Python : JSON game state, newline-terminated  (plain text)
--   Python → Lua  : "$N action_or_cmd"  where N = byte length of payload
--                   Lua receives just the payload (BizHawk strips the prefix)
--
-- Special commands Python can send:
--   SAVE  – save BizHawk state to slot 1
--   RESET – load BizHawk state from slot 1 (episode reset)
--
-- Memory addresses (Pokémon Blue / Red USA, Gambatte core):
--   Player X:          0xD362
--   Player Y:          0xD361
--   Map ID:            0xD35E
--   In Battle:         0xD057  (0=none, 1=wild, 2=trainer)
--   Party Count:       0xD163
--   Party HP cur/max:  0xD16C/0xD18D (slot 1), +44 bytes per slot
--   Party Level:       0xD18C (slot 1), +44 bytes per slot
--   Party Species:     0xD164–0xD169
--   Enemy HP cur:      0xCFE6 (2 bytes BE)
--   Enemy HP max:      0xCFEA (2 bytes BE)
--   Enemy Species:     0xCFD8
--   Money:             0xD347 (3 bytes BCD)
--   Badges:            0xD356 (bitfield, popcount = badge count)
--   Pokédex Seen:      0xD30A
--   Pokédex Caught:    0xD2F7
--   Game Clock h/m:    0xDA40 / 0xDA41
--   Text on screen:    0xD730
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------
local CONFIG = {
    port          = 65432,
    frame_skip    = 4,       -- communicate every N frames
    action_hold   = 8,       -- hold each button for N frames
    savestate_slot = 1,      -- BizHawk save-state slot for episode resets
    debug_hud     = true,    -- draw overlay on screen
}

-- ---------------------------------------------------------------------------
-- Memory addresses
-- ---------------------------------------------------------------------------
local ADDR = {
    player_x        = 0xD362,
    player_y        = 0xD361,
    map_id          = 0xD35E,
    in_battle       = 0xD057,
    party_count     = 0xD163,
    -- Slot offsets: each party member occupies 44 bytes past slot 1 base
    party_hp_cur    = { 0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248 },
    party_hp_max    = { 0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269 },
    party_level     = { 0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268 },
    party_species   = { 0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169 },
    enemy_hp_cur    = 0xCFE6,
    enemy_hp_max    = 0xCFEA,
    enemy_species   = 0xCFD8,
    money           = 0xD347,   -- 3 bytes BCD
    badges          = 0xD356,
    pokedex_seen    = 0xD30A,
    pokedex_caught  = 0xD2F7,
    clock_hours     = 0xDA40,
    clock_minutes   = 0xDA41,
    text_on_screen  = 0xD730,
    bag_item1       = 0xD31E,
    bag_item1_cnt   = 0xD31F,
    bag_item2       = 0xD320,
    bag_item2_cnt   = 0xD321,
}

-- ---------------------------------------------------------------------------
-- Action index → BizHawk joypad button name
-- 0=Up 1=Down 2=Left 3=Right 4=A 5=B 6=Start 7=NoOp
-- ---------------------------------------------------------------------------
local ACTION_MAP = {
    [0] = "Up", [1] = "Down", [2] = "Left", [3] = "Right",
    [4] = "A",  [5] = "B",    [6] = "Start", [7] = nil,
}

-- ---------------------------------------------------------------------------
-- Memory helpers
-- ---------------------------------------------------------------------------
local function read_u16_be(addr)
    return memory.read_u8(addr) * 256 + memory.read_u8(addr + 1)
end

local function read_money()
    local function bcd(b) return (math.floor(b / 16) * 10) + (b % 16) end
    local b1 = memory.read_u8(ADDR.money)
    local b2 = memory.read_u8(ADDR.money + 1)
    local b3 = memory.read_u8(ADDR.money + 2)
    return bcd(b1) * 10000 + bcd(b2) * 100 + bcd(b3)
end

local function popcount(b)
    local n = 0
    while b > 0 do n = n + (b % 2); b = math.floor(b / 2) end
    return n
end

-- ---------------------------------------------------------------------------
-- Build game-state table (read from emulator RAM)
-- ---------------------------------------------------------------------------
local function read_game_state()
    local party_count = memory.read_u8(ADDR.party_count)
    if party_count > 6 then party_count = 0 end

    local party = {}
    for i = 1, math.max(party_count, 1) do
        table.insert(party, {
            species = memory.read_u8(ADDR.party_species[i]),
            level   = memory.read_u8(ADDR.party_level[i]),
            hp_cur  = read_u16_be(ADDR.party_hp_cur[i]),
            hp_max  = read_u16_be(ADDR.party_hp_max[i]),
        })
    end

    local in_battle     = memory.read_u8(ADDR.in_battle)
    local enemy_hp_cur  = 0
    local enemy_hp_max  = 0
    local enemy_species = 0
    if in_battle > 0 then
        enemy_hp_cur  = read_u16_be(ADDR.enemy_hp_cur)
        enemy_hp_max  = read_u16_be(ADDR.enemy_hp_max)
        enemy_species = memory.read_u8(ADDR.enemy_species)
    end

    return {
        frame          = emu.framecount(),
        player_x       = memory.read_u8(ADDR.player_x),
        player_y       = memory.read_u8(ADDR.player_y),
        map_id         = memory.read_u8(ADDR.map_id),
        in_battle      = in_battle,
        party_count    = party_count,
        party          = party,
        enemy_hp_cur   = enemy_hp_cur,
        enemy_hp_max   = enemy_hp_max,
        enemy_species  = enemy_species,
        money          = read_money(),
        badges         = popcount(memory.read_u8(ADDR.badges)),
        pokedex_seen   = memory.read_u8(ADDR.pokedex_seen),
        pokedex_caught = memory.read_u8(ADDR.pokedex_caught),
        clock_hours    = memory.read_u8(ADDR.clock_hours),
        clock_minutes  = memory.read_u8(ADDR.clock_minutes),
        text_on_screen = memory.read_u8(ADDR.text_on_screen),
        bag = {
            { id = memory.read_u8(ADDR.bag_item1), count = memory.read_u8(ADDR.bag_item1_cnt) },
            { id = memory.read_u8(ADDR.bag_item2), count = memory.read_u8(ADDR.bag_item2_cnt) },
        },
    }
end

-- ---------------------------------------------------------------------------
-- Minimal JSON serialiser (no external library needed)
-- ---------------------------------------------------------------------------
local function to_json(val)
    local t = type(val)
    if t == "nil"     then return "null"
    elseif t == "boolean" then return tostring(val)
    elseif t == "number"  then
        if val ~= val then return "null" end   -- NaN guard
        return (val == math.floor(val)) and tostring(math.floor(val)) or tostring(val)
    elseif t == "string"  then
        return '"' .. val:gsub('\\','\\\\'):gsub('"','\\"')
                         :gsub('\n','\\n'):gsub('\r','\\r'):gsub('\t','\\t') .. '"'
    elseif t == "table" then
        if #val > 0 then
            local parts = {}
            for _, v in ipairs(val) do parts[#parts+1] = to_json(v) end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            local parts = {}
            for k, v in pairs(val) do
                parts[#parts+1] = '"' .. tostring(k) .. '":' .. to_json(v)
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    end
    return "null"
end

-- ---------------------------------------------------------------------------
-- Joypad input helpers
-- ---------------------------------------------------------------------------
local held_button  = nil
local hold_counter = 0

local function apply_action(idx)
    held_button  = ACTION_MAP[idx]   -- nil = NoOp
    hold_counter = CONFIG.action_hold
end

local function step_input()
    if hold_counter > 0 and held_button ~= nil then
        joypad.set({ [held_button] = true }, 1)
        hold_counter = hold_counter - 1
    else
        joypad.set({}, 1)
    end
end

-- ---------------------------------------------------------------------------
-- Save-state helpers (slot-based; Python resets episodes via "RESET" command)
-- ---------------------------------------------------------------------------
local function save_state()
    savestate.saveslot(CONFIG.savestate_slot)
    console.log("[LUA] Saved state to slot " .. CONFIG.savestate_slot)
end

local function load_state()
    savestate.loadslot(CONFIG.savestate_slot)
    console.log("[LUA] Loaded state from slot " .. CONFIG.savestate_slot)
end

-- ---------------------------------------------------------------------------
-- Debug HUD overlay
-- ---------------------------------------------------------------------------
local function draw_hud(s)
    gui.text(2,  2, string.format("Map:%d  X:%d Y:%d", s.map_id, s.player_x, s.player_y), "white")
    gui.text(2, 12, string.format("Battle:%d  Money:%d", s.in_battle, s.money), "yellow")
    if s.party_count > 0 and s.party[1] then
        local p = s.party[1]
        gui.text(2, 22, string.format("Lv%d HP:%d/%d", p.level, p.hp_cur, p.hp_max), "lime")
    end
    gui.text(2, 32, string.format("Badges:%d Seen:%d Caught:%d",
        s.badges, s.pokedex_seen, s.pokedex_caught), "cyan")
end

-- ---------------------------------------------------------------------------
-- Socket setup
-- ---------------------------------------------------------------------------
-- The connection to Python is established automatically by the CLI flags:
--   EmuHawk.exe --socket_ip=127.0.0.1 --socket_port=65432
--
-- socketServerSetTimeout sets how long socketServerResponse() blocks before
-- giving up (milliseconds). 3000ms gives Python enough time to respond even
-- if it is momentarily busy with RL inference or LLM calls.
--
-- DO NOT call comm.socketServerSetIp() or comm.socketServerSetPort() here –
-- those reconnect the socket and break the CLI-established connection.
-- ---------------------------------------------------------------------------
comm.socketServerSetTimeout(3000)

-- ---------------------------------------------------------------------------
-- communicate(state) – send game state to Python, receive action back
--
-- Uses the two-step BizHawk 2.11 API:
--   1. comm.socketServerSend(json)        – send state (plain text)
--   2. comm.socketServerResponse()        – read Python's reply
--      Python MUST reply with "$<len> <payload>" (length-prefixed).
--      BizHawk strips the prefix; this function receives just <payload>.
--
-- Returns action index (0–7) or nil (no-op / not connected).
-- ---------------------------------------------------------------------------
local function communicate(state)
    local json_str = to_json(state) .. "\n"

    -- Step 1: send state to Python
    local ok_send, send_err = pcall(comm.socketServerSend, json_str)
    if not ok_send then
        -- Connection not ready yet; Python may not have started
        return nil
    end

    -- Step 2: read Python's response (blocks up to socketServerSetTimeout ms)
    -- Python sends: "$N payload" where N = byte length of payload
    -- BizHawk parses the length-prefix and returns just the payload string.
    local ok_recv, response = pcall(comm.socketServerResponse)
    if not ok_recv or response == nil or response == "" then
        return nil
    end

    -- Trim whitespace
    response = response:match("^%s*(.-)%s*$")

    -- Handle special commands
    if response == "SAVE" then
        save_state()
        return nil   -- no joypad input this frame
    elseif response == "RESET" or response == "LOAD" then
        load_state()
        return nil   -- Python will receive the post-reset state next frame
    end

    -- Parse numeric action index
    local n = tonumber(response)
    return n and math.floor(n) or nil
end

-- ---------------------------------------------------------------------------
-- Main loop
-- ---------------------------------------------------------------------------
local frame_counter = 0

console.log("[LUA] Pokémon Blue AI script starting…")
console.log("[LUA] Connecting to Python server on port " .. CONFIG.port)
console.log("[LUA] Start the Python training script first, then this script.")
console.log("[LUA] Running main loop…")

while true do
    frame_counter = frame_counter + 1

    -- Apply held button every frame for smooth movement
    step_input()

    -- Communicate every frame_skip frames
    if frame_counter % CONFIG.frame_skip == 0 then
        local state  = read_game_state()

        if CONFIG.debug_hud then
            draw_hud(state)
        end

        local action = communicate(state)
        if action ~= nil then
            apply_action(action)
        end
    end

    emu.frameadvance()
end
