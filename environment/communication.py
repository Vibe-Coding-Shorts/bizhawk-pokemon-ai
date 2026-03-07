"""
communication.py
================
TCP server that the BizHawk Lua script connects to.

Protocol (BizHawk 2.11)
-----------------------
  Lua → Python   : JSON-encoded game state, newline-terminated (plain text)
  Python → Lua   : length-prefixed reply: "$<len> <payload>"
                   BizHawk's comm.socketServerResponse() reads the prefix,
                   validates the length, then returns just <payload> to Lua.

  Examples of valid Python→Lua messages:
    "$1 7"        action 7 (NoOp)
    "$1 0"        action 0 (Up)
    "$5 RESET"    reload savestate (episode reset)
    "$4 SAVE"     save current state to slot

Special payloads Python can send to Lua:
  "SAVE"  – Lua saves BizHawk state to slot 1
  "RESET" – Lua loads BizHawk state from slot 1 (episode reset)

reset_episode() flow
--------------------
1. Python sets _reset_flag, clears _reset_event.
2. On the next Lua→Python exchange, _recv_loop sends "RESET" to BizHawk.
3. BizHawk loads the savestate and on the very next frame sends a fresh state.
4. _recv_loop reads that post-reset state, sends a NoOp response, then
   signals _reset_event and puts the state on the queue.
5. reset_episode() unblocks and returns the post-reset state.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Optional

from environment.memory_reader import GameState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PartyMember:
    species: int = 0
    level:   int = 0
    hp_cur:  int = 0
    hp_max:  int = 0


@dataclass
class BagItem:
    id:    int = 0
    count: int = 0


@dataclass
class EmulatorState:
    """Structured snapshot received from BizHawk each step."""
    frame:          int = 0
    player_x:       int = 0
    player_y:       int = 0
    map_id:         int = 0
    in_battle:      int = 0   # 0=none, 1=wild, 2=trainer
    party_count:    int = 0
    party:          list[PartyMember] = field(default_factory=list)
    enemy_hp_cur:   int = 0
    enemy_hp_max:   int = 0
    enemy_species:  int = 0
    money:          int = 0
    badges:         int = 0
    pokedex_seen:   int = 0
    pokedex_caught: int = 0
    clock_hours:    int = 0
    clock_minutes:  int = 0
    text_on_screen: int = 0
    bag:            list[BagItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "EmulatorState":
        state = cls()
        for key in (
            "frame", "player_x", "player_y", "map_id", "in_battle",
            "party_count", "enemy_hp_cur", "enemy_hp_max", "enemy_species",
            "money", "badges", "pokedex_seen", "pokedex_caught",
            "clock_hours", "clock_minutes", "text_on_screen",
        ):
            setattr(state, key, int(d.get(key, 0)))

        state.party = [
            PartyMember(
                species=int(m.get("species", 0)),
                level=int(m.get("level", 0)),
                hp_cur=int(m.get("hp_cur", 0)),
                hp_max=int(m.get("hp_max", 0)),
            )
            for m in d.get("party", [])
        ]
        state.bag = [
            BagItem(id=int(b.get("id", 0)), count=int(b.get("count", 0)))
            for b in d.get("bag", [])
        ]
        return state

    @property
    def player_hp(self) -> int:
        return self.party[0].hp_cur if self.party else 0

    @property
    def player_hp_max(self) -> int:
        return self.party[0].hp_max if self.party else 1

    @property
    def player_level(self) -> int:
        return self.party[0].level if self.party else 0

    def to_game_state(self) -> GameState:
        return GameState(
            player_x=self.player_x,
            player_y=self.player_y,
            map_id=self.map_id,
            in_battle=self.in_battle,
            player_hp=self.player_hp,
            player_hp_max=self.player_hp_max,
            enemy_hp_cur=self.enemy_hp_cur,
            enemy_hp_max=self.enemy_hp_max,
            money=self.money,
            badges=self.badges,
            pokedex_seen=self.pokedex_seen,
            pokedex_caught=self.pokedex_caught,
            party_levels=[m.level for m in self.party],
            party_count=self.party_count,
        )


# ---------------------------------------------------------------------------
# TCP Bridge
# ---------------------------------------------------------------------------

class EmulatorBridge:
    """
    Runs a TCP server; BizHawk connects as a client (via --socket_ip/port).

    Thread model
    ------------
    _accept_loop  – waits for BizHawk to connect, spawns _recv_loop.
    _recv_loop    – per-exchange: read JSON state → send length-prefixed reply.

    Public API (called from the main training thread)
    --------------------------------------------------
    start()                    – bind, listen, start accept thread.
    stop()                     – shut down everything.
    send_action(action: int)   – queue action index for next exchange.
    send_command(cmd: str)     – queue special command (e.g. "SAVE").
    get_state(timeout) → EmulatorState | None
    reset_episode()    → EmulatorState  (raises RuntimeError on timeout)
    """

    _NOOP = "7"

    def __init__(self, host: str = "127.0.0.1", port: int = 65432):
        self.host = host
        self.port = port

        # State queue: recv_loop puts states here; get_state() consumes them.
        self._state_queue: Queue[EmulatorState] = Queue(maxsize=8)

        # Pending action/command to send on the next Lua→Python exchange.
        # Protected by _lock.
        self._pending: str = self._NOOP

        # Reset protocol flags (see reset_episode() docstring).
        self._reset_flag  = False
        self._reset_event = threading.Event()

        self._lock = threading.Lock()

        self._server_sock: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        self._client_file = None

        self._server_thread: Optional[threading.Thread] = None
        self._recv_thread:   Optional[threading.Thread] = None
        self._running = False
        self.connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the TCP server socket and start the accept thread."""
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(1.0)
        logger.info("EmulatorBridge listening on %s:%d", self.host, self.port)

        self._server_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="emulator-accept"
        )
        self._server_thread.start()

    def stop(self) -> None:
        """Shut down the bridge gracefully."""
        self._running = False
        self._disconnect_client()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        if self._server_thread:
            self._server_thread.join(timeout=2.0)

    def get_state(self, timeout: float = 10.0) -> Optional[EmulatorState]:
        """Block until the next emulator state arrives. Returns None on timeout."""
        try:
            return self._state_queue.get(timeout=timeout)
        except Empty:
            return None

    def send_action(self, action: int) -> None:
        """Queue an action index (0–7) to send on the next exchange."""
        with self._lock:
            self._pending = str(int(action))

    def send_command(self, cmd: str) -> None:
        """Queue a special command (e.g. 'SAVE') to send on the next exchange."""
        with self._lock:
            self._pending = cmd

    def reset_episode(self) -> EmulatorState:
        """
        Trigger a savestate reload in BizHawk and return the post-reset state.

        Steps:
        1. Set _reset_flag so _recv_loop sends "RESET" on the next exchange.
        2. Wait for _reset_event (set by _recv_loop after the post-reset state
           is queued).
        3. Return the post-reset EmulatorState from the queue.

        Raises RuntimeError if BizHawk does not respond within 30 s.
        """
        self._reset_event.clear()
        with self._lock:
            self._reset_flag = True
            self._pending = self._NOOP  # clear any stale pending action

        if not self._reset_event.wait(timeout=30.0):
            raise RuntimeError(
                "Timed out waiting for emulator after reset. "
                "Is BizHawk running with the Lua script active?"
            )

        try:
            return self._state_queue.get(timeout=5.0)
        except Empty:
            raise RuntimeError("Reset state missing from queue after event signal.")

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Accept client connections and hand off to _recv_loop."""
        while self._running:
            try:
                client, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    logger.error("Accept error: %s", exc)
                break

            logger.info("Emulator connected from %s", addr)
            with self._lock:
                self._disconnect_client()
                self._client_sock = client
                self._client_sock.settimeout(None)   # blocking reads
                self._client_file = self._client_sock.makefile(
                    "r", encoding="utf-8"
                )
                self.connected = True

            self._recv_thread = threading.Thread(
                target=self._recv_loop, daemon=True, name="emulator-recv"
            )
            self._recv_thread.start()
            self._recv_thread.join()   # block until connection drops

        logger.info("Accept loop exiting.")

    def _recv_loop(self) -> None:
        """
        Core exchange loop.

        For every JSON line received from Lua:
          1. Decide what to reply (reset, queued action, or NoOp).
          2. Send the reply in BizHawk's length-prefixed format: "$N payload".
          3a. Normal reply: put the received state on the queue.
          3b. RESET reply: read the post-reset state, send NoOp for it, then
              clear the queue, put the post-reset state, and fire _reset_event.
        """
        while self._running and self.connected:
            try:
                line = self._client_file.readline()
            except Exception as exc:
                if self._running:
                    logger.error("Socket read error: %s", exc)
                break

            if not line:
                logger.warning("Emulator disconnected (EOF).")
                break

            line = line.strip()
            if not line:
                continue

            # Parse JSON state
            try:
                raw   = json.loads(line)
                state = EmulatorState.from_dict(raw)
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("JSON parse error: %s | raw: %.120s", exc, line)
                # Send NoOp so BizHawk's socketServerResponse() unblocks
                self._bzk_send(self._NOOP)
                continue

            # Determine reply
            with self._lock:
                if self._reset_flag:
                    self._reset_flag = False
                    reply = "RESET"
                else:
                    reply = self._pending
                    self._pending = self._NOOP  # consume; default next to NoOp

            # Send reply to BizHawk in "$N payload" format
            self._bzk_send(reply)

            if reply in ("RESET", "LOAD"):
                # The pre-reset state is discarded.
                # BizHawk loads the savestate and immediately calls
                # communicate() again, sending the post-reset state.
                post_state = self._read_post_reset_state()
                if post_state is None:
                    break   # connection dropped during reset

                # Send NoOp for the post-reset exchange so BizHawk unblocks.
                self._bzk_send(self._NOOP)

                # Clear any stale pre-reset states and queue the fresh one.
                while not self._state_queue.empty():
                    try:
                        self._state_queue.get_nowait()
                    except Empty:
                        break
                self._state_queue.put(post_state)
                self._reset_event.set()

            else:
                # Normal exchange: queue the received state (drop oldest if full).
                if self._state_queue.full():
                    try:
                        self._state_queue.get_nowait()
                    except Empty:
                        pass
                self._state_queue.put(state)

        self._disconnect_client()

    def _read_post_reset_state(self) -> Optional[EmulatorState]:
        """
        Read the single state BizHawk sends right after processing RESET.
        Returns None if the connection drops.
        """
        try:
            line = self._client_file.readline()
        except Exception as exc:
            logger.error("Error reading post-reset state: %s", exc)
            return None

        if not line:
            logger.warning("Connection dropped while waiting for post-reset state.")
            return None

        try:
            return EmulatorState.from_dict(json.loads(line.strip()))
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Could not parse post-reset state: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Low-level send helpers
    # ------------------------------------------------------------------

    def _bzk_send(self, payload: str) -> None:
        """
        Send a reply to BizHawk in the required length-prefixed format.

        BizHawk 2.6.2+ requires comm.socketServerResponse() replies to be:
            "$<byte_length> <payload>"
        BizHawk reads exactly <byte_length> bytes after the space and returns
        that as the Lua string.  No trailing newline needed.
        """
        data = f"${len(payload)} {payload}".encode("utf-8")
        with self._lock:
            if self._client_sock:
                try:
                    self._client_sock.sendall(data)
                except Exception as exc:
                    logger.error("Send error: %s", exc)
                    self.connected = False

    def _disconnect_client(self) -> None:
        with self._lock:
            self.connected = False
            if self._client_file:
                try:
                    self._client_file.close()
                except Exception:
                    pass
                self._client_file = None
            if self._client_sock:
                try:
                    self._client_sock.close()
                except Exception:
                    pass
                self._client_sock = None
