"""
communication.py
================
TCP socket server that the BizHawk Lua script connects to.

Protocol
--------
  Lua  → Python : JSON-encoded game state, newline-terminated
  Python → Lua  : ASCII action index ("0"–"7") or command, newline-terminated

Special commands Python can send to Lua:
  "SAVE"  – save BizHawk state to slot
  "LOAD"  – load BizHawk state from slot
  "RESET" – alias for LOAD (reload savestate = episode reset)
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Optional

from environment.memory_reader import GameState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass that wraps the raw dict sent by Lua into a typed object
# ---------------------------------------------------------------------------

@dataclass
class PartyMember:
    species: int = 0
    level: int = 0
    hp_cur: int = 0
    hp_max: int = 0


@dataclass
class BagItem:
    id: int = 0
    count: int = 0


@dataclass
class EmulatorState:
    """Structured snapshot received from BizHawk each step."""
    frame: int = 0
    player_x: int = 0
    player_y: int = 0
    map_id: int = 0
    in_battle: int = 0          # 0=none, 1=wild, 2=trainer
    party_count: int = 0
    party: list[PartyMember] = field(default_factory=list)
    enemy_hp_cur: int = 0
    enemy_hp_max: int = 0
    enemy_species: int = 0
    money: int = 0
    badges: int = 0
    pokedex_seen: int = 0
    pokedex_caught: int = 0
    clock_hours: int = 0
    clock_minutes: int = 0
    text_on_screen: int = 0
    bag: list[BagItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "EmulatorState":
        state = cls()
        for key in (
            "frame", "player_x", "player_y", "map_id", "in_battle",
            "party_count", "enemy_hp_cur", "enemy_hp_max", "enemy_species",
            "money", "badges", "pokedex_seen", "pokedex_caught",
            "clock_hours", "clock_minutes", "text_on_screen",
        ):
            setattr(state, key, d.get(key, 0))

        state.party = [
            PartyMember(
                species=m.get("species", 0),
                level=m.get("level", 0),
                hp_cur=m.get("hp_cur", 0),
                hp_max=m.get("hp_max", 0),
            )
            for m in d.get("party", [])
        ]
        state.bag = [
            BagItem(id=b.get("id", 0), count=b.get("count", 0))
            for b in d.get("bag", [])
        ]
        return state

    @property
    def player_hp(self) -> int:
        if self.party:
            return self.party[0].hp_cur
        return 0

    @property
    def player_hp_max(self) -> int:
        if self.party:
            return self.party[0].hp_max
        return 1

    @property
    def player_level(self) -> int:
        if self.party:
            return self.party[0].level
        return 0

    def to_game_state(self) -> GameState:
        """Convert to the GameState used by the reward function."""
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
    Runs a TCP server in a background thread.
    The BizHawk Lua script connects as a client.

    Usage
    -----
    bridge = EmulatorBridge(host="127.0.0.1", port=65432)
    bridge.start()
    state = bridge.get_state(timeout=5.0)   # blocks until a frame arrives
    bridge.send_action(4)                    # send action index
    bridge.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 65432,
        state_queue_size: int = 4,
    ):
        self.host = host
        self.port = port

        self._state_queue: Queue[EmulatorState] = Queue(maxsize=state_queue_size)
        self._action_queue: Queue[str] = Queue(maxsize=1)

        self._server_sock: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        self._client_file = None          # file wrapper for line-by-line reading
        self._lock = threading.RLock()

        self._server_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False
        self.connected = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the TCP server and begin listening."""
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
        """Shut everything down cleanly."""
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
        """
        Block until a new state arrives from the emulator.
        Returns None on timeout.
        """
        try:
            return self._state_queue.get(timeout=timeout)
        except Empty:
            return None

    def send_action(self, action: int) -> None:
        """Queue an action to be sent to the emulator."""
        # Drain old pending action first (non-blocking)
        try:
            self._action_queue.get_nowait()
        except Empty:
            pass
        self._action_queue.put(str(action))

    def send_command(self, cmd: str) -> None:
        """Send a special command: SAVE, LOAD, RESET."""
        try:
            self._action_queue.get_nowait()
        except Empty:
            pass
        self._action_queue.put(cmd)

    def reset_episode(self) -> Optional[EmulatorState]:
        """Tell emulator to reload savestate, then wait for next state."""
        self.send_command("RESET")
        # Flush any stale states from the queue
        time.sleep(0.1)
        while not self._state_queue.empty():
            try:
                self._state_queue.get_nowait()
            except Empty:
                break
        return self.get_state(timeout=15.0)

    # ------------------------------------------------------------------ #
    # Internal threads                                                     #
    # ------------------------------------------------------------------ #

    def _accept_loop(self) -> None:
        """Accept a single client connection, then hand off to recv thread."""
        while self._running:
            try:
                client, addr = self._server_sock.accept()
                logger.info("Emulator connected from %s", addr)
                with self._lock:
                    self._disconnect_client()
                    self._client_sock = client
                    self._client_sock.settimeout(None)
                    self._client_file = self._client_sock.makefile("r", encoding="utf-8")
                    self.connected = True

                # Send initial no-op so BizHawk's first socketServerResponse()
                # call unblocks immediately (protocol: Python sends action first,
                # then BizHawk sends state).
                self._send_raw("7\n")

                self._recv_thread = threading.Thread(
                    target=self._recv_loop, daemon=True, name="emulator-recv"
                )
                self._recv_thread.start()
                self._recv_thread.join()  # wait until connection drops

            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    logger.error("Accept loop error: %s", exc)
                break

    def _recv_loop(self) -> None:
        """
        Receive JSON state lines from Lua, put them on the state queue,
        and immediately send back the queued action.
        """
        while self._running and self.connected:
            try:
                line = self._client_file.readline()
                if not line:
                    logger.warning("Emulator disconnected.")
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                    state = EmulatorState.from_dict(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("JSON parse error: %s | raw: %.120s", exc, line)
                    continue

                # Drop oldest state if queue is full (prefer freshness)
                if self._state_queue.full():
                    try:
                        self._state_queue.get_nowait()
                    except Empty:
                        pass
                self._state_queue.put(state)

                # Immediately send pending action (or no-op "7")
                try:
                    action_str = self._action_queue.get_nowait()
                except Empty:
                    action_str = "7"  # no-op

                self._send_raw(action_str + "\n")

            except Exception as exc:
                if self._running:
                    logger.error("Recv loop error: %s", exc)
                break

        self._disconnect_client()

    def _send_raw(self, data: str) -> None:
        with self._lock:
            if self._client_sock:
                try:
                    self._client_sock.sendall(data.encode("utf-8"))
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
