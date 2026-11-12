"""Pacing of the supervisor loop and the knock that ends an idle rest early.

Loopback sockets and a fake clock only. No identity, authority or installed
supervisor is touched; the loop's ``step`` is replaced by a scripted sequence.
"""
from __future__ import annotations

import socket
import time

import pytest

pytest.importorskip("shore_control")

from msabp_opt.simulation.distributed import control_supervisor_controller as controller
from msabp_opt.simulation.distributed import control_supervisor_runtime as runtime
from msabp_opt.simulation.distributed.control_remote_worker import LOCAL_NODE_ID


def loopback_listener():
    listener = runtime.KnockListener("127.0.0.1", 0)
    assert listener.listening
    return listener, listener.socket.getsockname()[1]


def send(port, payload=b"\0"):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(payload, ("127.0.0.1", port))


def test_rest_depends_on_what_the_last_step_found():
    assert runtime.pause_seconds("IDLE") == runtime.IDLE_SECONDS == 300
    assert runtime.pause_seconds("MAINTENANCE") == runtime.IDLE_SECONDS
    assert runtime.pause_seconds("BLOCKED") == runtime.BLOCKED_SECONDS == 30
    # Anything else means a job is in progress and keeps the short step.
    for state in ("DISCOVERED", "COMMITTED", "UNKNOWN", "AWAITING_RELEASE", "RELEASED", "anything-new"):
        assert runtime.pause_seconds(state) == runtime.STEP_SECONDS == 2


def test_a_knock_ends_the_rest_at_once():
    listener, port = loopback_listener()
    try:
        send(port)
        started = time.monotonic()
        assert listener.wait(30) is True
        assert time.monotonic() - started < 5
    finally:
        listener.close()


def test_without_a_knock_the_rest_runs_its_full_time():
    listener, _ = loopback_listener()
    try:
        started = time.monotonic()
        assert listener.wait(0.3) is False
        assert time.monotonic() - started >= 0.25
    finally:
        listener.close()


def test_a_knock_is_never_parsed_and_any_content_only_wakes():
    listener, port = loopback_listener()
    try:
        for payload in (b"", b"\0", b'{"state":"ACTIVE","job_id":"forged"}', b"x" * 4096):
            send(port, payload)
            assert listener.wait(5) is True
            listener.drain()
        assert listener.wait(0.2) is False
    finally:
        listener.close()


def test_drain_discards_knocks_that_came_before_the_step():
    listener, port = loopback_listener()
    try:
        for _ in range(5):
            send(port)
        assert listener.wait(5) is True
        listener.drain()
        assert listener.wait(0.2) is False
    finally:
        listener.close()


@pytest.mark.parametrize("address", [None, "not an address", "203.0.113.7"])
def test_an_unusable_address_falls_back_to_the_timer(address, monkeypatch):
    slept = []
    monkeypatch.setattr(runtime.time, "sleep", slept.append)
    listener = runtime.KnockListener(address)
    assert listener.listening is False
    assert listener.wait(300) is False
    assert slept == [300]
    listener.drain()
    listener.close()


def test_a_port_already_in_use_falls_back_to_the_timer():
    first, port = loopback_listener()
    try:
        second = runtime.KnockListener("127.0.0.1", port)
        assert second.listening is False
    finally:
        first.close()


class ScriptedKnock:
    """Records what the loop asks for; optionally reports a knock."""

    listening = True

    def __init__(self):
        self.events = []

    def drain(self):
        self.events.append("drain")

    def wait(self, seconds):
        self.events.append(("wait", seconds))
        return False

    def close(self):
        self.events.append("close")


def scripted_runtime(states):
    service = object.__new__(runtime.SupervisorRuntime)
    pending = list(states)

    def step():
        state = pending.pop(0)
        if isinstance(state, Exception):
            raise state
        return {"state": state}

    service.step = step
    return service


def test_loop_keeps_the_minimum_gap_and_rests_by_state(monkeypatch, capsys):
    slept = []
    monkeypatch.setattr(runtime.time, "sleep", slept.append)
    knock = ScriptedKnock()
    service = scripted_runtime(["IDLE", RuntimeError("boom"), "DISCOVERED", "MAINTENANCE", "STOPPED"])
    assert service.run(knock=knock) == 0
    # One unconditional short sleep after every step that did not stop the loop.
    assert slept == [runtime.STEP_SECONDS] * 4
    waits = [event[1] for event in knock.events if isinstance(event, tuple)]
    assert waits == [298.0, 28.0, 0.0, 298.0]
    # Every step is preceded by a drain, and the listener is closed at the end.
    assert knock.events.count("drain") == 5 and knock.events[-1] == "close"
    lines = capsys.readouterr().out.splitlines()
    # The listener state is reported once, after the first step, before the first rest.
    assert lines[0] == '{"state": "IDLE"}' and lines[1] == '{"knock": "listening"}'
    assert sum(line.startswith('{"knock"') for line in lines) == 1
    assert '"error": "RuntimeError"' in lines[2] and '"state": "BLOCKED"' in lines[2]


def test_loop_reports_timer_only_when_the_port_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    knock = ScriptedKnock()
    knock.listening = False
    assert scripted_runtime(["IDLE", "STOPPED"]).run(knock=knock) == 0
    assert capsys.readouterr().out.splitlines()[1] == '{"knock": "timer-only"}'


def test_a_supervisor_that_stops_at_once_never_opens_a_listener(monkeypatch, capsys):
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    service = scripted_runtime(["STOPPED"])
    service._knock_listener = lambda: pytest.fail("no listener before the first rest")
    assert service.run() == 0
    assert capsys.readouterr().out.splitlines() == ['{"state": "STOPPED"}']


def test_the_listener_is_opened_at_the_first_rest_and_closed_on_exit(monkeypatch):
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    made = []

    def make():
        made.append(ScriptedKnock())
        return made[-1]

    service = scripted_runtime(["IDLE", "IDLE", "STOPPED"])
    service._knock_listener = make
    assert service.run() == 0
    assert len(made) == 1 and made[0].events[-1] == "close"
    # No drain before the very first step: there is no listener yet.
    assert made[0].events.count("drain") == 2


def test_knocking_cannot_make_the_loop_step_faster_than_the_short_step(monkeypatch):
    # A flood of knocks ends every rest immediately, yet each step is still
    # followed by the unconditional short sleep.
    slept = []
    monkeypatch.setattr(runtime.time, "sleep", slept.append)

    class Flood(ScriptedKnock):
        def wait(self, seconds):
            super().wait(seconds)
            return True

    assert scripted_runtime(["IDLE"] * 6 + ["STOPPED"]).run(knock=Flood()) == 0
    assert slept == [runtime.STEP_SECONDS] * 6


def test_princess_knock_reaches_a_listening_supervisor():
    listener, port = loopback_listener()
    try:
        bootstrap = {"network": {"nodes": {LOCAL_NODE_ID: {"address": "127.0.0.1"}}}}
        assert controller.knock(bootstrap, port=port) is True
        assert listener.wait(5) is True
    finally:
        listener.close()


@pytest.mark.parametrize("bootstrap", [{}, {"network": {}}, {"network": {"nodes": {}}},
                                       {"network": {"nodes": {LOCAL_NODE_ID: {"address": "not an address"}}}}])
def test_princess_knock_never_raises(bootstrap):
    assert controller.knock(bootstrap, port=9) is False


def test_default_knock_port_is_the_one_the_supervisor_listens_on(monkeypatch):
    sent = []

    class Sender:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def sendto(self, payload, target): sent.append(target)

    monkeypatch.setattr(controller.socket, "socket", Sender)
    bootstrap = {"network": {"nodes": {LOCAL_NODE_ID: {"address": "100.108.0.45"}}}}
    assert controller.knock(bootstrap) is True
    assert sent == [("100.108.0.45", runtime.KNOCK_PORT)]
