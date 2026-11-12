"""Unadapted entry points are denied before any process/network/file action."""
import importlib
from unittest.mock import Mock

import pytest

from msabp_opt.simulation.distributed import transport
from msabp_opt.simulation.distributed.bell import MaidBellClient, MaidBellController
from msabp_opt.simulation.distributed.protocol import make_message, validate_message


@pytest.fixture(autouse=True)
def shore_mode(monkeypatch):
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-fake-v1")


def test_raw_ssh_or_gpu_command_is_denied():
    runner = Mock(side_effect=AssertionError("Must not execute"))
    with pytest.raises(RuntimeError, match="refuses"):
        transport._run_checked(["ssh", "unused"], runner=runner, cwd=None, timeout=1, action="GPU relay")
    runner.assert_not_called()


@pytest.mark.parametrize("operation,args", [
    (transport.launch_maid, (None,)),
    (transport.push_file_atomic, (None, "missing", "missing")),
    (transport.copy_device_file_atomic, (None, "missing", "missing")),
    (transport.pull_file_atomic, (None, "missing", "missing")),
])
def test_all_transport_modes_fail_before_access(operation, args):
    with pytest.raises(RuntimeError, match="refuses"):
        operation(*args)


def test_bell_client_and_server_are_explicitly_disabled():
    with pytest.raises(RuntimeError, match="refuses"):
        MaidBellClient("127.0.0.1", 1).wake(device_id="lab", runtime_config_path="missing", api_token="none")
    with pytest.raises(RuntimeError, match="refuses"):
        MaidBellController._wake(None, {})


@pytest.mark.parametrize("module", ["proposal_relay", "krvea_relay", "phase2_krvea_relay", "phase2_krvea_tree_relay"])
def test_gpu_compute_entry_rejects_before_parsing_or_fitting(module):
    relay = importlib.import_module("msabp_opt.optimization." + module)
    with pytest.raises(RuntimeError, match="refuses"):
        relay.run_request_payload({})


def test_protocol_version_is_explicit_and_strict():
    message = make_message("hello", run_id="lab", protocol_version=2)
    assert validate_message(message)["protocol_version"] == 2
    message["protocol_version"] = True
    with pytest.raises(ValueError, match="unsupported"):
        validate_message(message)
