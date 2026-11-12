"""Candidate package is a closed public copy, not an installer."""
# ruff: noqa: F811 -- imported pytest fixture
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import json

import pytest

from test_shore_release import package  # noqa: F401
from msabp_opt.simulation.distributed.control_release import verify_release


@pytest.fixture
def builder(package, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/simulation/prepare_shore_supervisor_release.py"
    spec = spec_from_file_location("supervisor_package_builder", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    (package.app / "scripts/simulation/supervisor_host.py").write_bytes(b"# fixed host fixture\n")
    public = tmp_path / "public-bootstrap.json"
    public.write_bytes(b"public bootstrap fixture")
    return module, {"output": tmp_path / "candidate", "source_root": package.app,
        "shore_root": package.shore, "bootstrap_path": public,
        "bootstrap_sha256": hashlib.sha256(public.read_bytes()).hexdigest()}


def test_package_exact_closure_and_no_installation(builder, package):
    module, args = builder
    result = module.build_package(**args)
    assert result["status"] == "REQUIRES_SHORE_INSTALLER_REVIEW"
    assert result["installed"] is result["activated"] is False
    output = args["output"]
    profile = json.loads((output / "public-release-profile.json").read_bytes())
    verify_release(profile, output / "releases" / profile["release_id"] / "app", package.shore)
    assert (output / "supervisor/supervisor_host.py").is_file()
    assert not list(output.rglob("*key*"))
    assert not (output / "supervisor/policy.json").exists()
    assert not (output / "supervisor/active-release.json").exists()
    with pytest.raises(FileExistsError):
        module.build_package(**args)


def test_bad_public_pin_writes_nothing(builder):
    module, args = builder
    with pytest.raises(ValueError, match="bootstrap pin"):
        module.build_package(**{**args, "bootstrap_sha256": "f" * 64})
    assert not args["output"].exists()
