"""Fixed, stdlib-only Windows trust bootstrap for the installed supervisor.

Install at the one HOST_PATH below, and execute with the one pinned interpreter
using -I -B. This is NOT a source-checkout launcher or an installer. No download,
repair, discovery of versions, identity replacement, or command override exists.

Policy and active-pointer publication, quiescence evidence, and rollback are
Shore installer responsibilities. An interrupted/missing pair fails closed.
The Python runtime starts before this script: its full reviewed, ACL-protected
baseline is therefore also an INSTALLATION prerequisite, not a sandbox created
by these after-start checks. Arbitrary writable sitecustomize/.pth is not safe.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import ctypes
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
import time
from uuid import uuid4

ROOT = PureWindowsPath(r"C:\ProgramData\ShorePrincess")
SUPERVISOR_ROOT = ROOT / "project-msabp" / "supervisor"
HOST_PATH = SUPERVISOR_ROOT / "supervisor_host.py"
FIXED_POLICY = SUPERVISOR_ROOT / "policy.json"
RUNTIME_ROOT = ROOT / "runtime-msabp"
VENV_PATH = RUNTIME_ROOT / "venv"
BASE_PATH = RUNTIME_ROOT / "python312"
PYTHON_PATH = VENV_PATH / "Scripts" / "python.exe"
BASE_PYTHON_PATH = BASE_PATH / "python.exe"
MODE = "shore-remote-synthetic-v1"
POLICY_SCHEMA = "msabp.supervisor-policy"
ACTIVE_SCHEMA = "msabp.supervisor-active-release"
STATE_SCHEMA = {"schema": "msabp.supervisor-state", "schema_version": 1}
LIMITS = {"json_bytes": 65536, "runtime_bytes": 65536, "csv_bytes": 262144,
          "material_bytes": 1048576, "synthetic_points": [1, 4]}
POLICY_FIELDS = {"schema", "schema_version", "mode", "project_id", "installation_id",
                 "discovery_enabled", "stop_when_idle", "profile", "bootstrap", "host_sha256",
                 "interpreter", "limits", "state_schema"}
SHORE_FILES = {"shore_control/" + name + ".py" for name in (
    "__init__", "authority", "client", "crypto", "evidence", "job_contract", "permits",
    "registry", "remote", "remote_identity", "rpc", "tailnet", "tailnet_rpc", "usb_identity")}
REQUIRED_PROJECT_FILES = {
    "scripts/simulation/shore_supervisor.py", "src/msabp_opt/__init__.py",
    "src/msabp_opt/simulation/__init__.py", "src/msabp_opt/simulation/distributed/__init__.py",
    "src/msabp_opt/simulation/distributed/control_winjob.py",
}
TRUSTED_SIDS = {"S-1-5-18", "S-1-5-32-544"}
# Windows Modules Installer is permitted only as an owner of OS ancestors,
# never as a substitute for the required SYSTEM/Admin installed-tree owner.
TRUSTED_INSTALLER_SID = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
WRITE_MASK = (0x2 | 0x4 | 0x10 | 0x40 | 0x100 | 0x10000 | 0x40000 | 0x80000
              | 0x10000000 | 0x40000000 | 0x02000000)
ANCESTOR_REPLACE_MASK = 0x40 | 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000 | 0x02000000
_RESERVED = {"con", "prn", "aux", "nul", "conin$", "conout$", "clock$",
             *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


class HostMaintenanceRequired(RuntimeError):
    """Trust, original-state, or lifecycle proof is absent; never repair it here."""


def canonical_bytes(value):
    def check(item):
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("Only canonical integer JSON values are permitted")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json(raw, *, maximum=65536):
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise ValueError("Missing or oversized trusted JSON")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    def forbidden(_value):
        raise ValueError("Floating point JSON is forbidden")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                       parse_float=forbidden, parse_constant=forbidden)
    canonical_bytes(value)
    return value


def _shape(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("Unexpected " + label + " schema")


def _hash(value):
    if type(value) is not str or re.fullmatch("[a-f0-9]{64}", value) is None or value == "0" * 64:
        raise ValueError("Missing non-placeholder SHA256 pin")
    return value


def _component(value):
    if (not value or value in {".", ".."} or value[-1] in ". "
            or any(ord(ch) < 32 or ch in '<>:"/\\|?*' for ch in value)
            or value.split(".", 1)[0].lower() in _RESERVED):
        raise ValueError("Unsafe Windows path component")


def _relative(value):
    if type(value) is not str or not value.isascii() or "\\" in value:
        raise ValueError("Noncanonical pinned relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        raise ValueError("Noncanonical pinned relative path")
    for part in value.split("/"):
        _component(part)
    return value


def _absolute(value):
    if type(value) is not str or re.match(r"^[A-Z]:\\", value) is None:
        raise ValueError("A canonical absolute local Windows path is required")
    path = PureWindowsPath(value)
    if str(path) != value or path.anchor.startswith("\\\\"):
        raise ValueError("A canonical absolute local Windows path is required")
    for part in path.parts[1:]:
        _component(part)
    return path


def _pins(value, *, absolute=False):
    if type(value) is not dict or not value or len(value) > 50000:
        raise ValueError("Missing or oversized pinned file whitelist")
    seen = set()
    for name, digest in value.items():
        (_absolute if absolute else _relative)(name)
        if name.casefold() in seen:
            raise ValueError("Case-colliding pinned paths")
        seen.add(name.casefold())
        _hash(digest)
    return value


def _profile(profile):
    _shape(profile, {"schema", "schema_version", "mode", "project_id", "release_id",
                     "release_manifest", "bootstrap_sha256"}, "release profile")
    manifest = profile["release_manifest"]
    _shape(manifest, {"schema", "schema_version", "mode", "project_id", "source_hashes",
                      "shore_source_hashes"}, "release manifest")
    for record, schema in ((profile, "msabp.supervisor-release-profile"),
                           (manifest, "msabp.supervisor-release")):
        if (record["schema"] != schema or type(record["schema_version"]) is not int
                or record["schema_version"] != 1 or record["mode"] != MODE
                or record["project_id"] != "MSABP_2"):
            raise ValueError("Unexpected pinned release identity")
    _pins(manifest["source_hashes"])
    _pins(manifest["shore_source_hashes"])
    if (not REQUIRED_PROJECT_FILES <= manifest["source_hashes"].keys()
            or set(manifest["shore_source_hashes"]) != SHORE_FILES):
        raise ValueError("Missing supervisor closure or changed Shore baseline")
    if _hash(profile["release_id"]) != hashlib.sha256(canonical_bytes(manifest)).hexdigest():
        raise ValueError("Release ID differs from the canonical approved manifest")
    _hash(profile["bootstrap_sha256"])


def _public_key(value):
    if type(value) is not str or not value.startswith("ssh-ed25519 ") or len(value.split()) != 2:
        raise ValueError("Canonical OpenSSH Ed25519 public key required")
    wire = base64.b64decode(value[12:], validate=True)
    if (len(wire) != 51 or wire[:19] != b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"
            or "ssh-ed25519 " + base64.b64encode(wire).decode("ascii") != value):
        raise ValueError("Canonical OpenSSH Ed25519 public key required")
    return "SHA256:" + base64.b64encode(hashlib.sha256(wire).digest()).decode("ascii").rstrip("=")


def _bootstrap(value):
    _shape(value, {"schema", "schema_version", "mode", "fleet_id", "authority_id",
        "authority_public_key", "endpoint", "tls_sha256", "network", "controllers",
        "workers", "controller_nodes", "collector"}, "public bootstrap")
    if (value["schema"] != "shore.remote-bootstrap" or type(value["schema_version"]) is not int
            or value["schema_version"] != 1 or value["mode"] != MODE
            or type(value["fleet_id"]) is not str
            or re.fullmatch("shore-princess-[a-f0-9]{24}", value["fleet_id"]) is None
            or _public_key(value["authority_public_key"]) != value["authority_id"]):
        raise ValueError("Unexpected public bootstrap identity")
    network = value["network"]
    _shape(network, {"tailnet_name", "magic_dns_suffix", "local_node_id", "local_address", "nodes"},
           "fixed network")
    if type(network["nodes"]) is not dict or len(network["nodes"]) != 2:
        raise ValueError("Exactly two pinned StableNodeIDs required")
    for node_id, node in network["nodes"].items():
        _shape(node, {"address", "dns_name"}, "fixed node")
        if (type(node_id) is not str or re.fullmatch(r"n[A-Za-z0-9]+", node_id) is None
                or type(node["address"]) is not str
                or str(ipaddress.IPv4Address(node["address"])) != node["address"]
                or ipaddress.IPv4Address(node["address"]) not in ipaddress.IPv4Network("100.64.0.0/10")
                or type(node["dns_name"]) is not str or not node["dns_name"].endswith(".")):
            raise ValueError("Invalid pinned Tailnet node")
    if (network["local_node_id"] not in network["nodes"]
            or network["local_node_id"] == "nJc1aWG7ZA11CNTRL"
            or "nJc1aWG7ZA11CNTRL" not in network["nodes"]
            or network["nodes"][network["local_node_id"]]["address"] != network["local_address"]
            or type(value["controller_nodes"]) is not list
            or set(value["controller_nodes"]) != set(network["nodes"])
            or len(value["controller_nodes"]) != 2):
        raise ValueError("Changed authority/worker StableNodeID binding")
    if (type(value["controllers"]) is not dict or len(value["controllers"]) != 1
            or any(re.fullmatch(r"princess-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
                                actor) is None for actor in value["controllers"])):
        raise ValueError("Exactly one enrolled Princess identity required")
    fingerprints = [value["authority_id"], *(_public_key(key) for key in value["controllers"].values())]
    _shape(value["workers"], {"coconutg2-bell", "coconutg2-maid"}, "worker registry")
    for actor, role, action in (("coconutg2-bell", "bell", "wake"), ("coconutg2-maid", "maid", "attempt")):
        row = value["workers"][actor]
        _shape(row, {"public_key", "device_id", "tailnet_node_id", "role", "allowed_actions"},
               "worker registration")
        if (row["device_id"] != "coconutg2" or row["tailnet_node_id"] != "nJc1aWG7ZA11CNTRL"
                or row["role"] != role or row["allowed_actions"] != [action]):
            raise ValueError("Changed fixed worker registration")
        fingerprints.append(_public_key(row["public_key"]))
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Enrolled identities must use distinct public keys")
    _shape(value["collector"], {"endpoint", "tls_sha256"}, "collector")
    for endpoint, pin, port in ((value["endpoint"], value["tls_sha256"], 18766),
                               (value["collector"]["endpoint"], value["collector"]["tls_sha256"], 18767)):
        if endpoint != "https://" + network["local_address"] + ":" + str(port):
            raise ValueError("Changed fixed collector/authority endpoint")
        _hash(pin)
    if value["tls_sha256"] == value["collector"]["tls_sha256"]:
        raise ValueError("Authority and collector TLS pins must be independent")


def _interpreter(value):
    _shape(value, {"executable", "base_executable", "prefix", "base_prefix", "roots", "files", "directories"},
           "reviewed interpreter baseline")
    fixed = {"executable": PYTHON_PATH, "base_executable": BASE_PYTHON_PATH,
             "prefix": VENV_PATH, "base_prefix": BASE_PATH}
    if any(value[field] != str(path) for field, path in fixed.items()):
        raise ValueError("Interpreter differs from the fixed installation")
    if type(value["roots"]) is not list or len(value["roots"]) != 2:
        raise ValueError("Exactly the two fixed dedicated runtime roots are required")
    roots = [_absolute(root) for root in value["roots"]]
    if set(value["roots"]) != {str(VENV_PATH), str(BASE_PATH)}:
        raise ValueError("Exactly the two fixed dedicated runtime roots are required")
    files = _pins(value["files"], absolute=True)
    required = {str(PYTHON_PATH), str(VENV_PATH / "pyvenv.cfg"), str(BASE_PYTHON_PATH)}
    if not required <= files.keys():
        raise ValueError("Missing interpreter/venv/base executable pin")
    for name in files:
        if not any(root in PureWindowsPath(name).parents for root in roots):
            raise ValueError("Interpreter pin outside approved roots")
    for root in roots:
        if not any(root in PureWindowsPath(name).parents for name in files):
            raise ValueError("Empty reviewed runtime baseline")
    if type(value["directories"]) is not list or len(value["directories"]) > 50000:
        raise ValueError("Explicit complete runtime directory whitelist required")
    directories = [_absolute(name) for name in value["directories"]]
    if len(set(directories)) != len(directories):
        raise ValueError("Duplicate runtime directory pin")
    if any(not any(root in path.parents for root in roots) for path in directories):
        raise ValueError("Runtime directory outside approved roots")
    required_directories = {parent for name in files for parent in PureWindowsPath(name).parents
                            if any(root in parent.parents for root in roots)}
    if not required_directories <= set(directories):
        raise ValueError("Runtime directory whitelist is incomplete")


def validate_policy(value):
    """Validate exact public policy, never grant trust to caller/network metadata."""
    _shape(value, POLICY_FIELDS, "supervisor policy")
    if (value["schema"] != POLICY_SCHEMA or type(value["schema_version"]) is not int
            or value["schema_version"] != 1 or value["mode"] != MODE or value["project_id"] != "MSABP_2"
            or type(value["installation_id"]) is not str
            or re.fullmatch("[a-f0-9]{32}", value["installation_id"]) is None
            or type(value["discovery_enabled"]) is not bool
            or type(value["stop_when_idle"]) is not bool
            or (value["stop_when_idle"] and value["discovery_enabled"])
            or canonical_bytes(value["limits"]) != canonical_bytes(LIMITS)
            or canonical_bytes(value["state_schema"]) != canonical_bytes(STATE_SCHEMA)):
        raise ValueError("Unexpected fixed supervisor policy identity or limits")
    _profile(value["profile"])
    _bootstrap(value["bootstrap"])
    _hash(value["host_sha256"])
    _interpreter(value["interpreter"])
    return _json(canonical_bytes(value), maximum=16 * 1024 * 1024)


def build_policy(*, profile, bootstrap, host_sha256, interpreter, installation_id):
    """Offline public candidate; reviewed runtime inventory must be supplied.

    Does not hash an arbitrary directory into trust, read identities, install,
    or enable discovery. Shore must supply its reviewed complete runtime pins.
    """
    return validate_policy({"schema": POLICY_SCHEMA, "schema_version": 1, "mode": MODE,
        "project_id": "MSABP_2", "installation_id": installation_id, "discovery_enabled": False,
        "stop_when_idle": False,
        "profile": profile, "bootstrap": bootstrap, "host_sha256": host_sha256,
        "interpreter": interpreter, "limits": LIMITS, "state_schema": STATE_SCHEMA})


def build_active_release(policy):
    policy = validate_policy(policy)
    return {"schema": ACTIVE_SCHEMA, "schema_version": 1,
        "installation_id": policy["installation_id"], "release_id": policy["profile"]["release_id"],
        "policy_sha256": hashlib.sha256(canonical_bytes(policy)).hexdigest(), "state_schema_version": 1}


def installed_paths(policy):
    release = _hash(policy["profile"]["release_id"])
    supervisor = Path(str(SUPERVISOR_ROOT))
    state = Path(str(ROOT / "state" / "project-msabp" / "supervisor"))
    return {"root": Path(str(ROOT)), "supervisor": supervisor, "host": supervisor / "supervisor_host.py",
        "policy": supervisor / "policy.json", "active_release": supervisor / "active-release.json",
        "bootstrap": supervisor / "authority-bootstrap.json",
        "app": Path(str(ROOT / "project-msabp" / "releases" / release / "app")),
        "manifest": Path(str(ROOT / "project-msabp" / "releases" / release / "release-manifest.json")),
        "shore": Path(str(ROOT / "app")), "python": Path(str(PYTHON_PATH)), "state": state,
        "config": Path(str(ROOT / "project-msabp" / "config")),
        "logs": Path(str(ROOT / "project-msabp" / "logs" / "supervisor")),
        "worker_config": Path(str(ROOT / "state" / "worker" / "worker-config.json")),
        "host_lock": state / "host.lock", "maintenance_lock": state / "maintenance.lock"}


def assert_acl_records(owner, aces, *, ancestor=False):
    """Pure conservative DACL check, also used by the typed Win32 reader below.

    ACE tuples are (type, flags, mask, SID). Unknown/callback/object ACE forms
    are rejected, not interpreted as if they were simple allow/deny records.
    """
    owners = TRUSTED_SIDS | ({TRUSTED_INSTALLER_SID} if ancestor else set())
    if owner not in owners or aces is None:
        raise HostMaintenanceRequired("Untrusted owner or unrestricted NULL DACL")
    for ace_type, flags, mask, sid in aces:
        if ace_type not in {0, 1}:
            raise HostMaintenanceRequired("Unsupported access-control entry")
        if ancestor and flags & 0x8:
            # INHERIT_ONLY does not grant access to this existing ancestor;
            # private child ACLs are checked separately without this exemption.
            continue
        trusted = owners | {"S-1-3-4"}
        forbidden = ANCESTOR_REPLACE_MASK if ancestor else WRITE_MASK
        if ace_type == 0 and sid not in trusted and mask & forbidden:
            raise HostMaintenanceRequired("Untrusted write access to protected infrastructure")


class _AclHeader(ctypes.Structure):
    _fields_ = [("revision", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                ("size", ctypes.c_uint16), ("count", ctypes.c_uint16), ("reserved2", ctypes.c_uint16)]


_ACL_API = None


def _acl_api():
    """The two libraries and their prototypes, set up once per process.

    One verification of the installation calls assert_private_acl tens of
    thousands of times. Declaring a ctypes type inside that function left one
    permanent entry in the ctypes pointer-type cache per call, about 200 MB per
    verification in a process that verifies again at every step. Nothing in the
    per-call path may create a type or load a library.
    """
    global _ACL_API
    if _ACL_API is None:
        c = ctypes
        advapi = c.WinDLL("advapi32", use_last_error=True)
        kernel = c.WinDLL("kernel32", use_last_error=True)
        pointer, dword = c.c_void_p, c.c_uint32
        advapi.GetNamedSecurityInfoW.argtypes = [c.c_wchar_p, c.c_int, dword,
            c.POINTER(pointer), c.POINTER(pointer), c.POINTER(pointer), c.POINTER(pointer), c.POINTER(pointer)]
        advapi.GetNamedSecurityInfoW.restype = dword
        advapi.ConvertSidToStringSidW.argtypes = [pointer, c.POINTER(pointer)]
        advapi.ConvertSidToStringSidW.restype = c.c_int
        advapi.IsValidSid.argtypes, advapi.IsValidSid.restype = [pointer], c.c_int
        advapi.GetAce.argtypes, advapi.GetAce.restype = [pointer, dword, c.POINTER(pointer)], c.c_int
        kernel.LocalFree.argtypes, kernel.LocalFree.restype = [pointer], pointer
        _ACL_API = (advapi, kernel)
    return _ACL_API


def assert_private_acl(path, *, ancestor=False):
    """Read owner/DACL with Win32 only; never edits permissions or uses a shell."""
    if os.name != "nt":
        raise HostMaintenanceRequired("Production ACL verification requires Windows")
    c = ctypes
    advapi, kernel = _acl_api()
    pointer = c.c_void_p
    owner, dacl, descriptor = pointer(), pointer(), pointer()
    result = advapi.GetNamedSecurityInfoW(str(path), 1, 0x1 | 0x4, c.byref(owner), None,
                                        c.byref(dacl), None, c.byref(descriptor))
    if result:
        raise HostMaintenanceRequired("Cannot read protected owner/DACL")
    def sid_text(address):
        text = pointer()
        if not advapi.IsValidSid(address) or not advapi.ConvertSidToStringSidW(address, c.byref(text)):
            raise HostMaintenanceRequired("Invalid protected SID")
        try:
            return c.wstring_at(text)
        finally:
            kernel.LocalFree(text)
    try:
        aces = None
        if dacl.value:
            acl = _AclHeader.from_address(dacl.value)
            aces = []
            for index in range(acl.count):
                ace = pointer()
                if not advapi.GetAce(dacl, index, c.byref(ace)):
                    raise HostMaintenanceRequired("Cannot read protected access entry")
                header = c.string_at(ace, 4)
                size = int.from_bytes(header[2:4], "little")
                if header[0] not in {0, 1} or size < 16:
                    raise HostMaintenanceRequired("Unsupported access-control entry")
                mask = c.c_uint32.from_address(ace.value + 4).value
                aces.append((header[0], header[1], mask, sid_text(ace.value + 8)))
        assert_acl_records(sid_text(owner), aces, ancestor=ancestor)
    finally:
        kernel.LocalFree(descriptor)


def checked_path(path):
    """Read-only local path check, including all ancestors and file hardlinks."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.anchor.startswith(("\\\\", "//")):
        raise HostMaintenanceRequired("Absolute local non-traversing path required")
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise HostMaintenanceRequired("Redirected protected path")
        if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
            raise HostMaintenanceRequired("Non-regular protected path")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise HostMaintenanceRequired("Hardlinked protected file")
    return path


def _trusted(path, *, root, acl_checker=assert_private_acl):
    path, root = checked_path(path), Path(root)
    if path != root and root not in path.parents:
        raise HostMaintenanceRequired("Protected path escaped its approved root")
    if acl_checker is assert_private_acl:
        # Root ACLs alone do not stop replacement through a writable ancestor.
        # OS roots can grant creation of unrelated children, but not ownership,
        # delete-child, DELETE, or ACL mutation to an untrusted principal.
        for ancestor in reversed(root.parents):
            assert_private_acl(ancestor, ancestor=True)
    for item in (*reversed(path.parents), path):
        if item == root or root in item.parents:
            acl_checker(item)
    return path


def _read(path, *, root, acl_checker=assert_private_acl, maximum=None):
    path = _trusted(path, root=root, acl_checker=acl_checker)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (maximum is not None and before.st_size > maximum):
        raise HostMaintenanceRequired("Missing, irregular, or oversized protected file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read() if maximum is None else stream.read(maximum + 1)
        finished = os.fstat(stream.fileno())
    after = checked_path(path).lstat()
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink)
    if (not identity(before) == identity(opened) == identity(finished) == identity(after)
            or before.st_ctime_ns != after.st_ctime_ns or opened.st_ctime_ns != finished.st_ctime_ns
            or (maximum is not None and len(raw) > maximum)):
        raise HostMaintenanceRequired("Protected file changed during verification")
    return raw


def verify_tree(root, pins, *, acl_checker=assert_private_acl, allowed_directories=None):
    """Pure package verification seam; overrides are not accepted by the CLI.

    Every file AND directory, including empty directories, must be in the
    protected local whitelist. No project module is imported to do this check.
    """
    _pins(pins)
    root = _trusted(Path(root), root=Path(root), acl_checker=acl_checker)
    directories = {str(parent) for name in pins for parent in PurePosixPath(name).parents
                   if str(parent) != "."}
    if allowed_directories is not None:
        declared = {_relative(name) for name in allowed_directories}
        if not directories <= declared:
            raise ValueError("Incomplete explicit directory whitelist")
        directories = declared
    seen, seen_directories, pending = set(), set(), [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = _trusted(Path(entry.path), root=root, acl_checker=acl_checker)
                name = path.relative_to(root).as_posix()
                if path.is_dir():
                    if name not in directories:
                        raise HostMaintenanceRequired("Unknown protected directory")
                    seen_directories.add(name)
                    pending.append(path)
                else:
                    if name not in pins or name in seen:
                        raise HostMaintenanceRequired("Unknown protected file")
                    seen.add(name)
                    if hashlib.sha256(_read(path, root=root, acl_checker=acl_checker)).hexdigest() != pins[name]:
                        raise HostMaintenanceRequired("Pinned protected file has drifted")
    if seen != set(pins) or seen_directories != directories:
        raise HostMaintenanceRequired("Missing protected source file")


def verify_active_release(policy, active):
    if canonical_bytes(active) != canonical_bytes(build_active_release(policy)):
        raise HostMaintenanceRequired("Active release and protected policy differ; installer recovery required")


def verify_execution_identity(policy, *, executable=None, base_executable=None, prefix=None,
                              base_prefix=None, flags=None, import_paths=None):
    """Pure comparison seam; production always uses current interpreter values."""
    flags = sys.flags if flags is None else flags
    if not flags.isolated or not flags.dont_write_bytecode or not flags.no_user_site or not flags.ignore_environment:
        raise HostMaintenanceRequired("The fixed interpreter must run with -I -B")
    values = {"executable": sys.executable if executable is None else executable,
        "base_executable": sys._base_executable if base_executable is None else base_executable,
        "prefix": sys.prefix if prefix is None else prefix,
        "base_prefix": sys.base_prefix if base_prefix is None else base_prefix}
    expected = policy["interpreter"]
    if any(str(PureWindowsPath(value)) != expected[name] for name, value in values.items()):
        raise HostMaintenanceRequired("Running interpreter differs from its protected baseline")
    roots = [PureWindowsPath(root) for root in expected["roots"]]
    paths = installed_paths(policy)
    release_imports = {PureWindowsPath(str(paths["app"] / "src")), PureWindowsPath(str(paths["shore"]))}
    for value in sys.path if import_paths is None else import_paths:
        path = _absolute(value)
        if path not in release_imports and not any(path == root or root in path.parents for root in roots):
            raise HostMaintenanceRequired("Unapproved interpreter import directory")


def verify_installation(policy_path=FIXED_POLICY):
    """Production pre-import gate. No installed-root override or state creation."""
    if os.name != "nt" or str(policy_path) != str(FIXED_POLICY):
        raise HostMaintenanceRequired("Only the fixed Windows installation policy is permitted")
    if str(Path(__file__).absolute()) != str(HOST_PATH):
        raise HostMaintenanceRequired("Only the installed fixed bootstrap may start production")
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise HostMaintenanceRequired("The fixed interpreter must run with -I -B")
    root = Path(str(ROOT))
    # Runtime inventory can exceed a task JSON limit; it is NOT downloaded.
    policy_raw = _read(Path(str(FIXED_POLICY)), root=root, maximum=16 * 1024 * 1024)
    policy = validate_policy(_json(policy_raw, maximum=16 * 1024 * 1024))
    if canonical_bytes(policy) != policy_raw:
        raise HostMaintenanceRequired("Installed policy must use canonical JSON bytes")
    paths = installed_paths(policy)
    verify_execution_identity(policy)
    if hashlib.sha256(_read(paths["host"], root=root)).hexdigest() != policy["host_sha256"]:
        raise HostMaintenanceRequired("Fixed bootstrap host pin mismatch")
    active = _json(_read(paths["active_release"], root=root, maximum=65536))
    verify_active_release(policy, active)
    bootstrap_raw = _read(paths["bootstrap"], root=root, maximum=65536)
    bootstrap = _json(bootstrap_raw)
    if (hashlib.sha256(bootstrap_raw).hexdigest() != policy["profile"]["bootstrap_sha256"]
            or bootstrap != policy["bootstrap"]):
        raise HostMaintenanceRequired("Pinned public bootstrap changed")
    manifest = _json(_read(paths["manifest"], root=root, maximum=65536))
    if manifest != policy["profile"]["release_manifest"]:
        raise HostMaintenanceRequired("Installed release manifest differs from approved policy")
    for runtime_root in policy["interpreter"]["roots"]:
        runtime_path = PureWindowsPath(runtime_root)
        pins = {PureWindowsPath(name).relative_to(runtime_path).as_posix(): digest
                for name, digest in policy["interpreter"]["files"].items()
                if runtime_path in PureWindowsPath(name).parents}
        directories = [PureWindowsPath(name).relative_to(runtime_path).as_posix()
                       for name in policy["interpreter"]["directories"]
                       if runtime_path in PureWindowsPath(name).parents]
        verify_tree(Path(runtime_root), pins, allowed_directories=directories)
    verify_tree(paths["app"], manifest["source_hashes"])
    # Shore retains its existing maintenance resources; its 14 reviewed source
    # dependencies and ancestors are verified, not a claimed full Shore closure.
    for name, expected in manifest["shore_source_hashes"].items():
        if hashlib.sha256(_read(paths["shore"] / name, root=root)).hexdigest() != expected:
            raise HostMaintenanceRequired("Pinned Shore dependency changed")
    # sys.path validation alone cannot explain a module injected by startup
    # configuration. Every file-backed module already loaded must belong to a
    # verified code baseline. Builtin/frozen modules have no separate file.
    module_files = set(policy["interpreter"]["files"])
    module_files.update(str(paths["app"] / name) for name in manifest["source_hashes"])
    module_files.update(str(paths["shore"] / name) for name in manifest["shore_source_hashes"])
    module_files.add(str(paths["host"]))
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and str(PureWindowsPath(filename)) not in module_files:
            raise HostMaintenanceRequired("Loaded module outside reviewed runtime/release baseline")
    for name in ("state", "config", "logs"):
        if not _trusted(paths[name], root=root).is_dir():
            raise HostMaintenanceRequired("Missing original installation directory")
    for name in ("host_lock", "maintenance_lock", "worker_config"):
        if not _trusted(paths[name], root=root).is_file():
            raise HostMaintenanceRequired("Missing original installation file")
    state = paths["state"]
    marker_raw = _read(state / "runtime-installed.json", root=root, maximum=65536)
    marker = _json(marker_raw)
    _shape(marker, {"schema", "schema_version", "installation_id", "config_parent", "profile"},
           "original installation marker")
    _profile(marker["profile"])
    expected_marker = {**STATE_SCHEMA, "installation_id": policy["installation_id"],
                       "config_parent": str(paths["config"]), "profile": marker["profile"]}
    if (marker_raw != canonical_bytes(expected_marker)
            or marker["profile"]["bootstrap_sha256"] != policy["profile"]["bootstrap_sha256"]):
        raise HostMaintenanceRequired("Original runtime marker differs from protected installation")
    # Initial profile is immutable history, not the current release pointer.
    # The runtime independently checks this original record against its DB and
    # refuses a version change while old jobs/processes remain unresolved.
    if marker["profile"] != policy["profile"]:
        original = installed_paths({"profile": marker["profile"]})
        old_manifest = marker["profile"]["release_manifest"]
        if _json(_read(original["manifest"], root=root, maximum=65536)) != old_manifest:
            raise HostMaintenanceRequired("Original release manifest missing or changed")
        verify_tree(original["app"], old_manifest["source_hashes"])
    if not _trusted(state / "runtime.sqlite", root=root).is_file():
        raise HostMaintenanceRequired("Original runtime ledger missing; no automatic initialization")
    # Double-read control records after expensive hashing: activation cannot be
    # silently accepted halfway through the preflight snapshot.
    if (_read(paths["policy"], root=root, maximum=16 * 1024 * 1024) != policy_raw
            or _json(_read(paths["active_release"], root=root, maximum=65536)) != active):
        raise HostMaintenanceRequired("Policy/activation changed during preflight")
    return {"policy": policy, "profile": policy["profile"], "bootstrap": bootstrap,
            "bootstrap_data": bootstrap_raw, "paths": paths, "active_release": active}


def preflight_worker_policy(policy_path=FIXED_POLICY):
    """Worker/maintenance read gate, separate from the parent's lifetime lock."""
    if os.name != "nt" or str(policy_path) != str(FIXED_POLICY):
        raise HostMaintenanceRequired("Only the fixed Windows installation policy is permitted")
    lock = Path(str(ROOT / "state" / "project-msabp" / "supervisor" / "maintenance.lock"))
    _trusted(lock, root=Path(str(ROOT)))
    with lifetime_lock(lock):
        return verify_installation(policy_path)


@contextmanager
def lifetime_lock(path):
    """OS-owned, noninherited exclusive byte lock; missing file is maintenance.

    Separate host.lock and maintenance.lock are installed explicitly; the
    MaterialJournal's operation lock is neither borrowed nor kept locked here.
    """
    path = checked_path(Path(path))
    if not path.is_file() or path.stat().st_size != 1:
        raise HostMaintenanceRequired("Original one-byte host/maintenance lock missing")
    with path.open("rb", buffering=0) as sentinel:
        if sentinel.read(1) != b"0":
            raise HostMaintenanceRequired("Original host/maintenance lock sentinel changed")
    with path.open("r+b") as stream:
        os.set_inheritable(stream.fileno(), False)
        # Lock beyond EOF so integrity checks can still read sentinel byte 0
        # while another handle holds the lifetime lock. Never truncate/write.
        stream.seek(1)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise HostMaintenanceRequired("Another host/maintenance instance owns the OS lock") from exc
        try:
            yield
        finally:
            stream.seek(1)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_new(path, value):
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def run_verified_host(context, *, job_factory=None):
    """Launch only the policy-derived worker in a suspended, assigned Job tree."""
    paths = context["paths"]
    # This call occurs ONLY after verify_installation enumerates the complete
    # release tree. Import roots are explicit, never derived from network data.
    if job_factory is None:
        sys.path[:0] = [str(paths["app"] / "src"), str(paths["shore"])]
        from msabp_opt.simulation.distributed.control_winjob import ManagedJob
        job_factory = ManagedJob
    argv = [str(paths["python"]), "-I", "-B", "-u",
            str(paths["app"] / "scripts" / "simulation" / "shore_supervisor.py"),
            "--policy", str(paths["policy"]), "--run"]
    invocation = uuid4().hex
    with (paths["logs"] / ("worker-" + invocation + ".log")).open("xb") as log:
        with job_factory() as job:
            process = job.spawn(argv, paths["app"], stdout=log, stderr=log)
            code = process.wait()
            record = process.exit_record
            if record is None:
                raise HostMaintenanceRequired("Worker exit lacks original-handle exit evidence")
            deadline = time.monotonic() + 5
            while job.active_processes() and time.monotonic() < deadline:
                time.sleep(0.05)
            normal_tree_exit = job.active_processes() == 0
            job.close()
            _write_new(paths["state"] / ("host-exit-" + invocation + ".json"), {
                "schema": "msabp.supervisor-host-exit", "schema_version": 1,
                "installation_id": context["policy"]["installation_id"],
                "release_id": context["profile"]["release_id"], "worker": record.to_dict(),
                "tree_exit_observed": job.active_processes() == 0,
                "normal_tree_exit": normal_tree_exit,
                "protocol_completion_proven": False})
            if not normal_tree_exit:
                raise HostMaintenanceRequired("Worker exited with descendants alive; emergency stop is not drain")
            return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(FIXED_POLICY), choices=[str(FIXED_POLICY)])
    args = parser.parse_args(argv)
    try:
        # Check fixed paths, ACLs and flags before taking any writable file lock.
        if (os.name != "nt" or str(Path(__file__).absolute()) != str(HOST_PATH)
                or sys.executable != str(PYTHON_PATH)
                or not sys.flags.isolated or not sys.flags.dont_write_bytecode):
            raise HostMaintenanceRequired("Installed fixed host and -I -B interpreter required")
        state = Path(str(ROOT / "state" / "project-msabp" / "supervisor"))
        for name in ("host.lock", "maintenance.lock"):
            _trusted(state / name, root=Path(str(ROOT)))
        with lifetime_lock(state / "host.lock"):
            with lifetime_lock(state / "maintenance.lock"):
                context = verify_installation(args.policy)
            return run_verified_host(context)
    except (OSError, ValueError, HostMaintenanceRequired):
        # Never render configuration objects or task API tokens to a console.
        print("Supervisor host requires local maintenance; original state preserved.", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
