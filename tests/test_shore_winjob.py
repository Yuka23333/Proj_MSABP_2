"""Real local Windows API tests: temporary Python children only, never CST/keys."""
from __future__ import annotations

import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import venv

import pytest

from msabp_opt.simulation.distributed import control_winjob as winjob


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Real Windows Job Object tests")
SOURCE_ROOT = str(Path(__file__).resolve().parents[1] / "src")


def _python(code, *args):
    return [sys.executable, "-I", "-B", "-u", "-c", code, *map(str, args)]


def _read_json_when_ready(path, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
    raise TimeoutError(f"Test child did not become ready: {path}")


def _observe_process(api, pid, *, terminate=False):
    function = api.kernel.OpenProcess
    function.argtypes = [winjob.DWORD, c.c_int, winjob.DWORD]
    function.restype = winjob.HANDLE
    # Retain the original object handle before killing the host, not a later PID lookup.
    rights = 0x100000 | 0x1000 | (1 if terminate else 0)
    return api.check(function(rights, False, pid), "OpenProcess(test observer)")


def _is_in_job(job, process_handle):
    function = job._api.kernel.IsProcessInJob
    function.argtypes = [winjob.HANDLE, winjob.HANDLE, c.POINTER(c.c_int)]
    function.restype = c.c_int
    contained = c.c_int()
    job._api.check(function(process_handle, job._handle, c.byref(contained)),
                   "IsProcessInJob(test original Job)")
    return bool(contained.value)


def test_normal_exit_record_and_argument_quoting(tmp_path):
    values = ["", "space value", 'a"b', "trailing\\", "& harmless ; text", "中文"]
    stdout_path, stderr_path = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
    code = "import json,sys; print(json.dumps(sys.argv[1:])); print('err',file=sys.stderr); sys.exit(7)"
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        with winjob.ManagedJob() as job:
            process = job.spawn(_python(code, *values), tmp_path, stdout=out, stderr=err)
            assert process.wait(10) == 7
            record = process.exit_record
            assert record.pid == process.pid
            assert record.creation_time == process.creation_time > 0
            assert record.exit_time >= record.creation_time
            assert record.exit_code == 7
            process.close()
            assert process.poll() == 7
            assert process.exit_record == record
    assert json.loads(stdout_path.read_text()) == values
    assert stderr_path.read_text().strip() == "err"
    assert job.active_processes() == 0
    assert record.to_dict()["creation_time"] == process.creation_time


def test_exit_code_still_active_is_a_real_exit(tmp_path):
    with winjob.ManagedJob() as job:
        process = job.spawn(_python("import sys; sys.exit(259)"), tmp_path)
        assert process.wait(10) == 259
        assert process.exit_record.exit_code == 259


def test_real_assignment_rejection_never_executes(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-run.txt"
    with winjob.ManagedJob() as job:
        limits = winjob._EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = (
            winjob.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | winjob.JOB_OBJECT_LIMIT_ACTIVE_PROCESS)
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        job._api.check(job._api.kernel.SetInformationJobObject(
            job._handle, 9, c.byref(limits), c.sizeof(limits)), "Test active process limit")
        first = job.spawn(_python("import time; time.sleep(30)"), tmp_path)
        assert first.poll() is None
        original_assign = job._api.assign
        observed = []

        def assign(job_handle, process_handle):
            observed.append(job._api.duplicate(process_handle))
            return original_assign(job_handle, process_handle)

        def forbidden_resume(_handle):
            pytest.fail("A process rejected by AssignProcessToJobObject must never resume")

        monkeypatch.setattr(job._api, "assign", assign)
        monkeypatch.setattr(job._api, "resume", forbidden_resume)
        try:
            with pytest.raises(OSError, match="AssignProcessToJobObject"):
                job.spawn(_python("from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                                  marker), tmp_path)
            assert len(observed) == 1
            assert job._api.wait(observed[0], 10000)
            assert not marker.exists()
            assert first.poll() is None
            # Windows may also associate a hidden conhost.exe with the tree.
            assert job.active_processes() >= 1
        finally:
            for handle in observed:
                job._api.close(handle)


def test_resume_failure_terminates_suspended_process(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-run.txt"
    with winjob.ManagedJob() as job:
        def fail(_handle):
            raise OSError("injected ResumeThread failure")

        monkeypatch.setattr(job._api, "resume", fail)
        with pytest.raises(OSError, match="ResumeThread"):
            job.spawn(_python("from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                              marker), tmp_path)
        assert not marker.exists()
        assert job.active_processes() == 0
        assert job._processes[0].exit_record is not None


def test_force_killed_host_closes_nested_job_and_descendants(tmp_path):
    grandchild = "import time; time.sleep(60)"
    child = (
        "import json,subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen({[sys.executable, '-I', '-B', '-c', grandchild]!r},"
        "creationflags=0x08000000); "
        "Path(sys.argv[1]).write_text(json.dumps({'grandchild':p.pid}),encoding='utf-8'); "
        "time.sleep(60)"
    )
    host = (
        f"import sys; sys.path.insert(0,{SOURCE_ROOT!r}); "
        "import json,time; from pathlib import Path; "
        "from msabp_opt.simulation.distributed.control_winjob import ManagedJob; "
        "root=Path(sys.argv[1]); job=ManagedJob(); "
        f"p=job.spawn({_python(child, tmp_path / 'grandchild.json')!r},root); "
        "(root/'child.json').write_text(json.dumps({'child':p.pid,"
        "'creation_time':p.creation_time}),encoding='utf-8'); time.sleep(60)"
    )
    with winjob.ManagedJob() as outer:
        process = outer.spawn(_python(host, tmp_path), tmp_path)
        child_info = _read_json_when_ready(tmp_path / "child.json")
        grandchild_info = _read_json_when_ready(tmp_path / "grandchild.json")
        handles = []
        try:
            handles.append(_observe_process(outer._api, child_info["child"]))
            handles.append(_observe_process(outer._api, grandchild_info["grandchild"]))
            assert outer._api.times(handles[0])[0] == child_info["creation_time"]
            assert outer.active_processes() >= 3
            # Only kill the host. The outer Job stays OPEN throughout assertions.
            process.terminate()
            assert process.wait(10) == winjob.TERMINATED_EXIT_CODE
            assert all(outer._api.wait(handle, 10000) for handle in handles)
            deadline = time.monotonic() + 10
            while outer.active_processes() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert outer.active_processes() == 0
        finally:
            for handle in handles:
                outer._api.close(handle)


def test_close_terminates_tree_and_retains_direct_exit_record(tmp_path):
    job = winjob.ManagedJob()
    try:
        process = job.spawn(_python("import time; time.sleep(30)"), tmp_path)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(0)
        with pytest.raises(RuntimeError, match="live process"):
            process.close()
        job.close()
        assert job.active_processes() == 0
        assert process.exit_record.exit_code == winjob.TERMINATED_EXIT_CODE
        process.close()
        job.close()
        with pytest.raises(RuntimeError, match="closed"):
            job.spawn(_python("pass"), tmp_path)
    finally:
        job.close()


@pytest.mark.parametrize("interpreter_kind", ["current", "base", "temporary-venv"])
def test_job_handle_not_inheritable_and_breakaway_not_permitted(tmp_path, interpreter_kind):
    interpreter = sys.executable
    if interpreter_kind == "base":
        interpreter = sys._base_executable
    elif interpreter_kind == "temporary-venv":
        # Local redirector only: no pip, package installation, network or real keys.
        venv.EnvBuilder(with_pip=False).create(tmp_path / "venv")
        interpreter = str(tmp_path / "venv" / "Scripts" / "python.exe")
    payload = (
        "import json,os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(json.dumps({'payload_pid':os.getpid()}),"
        "encoding='utf-8'); time.sleep(30)"
    )
    child_argv = [interpreter, "-I", "-B", "-c", payload, str(tmp_path / "payload.json")]
    code = (
        "import json,os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "result={'parent_payload_pid':os.getpid()}\n"
        "try:\n"
        f"    child=subprocess.Popen({child_argv!r},creationflags=0x1000000|0x08000000)\n"
        "except OSError as error:\n"
        "    result.update(outcome='denied',winerror=error.winerror)\n"
        "else:\n"
        "    result.update(outcome='started',child_launcher_pid=child.pid)\n"
        "Path(sys.argv[1]).write_text(json.dumps(result),encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    with winjob.ManagedJob() as job:
        flags = winjob.DWORD()
        get_flags = job._api.kernel.GetHandleInformation
        get_flags.argtypes = [winjob.HANDLE, c.POINTER(winjob.DWORD)]
        get_flags.restype = c.c_int
        job._api.check(get_flags(job._handle, c.byref(flags)), "GetHandleInformation")
        assert flags.value & 1 == 0
        process = job.spawn([interpreter, "-I", "-B", "-c", code,
                             str(tmp_path / "outcome.json")], tmp_path)
        assert _is_in_job(job, process._handle)
        result = _read_json_when_ready(tmp_path / "outcome.json")
        handles = []
        try:
            # The venv redirector's PID is NOT the Python payload's PID.
            # Assert original-Job membership of the actual code executing both
            # before and after the attempted breakaway, not just the launcher.
            handles.append(_observe_process(job._api, result["parent_payload_pid"],
                                            terminate=True))
            assert _is_in_job(job, handles[-1])
            if result["outcome"] == "denied":
                assert result["winerror"] == 5
                assert not (tmp_path / "payload.json").exists()
            else:
                assert result["outcome"] == "started"
                payload_info = _read_json_when_ready(tmp_path / "payload.json")
                for pid in (result["child_launcher_pid"], payload_info["payload_pid"]):
                    handles.append(_observe_process(job._api, pid, terminate=True))
                    assert _is_in_job(job, handles[-1]), "Payload escaped the managed Job"
            # No weakening to an exit-code-only assertion: every actual payload
            # is contained and must die when the original Job is closed.
            job.close()
            assert all(job._api.wait(handle, 10000) for handle in handles)
            assert job.active_processes() == 0
        finally:
            # Even a regression that permits escape must not leak a test child.
            # These are retained original handles to this test's own processes.
            for handle in handles:
                if not job._api.wait(handle, 0):
                    job._api.check(job._api.kernel.TerminateProcess(
                        handle, winjob.TERMINATED_EXIT_CODE), "TerminateProcess(test cleanup)")
                    assert job._api.wait(handle, 10000)
                job._api.close(handle)


def test_process_operations_keep_original_handle(tmp_path, monkeypatch):
    with winjob.ManagedJob() as job:
        done = job.spawn(_python("import sys; sys.exit(5)"), tmp_path)
        assert done.wait(10) == 5
        other = job.spawn(_python("import time; time.sleep(30)"), tmp_path)
        # A stale/reused PID must never become an OS lookup/termination target.
        monkeypatch.setattr(done, "_pid", other.pid)
        done.terminate()
        done.close()
        assert other.poll() is None
        assert done.exit_record.pid != other.pid


def test_independent_jobs_do_not_kill_each_other(tmp_path):
    with winjob.ManagedJob() as first, winjob.ManagedJob() as second:
        one = first.spawn(_python("import time; time.sleep(30)"), tmp_path)
        two = second.spawn(_python("import time; time.sleep(30)"), tmp_path)
        first.close()
        assert one.poll() == winjob.TERMINATED_EXIT_CODE
        assert two.poll() is None
        assert second.active_processes() >= 1


def test_wait_allows_concurrent_termination(tmp_path, monkeypatch):
    with winjob.ManagedJob() as job:
        process = job.spawn(_python("import time; time.sleep(30)"), tmp_path)
        waiting = threading.Event()
        original_wait = job._api.wait
        results = []

        def observed_wait(handle, milliseconds):
            if milliseconds == 5000:
                waiting.set()
            return original_wait(handle, milliseconds)

        def wait_in_thread():
            try:
                results.append(process.wait(5))
            except Exception as error:
                results.append(error)

        monkeypatch.setattr(job._api, "wait", observed_wait)
        thread = threading.Thread(target=wait_in_thread)
        thread.start()
        try:
            assert waiting.wait(2)
            process.terminate()
            thread.join(2)
            assert not thread.is_alive()
            assert results == [winjob.TERMINATED_EXIT_CODE]
        finally:
            thread.join(6)


@pytest.mark.parametrize("argv", [["python.exe", "-c", "pass"], [], "python.exe",
                                  [sys.executable, "x\0y"]])
def test_untrusted_command_forms_rejected(tmp_path, argv):
    with winjob.ManagedJob() as job:
        with pytest.raises(ValueError):
            job.spawn(argv, tmp_path)
