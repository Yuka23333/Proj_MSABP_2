"""Fail-closed Windows process-tree ownership for the fixed supervisor host.

This is a lifecycle primitive, not a command authorization boundary. The caller
must derive the absolute executable, arguments and cwd from its trusted policy.
No shell, PATH lookup, breakaway, PID-based reopening or Popen fallback is used.
Only duplicated stdin/stdout/stderr handles are inherited (NUL by default).
Nested launchers (including CPython's Windows venv redirector) may create their
own breakaway-enabled inner Jobs. Leaving those inner Jobs does not remove this
outer Job's membership; a successful nested CREATE_BREAKAWAY_FROM_JOB call alone
is not evidence of escape. Tree exit must still be observed on this outer Job.

Creation/exit times are exact Windows FILETIME ticks (100 ns since 1601 UTC).
Exit records are local observations, NOT proof of protocol completion or that a
task never started. Callers must persist them alongside their original ledger.
See https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects .
"""
from __future__ import annotations

from contextlib import ExitStack
import ctypes as c
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import BinaryIO
import weakref


DWORD = c.c_uint32
HANDLE = c.c_void_p
SIZE_T = c.c_size_t
INFINITE = 0xFFFFFFFF
WAIT_TIMEOUT = 258
CREATE_SUSPENDED = 0x4
CREATE_NO_WINDOW = 0x08000000
EXTENDED_STARTUPINFO_PRESENT = 0x80000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x8
TERMINATED_EXIT_CODE = 0xC000013A


class _FILETIME(c.Structure):
    _fields_ = [("low", DWORD), ("high", DWORD)]

    @property
    def ticks(self):
        return (self.high << 32) | self.low


class _BASIC_LIMIT(c.Structure):
    _fields_ = [("PerProcessUserTimeLimit", c.c_int64),
                ("PerJobUserTimeLimit", c.c_int64), ("LimitFlags", DWORD),
                ("MinimumWorkingSetSize", SIZE_T), ("MaximumWorkingSetSize", SIZE_T),
                ("ActiveProcessLimit", DWORD), ("Affinity", SIZE_T),
                ("PriorityClass", DWORD), ("SchedulingClass", DWORD)]


class _IO_COUNTERS(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _EXTENDED_LIMIT(c.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", SIZE_T), ("JobMemoryLimit", SIZE_T),
                ("PeakProcessMemoryUsed", SIZE_T), ("PeakJobMemoryUsed", SIZE_T)]


class _ACCOUNTING(c.Structure):
    _fields_ = [(name, c.c_int64) for name in (
        "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime",
        "ThisPeriodTotalKernelTime")] + [(name, DWORD) for name in (
            "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses",
            "TotalTerminatedProcesses")]


class _STARTUPINFO(c.Structure):
    _fields_ = [("cb", DWORD), ("lpReserved", c.c_wchar_p),
                ("lpDesktop", c.c_wchar_p), ("lpTitle", c.c_wchar_p)] + [
                    (name, DWORD) for name in (
                        "dwX", "dwY", "dwXSize", "dwYSize", "dwXCountChars",
                        "dwYCountChars", "dwFillAttribute", "dwFlags")] + [
                    ("wShowWindow", c.c_uint16), ("cbReserved2", c.c_uint16),
                    ("lpReserved2", c.c_void_p), ("hStdInput", HANDLE),
                    ("hStdOutput", HANDLE), ("hStdError", HANDLE)]


class _STARTUPINFOEX(c.Structure):
    _fields_ = [("StartupInfo", _STARTUPINFO), ("lpAttributeList", c.c_void_p)]


class _PROCESS_INFORMATION(c.Structure):
    _fields_ = [("hProcess", HANDLE), ("hThread", HANDLE),
                ("dwProcessId", DWORD), ("dwThreadId", DWORD)]


class _WinAPI:
    """Typed ctypes calls, also a narrow fault-injection seam for regression tests."""

    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows Job Objects require Windows; no fallback is allowed")
        self.kernel = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([c.c_void_p, c.c_wchar_p], HANDLE),
            "SetInformationJobObject": ([HANDLE, c.c_int, c.c_void_p, DWORD], c.c_int),
            "QueryInformationJobObject": ([HANDLE, c.c_int, c.c_void_p, DWORD,
                                            c.c_void_p], c.c_int),
            "AssignProcessToJobObject": ([HANDLE, HANDLE], c.c_int),
            "TerminateJobObject": ([HANDLE, DWORD], c.c_int),
            "CreateProcessW": ([c.c_wchar_p, c.c_wchar_p, c.c_void_p, c.c_void_p,
                                c.c_int, DWORD, c.c_void_p, c.c_wchar_p,
                                c.c_void_p, c.POINTER(_PROCESS_INFORMATION)], c.c_int),
            "ResumeThread": ([HANDLE], DWORD),
            "TerminateProcess": ([HANDLE, DWORD], c.c_int),
            "WaitForSingleObject": ([HANDLE, DWORD], DWORD),
            "GetExitCodeProcess": ([HANDLE, c.POINTER(DWORD)], c.c_int),
            "GetProcessTimes": ([HANDLE] + [c.POINTER(_FILETIME)] * 4, c.c_int),
            "CloseHandle": ([HANDLE], c.c_int),
            "SetHandleInformation": ([HANDLE, DWORD, DWORD], c.c_int),
            "GetCurrentProcess": ([], HANDLE),
            "DuplicateHandle": ([HANDLE, HANDLE, HANDLE, c.POINTER(HANDLE),
                                 DWORD, c.c_int, DWORD], c.c_int),
            "InitializeProcThreadAttributeList": ([c.c_void_p, DWORD, DWORD,
                                                    c.POINTER(SIZE_T)], c.c_int),
            "UpdateProcThreadAttribute": ([c.c_void_p, DWORD, SIZE_T, c.c_void_p,
                                           SIZE_T, c.c_void_p, c.c_void_p], c.c_int),
            "DeleteProcThreadAttributeList": ([c.c_void_p], None),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = argtypes, restype

    @staticmethod
    def check(result, operation):
        if not result:
            error = c.get_last_error()
            raise OSError(error, f"{operation}: {c.FormatError(error)}")
        return result

    def close(self, handle):
        self.check(self.kernel.CloseHandle(handle), "CloseHandle")

    def duplicate(self, handle, *, inherit=False):
        result = HANDLE()
        current = self.kernel.GetCurrentProcess()
        self.check(self.kernel.DuplicateHandle(
            current, handle, current, c.byref(result), 0, inherit, 2), "DuplicateHandle")
        return result.value

    def times(self, handle):
        creation, exit_time, kernel, user = (_FILETIME() for _ in range(4))
        self.check(self.kernel.GetProcessTimes(
            handle, c.byref(creation), c.byref(exit_time), c.byref(kernel),
            c.byref(user)), "GetProcessTimes")
        return creation.ticks, exit_time.ticks

    def wait(self, handle, milliseconds):
        result = self.kernel.WaitForSingleObject(handle, milliseconds)
        if result not in (0, WAIT_TIMEOUT):
            self.check(False, "WaitForSingleObject")
        return result == 0

    def active(self, handle):
        accounting = _ACCOUNTING()
        self.check(self.kernel.QueryInformationJobObject(
            handle, 1, c.byref(accounting), c.sizeof(accounting), None),
            "QueryInformationJobObject")
        return int(accounting.ActiveProcesses)

    def assign(self, job_handle, process_handle):
        self.check(self.kernel.AssignProcessToJobObject(job_handle, process_handle),
                   "AssignProcessToJobObject")

    def resume(self, thread_handle):
        result = self.kernel.ResumeThread(thread_handle)
        if result == INFINITE:
            self.check(False, "ResumeThread")
        if result != 1:
            raise RuntimeError(f"Unexpected primary-thread suspend count: {result}")

    def create_suspended(self, argv, cwd, stdout, stderr):
        info = _PROCESS_INFORMATION()
        try:
            return self._create_suspended(argv, cwd, stdout, stderr, info)
        except BaseException:
            # Handle cleanup itself can fail after CreateProcessW succeeded.
            # The process is still suspended and must not become an orphan.
            if info.hProcess:
                try:
                    if not self.wait(info.hProcess, 0):
                        self.check(self.kernel.TerminateProcess(
                            info.hProcess, TERMINATED_EXIT_CODE),
                            "TerminateProcess(startup cleanup failure)")
                    if not self.wait(info.hProcess, 10000):
                        raise RuntimeError("Suspended startup process did not exit")
                finally:
                    self.close(info.hProcess)
                    self.close(info.hThread)
            raise

    def _create_suspended(self, argv, cwd, stdout, stderr, info):
        import msvcrt

        with ExitStack() as stack:
            null_in = stack.enter_context(open(os.devnull, "rb"))
            null_out = stack.enter_context(open(os.devnull, "wb"))
            files = (null_in, stdout if stdout is not None else null_out,
                     stderr if stderr is not None else null_out)
            handles = []
            for stream in files:
                stream.flush()
                handle = self.duplicate(msvcrt.get_osfhandle(stream.fileno()), inherit=True)
                stack.callback(self.close, handle)
                handles.append(handle)
            handle_list = (HANDLE * len(handles))(*handles)
            size = SIZE_T()
            self.kernel.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
            if not size.value:
                self.check(False, "InitializeProcThreadAttributeList(size)")
            attributes = c.create_string_buffer(size.value)
            self.check(self.kernel.InitializeProcThreadAttributeList(
                attributes, 1, 0, c.byref(size)), "InitializeProcThreadAttributeList")
            stack.callback(self.kernel.DeleteProcThreadAttributeList, attributes)
            self.check(self.kernel.UpdateProcThreadAttribute(
                attributes, 0, 0x20002, handle_list, c.sizeof(handle_list), None, None),
                "UpdateProcThreadAttribute(handle list)")
            startup = _STARTUPINFOEX()
            startup.StartupInfo.cb = c.sizeof(startup)
            startup.StartupInfo.dwFlags = 0x100  # STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, \
                startup.StartupInfo.hStdError = handles
            startup.lpAttributeList = c.cast(attributes, c.c_void_p)
            command_line = c.create_unicode_buffer(subprocess.list2cmdline(argv))
            self.check(self.kernel.CreateProcessW(
                argv[0], command_line, None, None, True,
                CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT,
                None, str(cwd), c.byref(startup), c.byref(info)), "CreateProcessW")
            return info


@dataclass(frozen=True)
class ExitRecord:
    pid: int
    creation_time: int
    exit_time: int
    exit_code: int

    def to_dict(self):
        return asdict(self)


class ManagedProcess:
    """Own the original process handle until a real exit has been recorded."""

    def __init__(self, api, handle, pid, creation_time):
        self._api, self._handle = api, handle
        self._pid, self._creation_time = pid, creation_time
        self._exit_record = None
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, api.close, handle)

    @property
    def pid(self):
        return self._pid

    @property
    def creation_time(self):
        return self._creation_time

    @property
    def exit_record(self):
        with self._lock:
            self.poll()
            return self._exit_record

    def poll(self):
        with self._lock:
            if self._exit_record is not None:
                return self._exit_record.exit_code
            if not self._api.wait(self._handle, 0):
                return None
            code = DWORD()
            self._api.check(self._api.kernel.GetExitCodeProcess(
                self._handle, c.byref(code)), "GetExitCodeProcess")
            creation, exit_time = self._api.times(self._handle)
            if creation != self.creation_time or not exit_time:
                raise RuntimeError("Process-handle identity or exit-time mismatch")
            self._exit_record = ExitRecord(self.pid, creation, exit_time, code.value)
            return code.value

    def wait(self, timeout=None):
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        with self._lock:
            if self.poll() is not None:
                return self._exit_record.exit_code
            # Do not hold the process lock while waiting: another thread must
            # still be able to terminate. The duplicate preserves the same OS
            # process object even if a concurrent closer records and closes it.
            wait_handle = self._api.duplicate(self._handle)
        try:
            milliseconds = INFINITE if timeout is None else min(
                INFINITE - 1, math.ceil(timeout * 1000))
            if not self._api.wait(wait_handle, milliseconds):
                raise subprocess.TimeoutExpired(f"managed process {self.pid}", timeout)
            return self.poll()
        finally:
            self._api.close(wait_handle)

    def terminate(self, exit_code=TERMINATED_EXIT_CODE):
        """Terminate this exact process, not its tree; close the Job for the tree."""
        with self._lock:
            if self.poll() is None:
                result = self._api.kernel.TerminateProcess(self._handle, exit_code)
                if not result and self.poll() is None:
                    self._api.check(False, "TerminateProcess")

    def close(self):
        """Release a finished process handle, retaining its immutable exit record."""
        with self._lock:
            if self._handle is None:
                return
            if self.poll() is None:
                raise RuntimeError("Cannot discard a live process handle; wait or close its Job")
            self._finalizer()
            self._handle = None


class ManagedJob:
    """An unnamed, noninheritable, kill-on-close Job Object with no breakaway.

    Context exit is an emergency tree stop, NOT graceful drain. Normal lifecycle
    callers must first settle their ledger and wait for all processes to exit.
    Do not use this close operation just because F/USB disappears.
    """

    def __init__(self):
        self._api = _WinAPI()
        self._lock = threading.RLock()
        self._processes = []
        self._tree_exited = False
        handle = self._api.check(self._api.kernel.CreateJobObjectW(None, None),
                                 "CreateJobObjectW")
        try:
            self._api.check(self._api.kernel.SetHandleInformation(handle, 1, 0),
                            "SetHandleInformation(noninheritable Job)")
            limits = _EXTENDED_LIMIT()
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self._api.check(self._api.kernel.SetInformationJobObject(
                handle, 9, c.byref(limits), c.sizeof(limits)), "SetInformationJobObject")
        except BaseException:
            self._api.close(handle)
            raise
        self._handle = handle
        self._finalizer = weakref.finalize(self, self._api.close, handle)

    def __enter__(self):
        if self._handle is None:
            raise RuntimeError("Job is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def spawn(self, argv: list[str], cwd: Path, *,
              stdout: BinaryIO | None = None,
              stderr: BinaryIO | None = None) -> ManagedProcess:
        if (not isinstance(argv, list) or not argv or
                any(not isinstance(arg, str) or "\0" in arg for arg in argv)):
            raise ValueError("argv must be a nonempty list of NUL-free strings")
        if not Path(argv[0]).is_absolute() or Path(argv[0]).suffix.lower() != ".exe":
            raise ValueError("A policy-selected absolute .exe path is required")
        cwd = Path(cwd)
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError("cwd must be an existing absolute directory")
        with self._lock:
            if self._handle is None:
                raise RuntimeError("Job is closed")
            info = self._api.create_suspended(argv, cwd, stdout, stderr)
            process = None
            try:
                creation_time, _ = self._api.times(info.hProcess)
                self._api.assign(self._handle, info.hProcess)
                process = ManagedProcess(self._api, info.hProcess,
                                         int(info.dwProcessId), creation_time)
                self._processes.append(process)
                self._api.resume(info.hThread)
            except BaseException:
                # Includes assignment/nesting failure: NEVER resume or fall back.
                if not self._api.wait(info.hProcess, 0):
                    self._api.check(self._api.kernel.TerminateProcess(
                        info.hProcess, TERMINATED_EXIT_CODE), "TerminateProcess(start failure)")
                if not self._api.wait(info.hProcess, 10000):
                    raise RuntimeError("Failed to confirm suspended process termination")
                if process is not None:
                    process.close()
                else:
                    self._api.close(info.hProcess)
                raise
            finally:
                self._api.close(info.hThread)
            return process

    def active_processes(self) -> int:
        with self._lock:
            if self._handle is None:
                if self._tree_exited:
                    return 0
                raise RuntimeError("Closed Job has no confirmed tree-exit observation")
            return self._api.active(self._handle)

    def close(self, timeout=10.0):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        with self._lock:
            if self._handle is None:
                return
            deadline = time.monotonic() + timeout
            try:
                if self._api.active(self._handle):
                    self._api.check(self._api.kernel.TerminateJobObject(
                        self._handle, TERMINATED_EXIT_CODE), "TerminateJobObject")
                while self._api.active(self._handle):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Job tree exit not confirmed; keep ledger UNKNOWN")
                    time.sleep(0.01)
                self._tree_exited = True
            finally:
                # Last-handle close also kills the entire tree on errors/timeouts.
                self._finalizer()
                self._handle = None
            for process in self._processes:
                process.wait(max(0.0, deadline - time.monotonic()))
                process.close()
