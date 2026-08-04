"""PyWin32 Windows-service host for the persistent Maid Bell."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.bell import (  # noqa: E402
    MaidBellConfig,
    MaidBellServer,
    default_bell_config_path,
)

try:  # Imported lazily by the Windows service host and installer only.
    import servicemanager
    import win32service
    import win32serviceutil
except ImportError as exc:  # pragma: no cover - depends on Windows service runtime
    raise SystemExit(
        "pywin32 is required for Maid Bell service installation and hosting"
    ) from exc


class MaidBellWindowsService(win32serviceutil.ServiceFramework):
    _svc_name_ = "MSABPMaidBell"
    _svc_display_name_ = "MSABP Maid Bell"
    _svc_description_ = (
        "Receives authenticated Princess wake requests over Tailscale and "
        "starts one device-local CST Maid."
    )

    def __init__(self, args: list[str]) -> None:
        super().__init__(args)
        self.server: MaidBellServer | None = None

    def SvcStop(self) -> None:  # noqa: N802 - Windows service API
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self.server is not None:
            self.server.shutdown()

    def SvcDoRun(self) -> None:  # noqa: N802 - Windows service API
        try:
            config = MaidBellConfig.load(default_bell_config_path())
            log_path = config.repo_root / "logs" / f"maid-bell.{config.device_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=log_path,
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
                encoding="utf-8",
                force=True,
            )
            logging.info(
                "starting Maid Bell device=%s bind=%s:%s",
                config.device_id,
                config.listen_host,
                config.port,
            )
            self.server = MaidBellServer(config)
            servicemanager.LogInfoMsg(
                f"MSABP Maid Bell {config.device_id} listening on "
                f"{config.listen_host}:{config.port}"
            )
            self.server.serve_forever()
        except Exception:
            detail = traceback.format_exc()
            logging.exception("Maid Bell service failed")
            servicemanager.LogErrorMsg(detail)
            raise
        finally:
            if self.server is not None:
                self.server.shutdown()
                self.server = None


def main() -> int:
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(MaidBellWindowsService)
        servicemanager.StartServiceCtrlDispatcher()
        return 0
    return int(win32serviceutil.HandleCommandLine(MaidBellWindowsService) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
