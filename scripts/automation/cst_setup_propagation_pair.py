"""Create the mirrored 300 mm MSA-BP propagation pair and port 2 in CST.

The transform and port definitions are transcribed from a CST 2025.2 History
record.  The script deliberately refuses to run when the target component or
connector already exists, so an accidental second invocation cannot create a
third antenna or silently renumber the copied objects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation.cst_generate_polygen import (  # noqa: E402
    execute_project_vba,
    execute_save_project,
    open_cst_project,
)


DEFAULT_PROJECT_PATH = (
    REPOSITORY_ROOT / "simulations" / "models" / "msa-bp-propagation.cst"
)

HISTORY_STEPS: tuple[tuple[str, str], ...] = (
    (
        "transform: mirror Connector",
        """
With Transform
     .Reset
     .Name "Connector"
     .Origin "Free"
     .Center "0", "0", "0"
     .PlaneNormal "0", "1", "0"
     .MultipleObjects "True"
     .GroupObjects "False"
     .Repetitions "1"
     .MultipleSelection "True"
     .Destination ""
     .Material ""
     .AutoDestination "True"
     .Transform "Shape", "Mirror"
End With
""",
    ),
    (
        "transform: mirror component1",
        """
With Transform
     .Reset
     .Name "component1"
     .Origin "Free"
     .Center "0", "0", "0"
     .PlaneNormal "0", "1", "0"
     .MultipleObjects "True"
     .GroupObjects "False"
     .Repetitions "1"
     .MultipleSelection "False"
     .Destination ""
     .Material ""
     .AutoDestination "True"
     .Transform "Shape", "Mirror"
End With
""",
    ),
    (
        "transform: translate Connector_1",
        """
With Transform
     .Reset
     .Name "Connector_1"
     .Vector "0", "300", "0"
     .UsePickedPoints "False"
     .InvertPickedPoints "False"
     .MultipleObjects "False"
     .GroupObjects "False"
     .Repetitions "1"
     .MultipleSelection "True"
     .AutoDestination "True"
     .Transform "Shape", "Translate"
End With
""",
    ),
    (
        "transform: translate component1_1",
        """
With Transform
     .Reset
     .Name "component1_1"
     .Vector "0", "300", "0"
     .UsePickedPoints "False"
     .InvertPickedPoints "False"
     .MultipleObjects "False"
     .GroupObjects "False"
     .Repetitions "1"
     .MultipleSelection "False"
     .AutoDestination "True"
     .Transform "Shape", "Translate"
End With
""",
    ),
    (
        "pick face for port 2",
        'Pick.PickFaceFromId "Connector_1:ConFace", "11"',
    ),
    (
        "define port: 2",
        """
With Port
     .Reset
     .PortNumber "2"
     .Label ""
     .Folder ""
     .NumberOfModes "1"
     .AdjustPolarization "False"
     .PolarizationAngle "0.0"
     .ReferencePlaneDistance "0"
     .TextSize "50"
     .TextMaxLimit "0"
     .Coordinates "Picks"
     .Orientation "positive"
     .PortOnBound "False"
     .ClipPickedPortToBound "False"
     .Xrange "-1.76825", "1.76825"
     .Yrange "308.89", "308.89"
     .Zrange "-1.38725", "2.14925"
     .XrangeAdd "0.0", "0.0"
     .YrangeAdd "0.0", "0.0"
     .ZrangeAdd "0.0", "0.0"
     .SingleEnded "False"
     .WaveguideMonitor "False"
     .Create
End With
""",
    ),
)

PREFLIGHT_VBA = """
Sub Main()
    If Not Solid.DoesExist("component1:msabp_patch_solid") Then
        Err.Raise vbObjectError + 2000, , "source antenna does not exist"
    End If
    If Not Solid.DoesExist("Connector:ConFace") Then
        Err.Raise vbObjectError + 2001, , "source Connector:ConFace does not exist"
    End If
    If Solid.DoesExist("component1_1:msabp_patch_solid") Then
        Err.Raise vbObjectError + 2002, , "target component1_1 already exists"
    End If
    If Solid.DoesExist("Connector_1:ConFace") Then
        Err.Raise vbObjectError + 2003, , "target Connector_1 already exists"
    End If
End Sub
"""

VERIFY_VBA = """
Sub Main()
    If Not Solid.DoesExist("component1_1:msabp_patch_solid") Then
        Err.Raise vbObjectError + 2010, , "mirrored antenna does not exist"
    End If
    If Not Solid.DoesExist("component1_1:msabp_substrate_solid") Then
        Err.Raise vbObjectError + 2011, , "mirrored substrate does not exist"
    End If
    If Not Solid.DoesExist("component1_1:msabp_reflector_solid") Then
        Err.Raise vbObjectError + 2012, , "mirrored reflector does not exist"
    End If
    If Not Solid.DoesExist("Connector_1:ConFace") Then
        Err.Raise vbObjectError + 2013, , "mirrored connector does not exist"
    End If
End Sub
"""


def _as_vba_macro(code: str) -> str:
    """Wrap a CST History body for ``Schematic.execute_vba_code`` projects."""

    stripped = code.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().casefold() == "sub main()"
        and lines[-1].strip().casefold() == "end sub"
    ):
        return stripped
    return f"Sub Main()\n{stripped}\nEnd Sub"

def setup_propagation_pair(
    project_path: str | Path = DEFAULT_PROJECT_PATH,
    *,
    timeout: float = 60.0,
    save_project: bool = True,
) -> None:
    """Apply the recorded CST History to an open or newly opened project."""

    target = Path(project_path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"CST project does not exist: {target}")

    project = open_cst_project(target)
    execute_project_vba(project, "preflight second antenna", PREFLIGHT_VBA, timeout)
    print("[CST] preflight passed")

    for caption, code in HISTORY_STEPS:
        print(f"[CST] executing: {caption}")
        execute_project_vba(project, caption, _as_vba_macro(code), timeout)

    execute_project_vba(
        project,
        "verify second antenna and connector",
        VERIFY_VBA,
        timeout,
    )
    print("[CST] mirrored antenna and connector verified")

    if save_project:
        execute_save_project(project, timeout)
        print(f"[CST] saved: {target}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the mirrored 300 mm MSA-BP pair and port 2."
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_PATH)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_propagation_pair(
        args.project,
        timeout=args.timeout,
        save_project=not args.no_save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
