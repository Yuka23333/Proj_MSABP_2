"""CST commands for separately extruded copper regions, including interior holes.

The first region retains the established patch name. Remaining regions live in
an exclusively managed component so a subsequent smaller design leaves no debris.
"""

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from scripts.automation.cst_generate_polygen import (
    build_extrude_curve_vba,
    build_polygon_vba,
)

CONDUCTOR_COMPONENT_CURVE = "msabp_conductor_components"
ISLAND_COMPONENT_SUFFIX = "_msabp_islands"


def component_names(index, component_name, patch_name):
    if index == 0:
        return component_name, patch_name
    return f"{component_name}{ISLAND_COMPONENT_SUFFIX}", f"msabp_island_{index:03d}"


def component_commands(components, component_name, patch_name, material, thickness):
    """Extrude exteriors, subtract holes, then retain every source loop in CST."""
    commands = [
        (
            "create copper island component",
            f'''Sub Main()
    Component.New "{component_name}{ISLAND_COMPONENT_SUFFIX}"
End Sub''',
        )
    ]
    for index, component in enumerate(components):
        owner, solid = component_names(index, component_name, patch_name)
        for ring_index, ring in enumerate((component.exterior, *component.holes)):
            polygon_name = f"copper_{index:03d}_ring_{ring_index:03d}"
            tool = solid if ring_index == 0 else f"{solid}_hole_{ring_index:03d}"
            # Both exterior and hole *tools* must extrude toward +Z.
            points = list(orient(Polygon(ring), sign=1.0).exterior.coords)
            curve_vba = build_polygon_vba(
                points, polygon_name, CONDUCTOR_COMPONENT_CURVE
            )
            commands.extend(
                [
                    (f"create {polygon_name}", curve_vba),
                    (
                        f"extrude {tool}",
                        build_extrude_curve_vba(
                            tool,
                            owner,
                            material,
                            thickness,
                            CONDUCTOR_COMPONENT_CURVE,
                            polygon_name,
                        ),
                    ),
                ]
            )
            if ring_index:
                commands.append(
                    (
                        f"subtract {tool}",
                        f'''Sub Main()
    Solid.Subtract "{owner}:{solid}", "{owner}:{tool}"
End Sub''',
                    )
                )
            commands.append((f"restore {polygon_name}", curve_vba))
    return commands


def component_verification(components, component_name, patch_name, material, thickness):
    """Return VBA checks and accumulate all copper volumes in actualVolume."""
    lines = ["    actualVolume = 0"]
    for index, component in enumerate(components):
        owner, solid = component_names(index, component_name, patch_name)
        ref = f"{owner}:{solid}"
        volume = Polygon(component.exterior, component.holes).area * thickness
        tolerance = max(1e-6, abs(volume) * 1e-6)
        lines.extend(
            [
                f'    If Not Solid.DoesExist("{ref}") Then Err.Raise vbObjectError + 1200, , "Missing copper component {index}"',
                f'    If StrComp(Solid.GetMaterialNameForShape("{ref}"), "{material}", vbTextCompare) <> 0 Then Err.Raise vbObjectError + 1201, , "Copper material mismatch"',
                f'    If Abs(Solid.GetVolume("{ref}") - {volume:.15g}) > {tolerance:.15g} Then Err.Raise vbObjectError + 1202, , "Copper component volume mismatch"',
                f'    actualVolume = actualVolume + Solid.GetVolume("{ref}")',
            ]
        )
        for ring_index in range(1 + len(component.holes)):
            curve = (
                f"{CONDUCTOR_COMPONENT_CURVE}:copper_{index:03d}_ring_{ring_index:03d}"
            )
            lines.append(
                f'    If Not Curve.IsClosed("{curve}") Then Err.Raise vbObjectError + 1203, , "Copper component curve is not closed"'
            )
            if ring_index:
                tool = f"{ref}_hole_{ring_index:03d}"
                lines.append(
                    f'    If Solid.DoesExist("{tool}") Then Err.Raise vbObjectError + 1204, , "Copper hole tool remains"'
                )
    return "\n".join(lines)
