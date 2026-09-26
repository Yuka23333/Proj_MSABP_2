"""Opt-in arbitrary-tree Maid pipeline, sharing only exports/protocol with legacy."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from . import case_runner as common
from scripts.automation import cst_build_msabp_geometry_tree as builder
from scripts.automation import cst_run_and_export_s11 as exports

SIMULATION_MODE = 'antenna_tree'


def source_hash(path):
    # Git checkouts may use LF or CRLF; compare source content, not line endings.
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()


def run_csv_row(row, *, project_path, output_root, project=None, case_id=None,
                id_width=4, coordinate_quantum_mm=.01, allow_disconnected_conductor=False,
                command_timeout=15, overwrite=False, save_project_after_case=False,
                dry_run=False, stage_callback=None, local_artifact_root=None):
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = common._case_id_from_row(row, case_id)
    stage = 'precheck'
    opened_here = False
    def notify(name):
        nonlocal stage
        stage = name
        common._notify(stage_callback, name)
    try:
        if 'geometry_valid' in row and not common._parse_csv_bool(row['geometry_valid'], 'geometry_valid'):
            raise ValueError('CSV marks geometry invalid')
        raw = json.loads(row['tree_json'])
        prepared = builder.prepare_geometry(raw, coordinate_quantum_mm)
        request = prepared['request']
        path = Path(project_path).resolve()
        folder = Path(output_root).resolve() / common._case_directory_name(resolved, id_width)
        if folder.exists() and any(folder.iterdir()) and not overwrite:
            raise FileExistsError(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'geometry_tree.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')
        s11 = folder/common.S11_FILENAME
        ffs = folder/common.FARFIELD_SOURCE_FILENAME
        artifacts = {}
        geometry_report = {'copper_components': len(prepared['copper']), 'copper_holes': prepared['copper_holes']}
        if not dry_run:
            if not path.is_file():
                raise FileNotFoundError(path)
            if project is None:
                notify('opening_project')
                project = exports.open_cst_project(str(path))
                opened_here = True
            notify('clearing_results')
            exports.clear_results_on_project(project, timeout=command_timeout)
            notify('building_tree_geometry')
            geometry_report = builder.build_on_project(project, prepared, folder, command_timeout)
            notify('checking_simulation_setup')
            exports.inspect_recorded_simulation_setup(project, command_timeout)
            source = common.project_farfield_source_path(path)
            before = common._file_generation_signature(source)
            exports.solve_and_export_s11_on_project(project, s11, overwrite=overwrite,
                command_timeout=command_timeout, save_project=save_project_after_case,
                clear_results=False, stage_callback=notify)
            notify('copying_farfield_source')
            if not source.is_file() or not source.stat().st_size or common._file_generation_signature(source) == before:
                raise RuntimeError('Missing or stale FFS after current solve')
            shutil.copy2(source, ffs)
            for key, filename in [('s11',common.S11_FILENAME),('rad_eff',common.RAD_EFF_FILENAME),
                                  ('tot_eff',common.TOT_EFF_FILENAME),('farfield_source',common.FARFIELD_SOURCE_FILENAME)]:
                artifacts[key] = common._artifact_record(folder/filename, folder)
            artifacts['geometry_tree'] = common._artifact_record(folder/'geometry_tree.json', folder)
        manifest = {'schema_version': 1, 'simulation_mode': SIMULATION_MODE, 'case_id': resolved,
                    'status': 'dry_run' if dry_run else 'completed', 'dry_run': dry_run,
                    'started_at_utc': started_at, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'elapsed_seconds': time.perf_counter()-started, 'project_path': str(path),
                    'parameters': request, 'geometry': geometry_report, 'artifacts': artifacts,
                    'geometry_sha256': common.sha256_file(folder/'geometry_tree.json'),
                    'tree_request_sha256': hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
                    'baseline_history_sha256': hashlib.sha256(json.dumps([
                        {'name': e['name'], 'contents': e['contents'].replace('\r\n', '\n')}
                        for e in geometry_report.get('baseline_history', [])
                    ], sort_keys=True).encode()).hexdigest(),
                    'source_sha256': {'builder': source_hash(builder.__file__),
                                      'model': source_hash(builder.model.__file__),
                                      'runner': source_hash(__file__)}}
        notify('writing_manifest')
        manifest_path = folder/common.MANIFEST_FILENAME
        common._write_manifest(manifest_path, manifest)
        notify('completed')
        return common.CaseRunResult(case_id=resolved, case_directory=folder, manifest_path=manifest_path,
            s11_path=None if dry_run else s11, farfield_source_path=None if dry_run else ffs,
            dry_run=dry_run, elapsed_seconds=manifest['elapsed_seconds'],
            rad_eff_path=None if dry_run else folder/common.RAD_EFF_FILENAME,
            tot_eff_path=None if dry_run else folder/common.TOT_EFF_FILENAME, simulation_mode=SIMULATION_MODE)
    except Exception as exc:
        raise common.CaseRunError(resolved, stage, str(exc)) from exc
    finally:
        if opened_here:
            project.close()
