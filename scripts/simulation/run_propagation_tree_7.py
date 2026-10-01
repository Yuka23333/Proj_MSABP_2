"""Prepare/run the seven reviewed tree Pareto candidates on all three Maids.

Default/F5 only prepares files. --action run --execute asks RUN before CST.
Default project_mode=in_place uses each existing provisioned project directly.
Optional copy_bundle retains full template-copy support, including the sidecar.
Other Princess defaults do not change. Source checks reject stale remote code.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import io
import json
from pathlib import Path, PureWindowsPath
import shutil
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from scripts.optimization import morris_tree_common as common
from scripts.optimization.run_phase2_krvea_tree import freeze_json
from scripts.simulation import princess
from msabp_opt.simulation.distributed import propagation_case_runner_tree as runner
from msabp_opt.simulation.distributed.config import LaunchMode
from msabp_opt.simulation.distributed.runtime import device_run_paths
from msabp_opt.simulation.distributed.transport import run_remote_powershell, _ps_literal, push_file_atomic

CONFIG = ROOT/'configs/simulation/propagation_tree_kevin_7.json'
SOURCE_FILES = (
    'src/msabp_opt/simulation/distributed/propagation_case_runner.py',
    'src/msabp_opt/simulation/distributed/propagation_case_runner_tree.py',
    'scripts/automation/cst_build_msabp_geometry_tree.py',
    'scripts/automation/cst_build_msabp_geometry.py',
    'scripts/automation/cst_generate_polygen.py',
    'scripts/postprocessing/export_propagation_results.py',
    'scripts/postprocessing/export_s21_results_worker.py',
)


def bundle_manifest(template):
    sidecar = template.with_suffix('')
    if not template.is_file() or not sidecar.is_dir():
        raise FileNotFoundError('Both CST and companion directory are required')
    files = [template, *sorted(p for p in sidecar.rglob('*') if p.is_file())]
    if any(p.is_symlink() or getattr(p.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
           for p in [template, sidecar, *sidecar.rglob('*')]):
        raise ValueError('Reparse points not allowed in a template bundle')
    return {'files': [{'path': p.relative_to(template.parent).as_posix(), 'sha256': common.file_hash(p)} for p in files],
            'directories': [sidecar.name, *sorted(p.relative_to(template.parent).as_posix() for p in sidecar.rglob('*') if p.is_dir())]}


def prepare(config):
    folder = ROOT/config['output_directory']
    if config.get('project_mode') not in ('in_place', 'copy_bundle'):
        raise ValueError('Choose in_place or copy_bundle explicitly')
    if (folder/'plan.json').exists():
        plan = common.read_json(folder/'plan.json')
        if plan['sha256'] != common.digest({k: v for k, v in plan.items() if k != 'sha256'}):
            raise ValueError('Plan seal changed')
        if plan['config'] != config or plan['launcher_sha256'] != common.source_hash(__file__):
            raise ValueError('Frozen config/launcher changed')
        if plan['sources'] != {p: common.source_hash(ROOT/p) for p in SOURCE_FILES}:
            raise ValueError('Frozen runner sources changed')
        if common.file_hash(folder/'samples.csv') != plan['csv_sha256']:
            raise ValueError('Frozen worklist changed')
        with (folder/'samples.csv').open(encoding='utf-8', newline='') as stream:
            return folder, plan, list(csv.DictReader(stream))
    selection_path = ROOT/config['selection']
    selection = common.read_json(selection_path)
    if selection['sha256'] != common.digest({k: v for k, v in selection.items() if k != 'sha256'}):
        raise ValueError('Selection snapshot changed')
    selected = [r for r in selection['candidates'] if r['pareto']]
    if [r['rank'] for r in selected] != config['candidate_ranks'] or len(selected) != 7:
        raise ValueError('Expected exactly the seven reviewed Pareto cases')
    template = ROOT/config['template']
    if common.file_hash(template) != config['template_cst_sha256']:
        raise ValueError('Kevin template changed; review and create a new run')
    rows = []
    for r in selected:
        record = r['record']
        if not r['worst_s11_db'] < -7:
            raise ValueError('Candidate no longer passes the S11 threshold')
        payload = {'geometry': r['geometry'], 'request': record['request'],
                   'source_case_id': r['sample_id'], 'manufactured_copper_sha256': record['manufactured_copper_sha256']}
        row = {'sample_id': f"kevin_rank_{r['rank']:02d}", 'simulation_mode': 'propagation_s21',
               'geometry_engine': 'saved_tree_pair_v1', 'candidate_rank': r['rank'],
               'source_case_id': r['sample_id'], 'template_cst_sha256': config['template_cst_sha256'],
               'tree_geometry_json': json.dumps(payload, separators=(',', ':')),
               'tree_geometry_sha256': runner.digest(payload)}
        runner.validate_payload(row)
        rows.append(row)
    sources = {p: common.source_hash(ROOT/p) for p in SOURCE_FILES}
    bundle = bundle_manifest(template)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode('utf-8')
    folder.mkdir(parents=True, exist_ok=True)
    worklist = folder/'samples.csv'
    if worklist.exists() and worklist.read_bytes() != data:
        raise ValueError('Existing worklist changed')
    if not worklist.exists():
        worklist.write_bytes(data)
    manifest = {'config': config, 'selection_sha256': common.file_hash(selection_path),
                'sources': sources, 'bundle': bundle, 'csv_sha256': common.file_hash(worklist),
                'cases': [{k: row[k] for k in ('sample_id','candidate_rank','source_case_id')} for row in rows],
                'launcher_sha256': common.source_hash(__file__)}
    manifest['sha256'] = common.digest(manifest)
    freeze_json(folder/'plan.json', manifest)
    freeze_json(folder/'bundle.json', bundle)
    return folder, manifest, rows


def preflight(devices, manifest, *, check_bundle=True):
    """Read-only source/template comparison on all selected hosts."""
    for device in devices:
        if device.launch_mode is LaunchMode.LOCAL:
            project = ROOT/manifest['config']['template']
            if not project.is_file() or not project.with_suffix('').is_dir():
                raise FileNotFoundError('Missing local propagation project or sidecar')
            if check_bundle and bundle_manifest(project) != manifest['bundle']:
                raise ValueError('Local initial template bundle changed')
            continue
        checks = []
        for relative, sha in manifest['sources'].items():
            path = str(PureWindowsPath(device.repo_root)/relative)
            checks += [f"$text=[IO.File]::ReadAllText({_ps_literal(path)}).Replace(\"`r`n\",\"`n\")",
                       '$h=([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($text)))).Replace("-", "").ToLowerInvariant()',
                       f"if ($h -ne '{sha}') {{ throw 'Code mismatch: {relative}; push/pull required' }}"]
        models = PureWindowsPath(device.repo_root)/PureWindowsPath(manifest['config']['template']).parent
        if not check_bundle:
            project = models/'msa-bp-propagation.cst'
            checks += [f"if (-not (Test-Path -LiteralPath {_ps_literal(str(project))} -PathType Leaf) -or -not (Test-Path -LiteralPath {_ps_literal(str(project.with_suffix('')))} -PathType Container)) {{ throw 'Missing in-place project' }}",
                       "Write-Output 'IN-PLACE RESUME PREFLIGHT PASS'" ]
            result = run_remote_powershell(device, "$ErrorActionPreference='Stop'\n$ProgressPreference='SilentlyContinue'\n"+'\n'.join(checks), action='in-place propagation resume preflight', timeout=90)
            print(device.id, result.stdout.strip(), flush=True)
            continue
        checks += [f'$models={_ps_literal(str(models))}',
                   "$project=Join-Path $models 'msa-bp-propagation.cst'",
                   "$sidecar=Join-Path $models 'msa-bp-propagation'",
                   '$files=@(Get-Item -LiteralPath $project)+@(Get-ChildItem -LiteralPath $sidecar -File -Recurse -Force)',
                   "$entries=@($files | ForEach-Object { @{path=$_.FullName.Substring($models.Length+1).Replace('\\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant()} })",
                   "$dirs=@(Get-Item -LiteralPath $sidecar)+@(Get-ChildItem -LiteralPath $sidecar -Directory -Recurse -Force)",
                   "@{files=$entries;directories=@($dirs | ForEach-Object { $_.FullName.Substring($models.Length+1).Replace('\\','/') })} | ConvertTo-Json -Depth 4 -Compress"]
        result = run_remote_powershell(device, "$ErrorActionPreference='Stop'\n$ProgressPreference='SilentlyContinue'\n"+'\n'.join(checks), action='tree propagation preflight', timeout=90)
        actual = json.loads(result.stdout.strip())
        if ({e['path']: e['sha256'] for e in actual['files']} != {e['path']: e['sha256'] for e in manifest['bundle']['files']}
                or sorted(actual['directories']) != sorted(manifest['bundle']['directories'])):
            raise ValueError(f'{device.id}: template bundle differs from the frozen campaign')
        print(device.id, 'PREFLIGHT PASS (source + full template)', flush=True)


def bundle_copier(folder, manifest):
    """Injection point already supported by Princess; no change to other campaigns."""
    def copy(device, source_path, destination_path, *, overwrite=True):
        source, destination = PureWindowsPath(source_path), PureWindowsPath(destination_path)
        allowed = PureWindowsPath(device.repo_root)/'simulations/runs'/manifest['config']['run_id']/'workers'/device.id
        if not destination.is_relative_to(allowed) or destination.name != 'msa-bp.cst':
            raise ValueError('Unexpected private deployment path')
        if source != PureWindowsPath(device.repo_root)/manifest['config']['template']:
            raise ValueError('Unexpected source template')
        if device.launch_mode is LaunchMode.LOCAL:
            src, dst = Path(str(source)), Path(str(destination))
            if dst.exists() or dst.with_suffix('').exists():
                raise FileExistsError('Refusing to overwrite an existing private model; use recovery generation')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src.with_suffix(''), dst.with_suffix(''))
            shutil.copy2(src, dst)
            for entry in manifest['bundle']['files']:
                rel = Path(entry['path'])
                target = dst if len(rel.parts) == 1 else dst.with_suffix('').joinpath(*rel.parts[1:])
                if common.file_hash(target) != entry['sha256']:
                    raise ValueError('Local bundle copy hash mismatch')
        else:
            remote_manifest = str(destination.parent/'template_bundle.json')
            push_file_atomic(device, folder/'bundle.json', remote_manifest, overwrite=True)
            script = f'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$src={_ps_literal(str(source))}
$dst={_ps_literal(str(destination))}
$srcDir=[IO.Path]::Combine([IO.Path]::GetDirectoryName($src),[IO.Path]::GetFileNameWithoutExtension($src))
$dstDir=[IO.Path]::Combine([IO.Path]::GetDirectoryName($dst),[IO.Path]::GetFileNameWithoutExtension($dst))
if ((Test-Path -LiteralPath $dst) -or (Test-Path -LiteralPath $dstDir)) {{ throw 'Private model exists; use recovery generation' }}
Copy-Item -LiteralPath $srcDir -Destination $dstDir -Recurse
Copy-Item -LiteralPath $src -Destination $dst
$manifest=Get-Content -LiteralPath {_ps_literal(remote_manifest)} -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {{
    $parts=$entry.path.Split('/')
    $target=$dst
    if ($parts.Length -gt 1) {{ $target=Join-Path $dstDir ($parts[1..($parts.Length-1)] -join '\') }}
    if ((Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant() -ne $entry.sha256) {{ throw 'Copied bundle hash mismatch' }}
}}
Write-Output 'FULL BUNDLE COPY PASS'
'''
            print(run_remote_powershell(device, script, action='copy complete propagation bundle', timeout=120).stdout, flush=True)
    return copy


class InPlacePrincessRuntime(princess.PrincessRuntime):
    """Deploy worklist/runtime only; never copy, upload or replace the CST project."""
    def deploy_device(self, device, *, launch_generation=None, force_project=False):
        paths = device_run_paths(device, self.preparation.paths.run_id, launch_generation=launch_generation)
        paths = replace(paths, project_path=str(PureWindowsPath(device.repo_root)/self.device_project_relative_path))
        staged = self._stage_runtime(device, paths)
        self._push_file(device, self.preparation.paths.worklist_csv, paths.csv_path, overwrite=True)
        self._push_file(device, staged, paths.runtime_config_path, overwrite=True)
        return paths, staged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--action', choices=['prepare','preflight','run'], default='prepare')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = common.read_json(args.config)
    folder, manifest, rows = prepare(config)
    print(json.dumps(manifest['cases'], indent=2), flush=True)
    if args.action == 'prepare':
        for row in rows:
            runner.run_csv_row(row, project_path=ROOT/config['template'], output_root=folder/'dry_run', dry_run=True, overwrite=True)
        print(f'PREPARED: {folder}; 7 offline cases passed; no CST started.', flush=True)
        return 0
    registry = princess.load_device_registry(ROOT/config['devices_config'])
    devices = princess.select_devices(registry, config['devices'])
    # After first dispatch the in-place project/sidecar are expected to change.
    # Original provenance stays sealed; resume verifies code and existence,
    # while each build checks the baseline history, Kevin solid and ports.
    launch_marker = folder/'inplace_launch_authorized.json'
    check_bundle = config['project_mode'] != 'in_place' or not launch_marker.exists()
    if launch_marker.exists() and common.read_json(launch_marker) != {'plan_sha256': manifest['sha256']}:
        raise ValueError('In-place launch marker belongs to another plan')
    preflight(devices, manifest, check_bundle=check_bundle)
    if args.action != 'run':
        return 0
    if not args.execute or input('Type RUN to start/resume 7 Kevin propagation solves on all three Maids: ').strip() != 'RUN':
        return 0
    print('[Princess] Please verify that every Maid can open CST and solve; check License if CST_DE is unreachable.', flush=True)
    preparation = princess.prepare_run(source_csv=folder/'samples.csv', run_id=config['run_id'],
        registry=registry, devices=devices, repository_root=ROOT, results_root=ROOT/config['results_root'])
    runtime_class = InPlacePrincessRuntime if config['project_mode'] == 'in_place' else princess.PrincessRuntime
    runtime = runtime_class(preparation=preparation, registry=registry, devices=devices,
        project_template=ROOT/config['template'], device_project_relative_path=config['template'],
        copy_device_file=bundle_copier(folder, manifest), command_timeout_seconds=config['command_timeout_seconds'],
        max_attempts=config['max_attempts'])
    try:
        runtime.start_server()
        runtime.state.resume_run(config['run_id'])
        if config['project_mode'] == 'in_place':
            freeze_json(launch_marker, {'plan_sha256': manifest['sha256']})
        for deployment in runtime.start_workers():
            print(f'[Princess] Maid {deployment.device.id} awake, pid={deployment.launch.pid}', flush=True)
        result = runtime.monitor()
        print(json.dumps(result, indent=2), flush=True)
        return 0 if int(result['failed']) == 0 else 2
    finally:
        runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
