"""Replay frozen audit requests offline; never connect to CST or edit inputs.

Optional --baseline-module compares copper against an archived manufacturing.py
in the same model, in addition to the outcome labels in requests.jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shapely.affinity import translate, scale  # noqa: E402
from shapely.geometry import mapping  # noqa: E402
from scripts.geometry import shapely_antenna_tree_model as model  # noqa: E402
from scripts.geometry import manufacturing as mf  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(row,module):
    original = model.manufacturing
    model.manufacturing = module
    try:
        return model.build(row['params'],row['tree'],**row['build_options'])
    finally:
        model.manufacturing = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--baseline-module',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.requests.read_text(encoding='utf-8').splitlines() if line.strip()]
    meta,rows = records[0]['meta'],records[1:]
    if len(rows) != meta['n'] or len({row['sample'] for row in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate audit samples')
    baseline = None
    if args.baseline_module:
        if digest(args.baseline_module) != meta['model_sha256']['manufacturing.py']:
            raise ValueError('Baseline does not match the original audit hash')
        if digest(model.__file__) != meta['model_sha256']['shapely_antenna_tree_model.py']:
            raise ValueError('Model differs from the original audit; cannot isolate manufacturing change')
        spec = importlib.util.spec_from_file_location('mfg_audit_baseline',args.baseline_module)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    args.output.mkdir(parents=True,exist_ok=True)
    results = []
    start = time.perf_counter()
    for i,row in enumerate(rows):
        record = {'sample':row['sample'],'old_outcome':row['outcome'],'old_reason':row['reason']}
        old = None
        if baseline:
            try:
                old = replay(row,baseline)
                reproduced = 'built'
            except baseline.ManufacturingError:
                reproduced = 'rejected'
            if reproduced != row['outcome']:
                raise RuntimeError(f"Baseline outcome changed: {row['sample']}")
        try:
            geometry = replay(row,mf)
            copper = geometry['Copper']
            offset = -geometry['Substrate_Full'].bounds[1]
            shifted = translate(copper,yoff=offset)
            board = mf.quantize(translate(geometry['Substrate_Full'],yoff=offset),.01)
            mask = mf.quantize(translate(geometry['Manufacturing_Protected'],yoff=offset),.01)
            again,again_report = mf.repair_copper(shifted,board,mask)
            assert again.symmetric_difference(shifted).area < 1e-8 and not again_report['actions'], 'Not idempotent'
            assert copper.is_valid and not copper.is_empty
            assert copper.symmetric_difference(scale(copper,xfact=-1,origin=(0,0))).area < 1e-8
            assert not mf.thin_features(shifted,.1) and not mf.thin_features(board.buffer(.2,join_style=2).difference(shifted),.1)
            assert not mf._proximity_issues(shifted,.1)
            record.update(outcome='built',report=geometry['Manufacturing'],idempotent=True)
            if old is not None:
                record['baseline_copper_changed_area_mm2'] = old['Copper'].symmetric_difference(copper).area
            # All inputs and complete repaired geometry retained for review.
            detail = {'request':row,'report':geometry['Manufacturing'],
                      'copper_raw':mapping(geometry['Copper_Raw']),'copper':mapping(copper)}
        except mf.ManufacturingError as exc:
            record.update(outcome='rejected',report=exc.report)
            detail = {'request':row,'report':exc.report}
        (args.output/f"sample_{row['sample']:03d}.json").write_text(json.dumps(detail,indent=2),encoding='utf-8')
        results.append(record)
        if (i+1)%50 == 0:
            print(f'[audit] {i+1}/{len(rows)}',flush=True)
    summary = {'version':mf.VERSION,'input_sha256':digest(args.requests),'source_meta':meta,
               'source_sha256':{'model':digest(model.__file__),'manufacturing':digest(mf.__file__)},
               'n':len(rows),'old_rejected':sum(r['old_outcome']=='rejected' for r in results),
               'new_rejected':sum(r['outcome']=='rejected' for r in results),
               'rescued_samples':[r['sample'] for r in results if r['old_outcome']=='rejected' and r['outcome']=='built'],
               'regressed_samples':[r['sample'] for r in results if r['old_outcome']=='built' and r['outcome']=='rejected'],
               'previously_built_geometry_changed':[r['sample'] for r in results if r.get('baseline_copper_changed_area_mm2',0)>1e-8],
               'rejection_reasons':dict(Counter(r['report']['reason'] for r in results if r['outcome']=='rejected')),
               'seconds':time.perf_counter()-start,'records':results}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','source_meta')},indent=2))
    return int(bool(summary['regressed_samples']))


if __name__ == '__main__':
    raise SystemExit(main())
