"""Offline before/after audit of a frozen tree timing plan; no CST or solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.automation import cst_build_msabp_geometry_tree as builder  # noqa: E402
from scripts.geometry import manufacturing as mf  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'simulations/runs/local-timing-tree-8-005/timing_plan.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/processed/manufacturing_validation_20260927')
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    summary = []
    for sample in plan['samples']:
        try:
            prepared = builder.prepare_geometry(sample['parameters'])
            report = prepared['manufacturing']
            (args.output/f"{sample['sample_id']}.json").write_text(json.dumps(prepared,indent=2),encoding='utf-8')
            if sample['sample_id'] == 'sample_008':
                plot_case(prepared,args.output/'sample_008_before_after.png')
        except mf.ManufacturingError as exc:
            report = exc.report
        summary.append({'sample_id':sample['sample_id'],**report})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for r in summary:
        print(r['sample_id'],r['status'],'components',r['before']['components'],'->',r.get('after',{}).get('components'),
              'repairs',len(r['actions']))
    return int(any(r['status'] != 'passed' for r in summary))


def plot_case(prepared,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from scripts.geometry.prepare_tree_topology_cases import paint
    fig,axes = plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for col,(key,title) in enumerate([('copper_raw','Raw geometry'),('copper','Manufacturing projection (0.1 mm)')]):
        for body in prepared[key]:
            polygon = Polygon(body['exterior'],body['holes'])
            for ax in axes[:,col]:
                paint(ax,polygon,'#e6b44d')
        x0,y0,x1,y1 = prepared['substrate_bounds']
        axes[0,col].set(xlim=(x0-1,x1+1),ylim=(y0-1,y1+1),title=f'{title}\n{len(prepared[key])} copper components')
        axes[0,col].plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color='#777777',lw=.7)
        axes[1,col].set(xlim=(-.15,.15),ylim=(24.5,34),title='Symmetry-axis detail (X magnified)')
        axes[0,col].set_aspect('equal')
        axes[1,col].set_aspect('auto')
        for ax in axes[:,col]:
            ax.set(xlabel='X (mm)',ylabel='Y from substrate bottom (mm)')
            ax.axvline(0,color='#334455',lw=.5,ls=':')
            ax.set_facecolor('#edf3f4')
    fig.suptitle('sample_008: two thin copper bridges removed; feed unchanged')
    fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
