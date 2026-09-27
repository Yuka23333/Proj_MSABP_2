import json
from types import SimpleNamespace

import pytest

from scripts.simulation import 本地仿真计时_tree as benchmark


def test_eight_distinct_reproducible_samples():
    raw = benchmark.validation.case_request('case3')
    baseline, samples = benchmark.draw_samples(raw, 8, 42)
    assert benchmark.draw_samples(raw, 8, 42) == (baseline, samples)
    assert len(samples) == 8
    assert len({json.dumps(s['parameters'],sort_keys=True) for s in samples}) == 8
    for sample in samples:
        request = sample['parameters']
        assert benchmark.validation.builder.prepare_geometry(request)['copper_area_mm2'] > 0
        for name, value in request['params'].items():
            old = baseline['params'][name]
            if benchmark.validation.builder.model.PARAM_SPECS[name][0] == 'k':
                assert abs(value-old) <= .05+1e-12 and 0 <= value <= 1
            else:
                assert old*.98-1e-12 <= value <= old*1.02+1e-12


def test_prepare_local_pinned_and_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark,'ROOT',tmp_path)
    template=tmp_path/'test.cst'
    template.write_bytes(b'fixture')
    args=SimpleNamespace(run_id='timing-test',count=8,seed=42,request=None,case='case3',
                         project=template,devices_config=benchmark.validation.comparison.DEVICE_CONFIG)
    plan,folder=benchmark.prepare(args)
    assert benchmark.prepare(args)==(plan,folder)
    command=benchmark.validation.comparison.princess_command(plan,'local',folder)
    assert command.count('--device')==1 and command[command.index('--device')+1]=='local'
    args.seed=43
    with pytest.raises(ValueError,match='Changed plan'):
        benchmark.prepare(args)


def test_summary_mean_and_missing_result(tmp_path):
    params={}
    plan={'count':2,'samples':[{'sample_id':f'sample_{i:03d}','parameters':params} for i in (1,2)]}
    for i in (1,2):
        folder=tmp_path/'results/local'/f'case_sample_{i:03d}'
        folder.mkdir(parents=True)
        (folder/'S11.csv').write_text('3 -10\n4 -20\n5 -10\n')
        (folder/'Farfield Source [1].ffs').write_text('fixture FFS')
        artifacts={key:{'path':filename,'sha256':benchmark.validation.comparison.sha256(folder/filename)}
                   for key,filename in [('s11','S11.csv'),('farfield_source','Farfield Source [1].ffs')]}
        manifest={'status':'completed','dry_run':False,'simulation_mode':'antenna_tree',
                  'case_id':f'sample_{i:03d}','parameters':params,'artifacts':artifacts,
                  'stage_seconds':{'solving':10*i},'elapsed_seconds':10*i+3}
        (folder/'manifest.json').write_text(json.dumps(manifest))
    assert benchmark.summarize(plan,tmp_path)==0
    report=json.loads((tmp_path/'timing_summary.json').read_text())
    assert report['solver']['mean_seconds']==15
    assert report['case_total']['mean_seconds']==18
    assert report['solver_excluding_first']['mean_seconds']==20
    (tmp_path/'results/local/case_sample_002/Farfield Source [1].ffs').write_text('altered')
    assert benchmark.summarize(plan,tmp_path)==2
    report=json.loads((tmp_path/'timing_summary.json').read_text())
    assert report['completed']==1 and report['solver']['n']==1
