from copy import deepcopy
import json
from pathlib import Path

import pytest
from shapely.affinity import scale, translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from scripts.geometry import manufacturing as mf
from scripts.geometry import shapely_antenna_tree_model as model
from scripts.automation import cst_build_msabp_geometry_tree as builder


def repair(copper, **kwargs):
    return mf.repair_copper(copper,box(-5,-5,5,5),Polygon(),**kwargs)


@pytest.mark.parametrize('width,count',[(.02,2),(.049,2),(.05,1),(.07,1),(.1,1),(.2,1)])
def test_bridge_snap_boundary(width,count):
    copper = unary_union([box(-1,-2,1,-1),box(-width/2,-1,width/2,1),box(-1,1,1,2)])
    result,report = repair(copper)
    assert len(mf.polygons(result)) == count
    assert report['status'] == 'passed'
    assert not mf.thin_features(result,.1)
    assert result.equals(scale(result,xfact=-1,origin=(0,0)))
    again,_ = repair(result)
    assert result.equals(again)


@pytest.mark.parametrize('width,components',[(.02,1),(.049,1),(.05,2),(.07,2),(.1,2)])
def test_gap_snap_boundary(width,components):
    copper = unary_union([box(-2,-1,-width/2,1),box(width/2,-1,2,1)])
    result,_ = repair(copper)
    assert len(mf.polygons(result)) == components
    if components == 2:
        a,b = mf.polygons(result)
        assert a.distance(b) >= .1-1e-8


@pytest.mark.parametrize('size,exists',[(.02,False),(.05,True),(.07,True),(.1,True)])
def test_tiny_square_island(size,exists):
    # A valid large body keeps total copper nonempty if the island is removed.
    copper = box(-1,-3,1,-2).union(box(-size/2,1-size/2,size/2,1+size/2))
    result,_ = repair(copper)
    islands = [p for p in mf.polygons(result) if p.bounds[1]>0]
    assert bool(islands) is exists
    if exists:
        b = islands[0].bounds
        assert b[2]-b[0] >= .1-1e-8 and b[3]-b[1] >= .1-1e-8


def test_small_square_hole_removed_or_widened():
    for size,holes in ((.02,0),(.07,1)):
        result,_ = repair(box(-2,-2,2,2).difference(box(-size/2,-size/2,size/2,size/2)))
        assert len(result.interiors) == holes
        if holes:
            assert Polygon(result.interiors[0]).area >= .01-1e-8


def test_thin_long_island_is_not_an_area_test():
    copper = box(-1,-3,1,-2).union(box(-.01,-1,.01,2))
    result,_ = repair(copper)
    assert len(mf.polygons(result)) == 1  # island area .06 mm2, still illegal width


def test_harmless_short_edge_is_not_a_width():
    copper = Polygon([(-1,-1),(1,-1),(1,1),(.01,1),(-.01,1),(-1,1)])
    assert copper.minimum_clearance < .1
    result,report = repair(copper)
    assert result.equals(copper) and not report['actions']


def test_protected_bridge_is_rejected_not_silently_repaired():
    copper = unary_union([box(-1,-2,1,-1),box(-.01,-1,.01,1),box(-1,1,1,2)])
    with pytest.raises(mf.ManufacturingError,match='protected') as caught:
        mf.repair_copper(copper,box(-5,-5,5,5),box(-.1,-.5,.1,.5))
    assert caught.value.report['status'] == 'rejected'


def test_reject_and_off_modes():
    copper = unary_union([box(-1,-2,1,-1),box(-.01,-1,.01,1),box(-1,1,1,2)])
    with pytest.raises(mf.ManufacturingError,match='sub-minimum'):
        repair(copper,mode='reject')
    result,report = repair(copper,mode='off')
    assert result.equals(copper) and report['status']=='disabled'


def test_diagonal_corner_gap_rejected():
    # Axial separation .08/.02, diagonal separation < .1, no shared scan strip.
    right = box(1,1,2,2).union(box(2.02,2.08,3,3))
    copper = right.union(scale(right,xfact=-1,origin=(0,0)))
    with pytest.raises(mf.ManufacturingError,match='proximity'):
        repair(copper)


@pytest.mark.parametrize('kwargs', [{'threshold':.1},{'threshold':0},{'minimum':float('nan')},
                                   {'quantum':.1},{'mode':'magic'}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        repair(box(-1,-1,1,1),**kwargs)


def test_model_export_share_copper_and_preserve_tree():
    tree = model.default_tree()
    tree['U1']['k'] = [.15,.3,.8]
    model.add_branch(tree,'U1','L',(.7,.3,.998))
    model.add_branch(tree,'U1','L',(.3,.2,.998))
    before = deepcopy(tree)
    params = model.default_params()
    shapes = model.build(params,tree)
    prepared = builder.prepare_geometry({'params':params,'tree':tree})
    exported = unary_union([Polygon(p['exterior'],p['holes']) for p in prepared['copper']])
    from shapely.affinity import translate
    assert exported.equals(translate(shapes['Copper'],yoff=prepared['y_offset_mm']))
    assert shapes['Manufacturing']['after']['components'] > shapes['Manufacturing']['before']['components']
    assert tree == before
    assert prepared['manufacturing']['status']=='passed'
    assert len(prepared['manufactured_copper_sha256']) == 64
    assert json.loads(json.dumps(prepared))['manufacturing']['actions']


def test_tiny_raw_slot_can_collapse_without_invalidating_repaired_export():
    tree = model.default_tree()
    tree['U1']['k'] = [.31,.0001,.5]
    prepared = builder.prepare_geometry({'tree':tree})
    assert prepared['manufacturing']['status'] == 'passed'
    assert any(a['phase']=='gap' and a['action']=='remove' for a in prepared['manufacturing']['actions'])


def test_alternate_snap_threshold():
    copper = unary_union([box(-1,-2,1,-1),box(-.02,-1,.02,1),box(-1,1,1,2)])
    removed,_ = repair(copper)
    widened,_ = repair(copper,threshold=.033)
    assert len(mf.polygons(removed)) == 2
    assert len(mf.polygons(widened)) == 1


def test_runner_rejection_has_audit_and_never_calls_cst(tmp_path,monkeypatch):
    from msabp_opt.simulation.distributed import case_runner_tree as runner
    request = {'tree':model.default_tree(), 'build_options':{'manufacturing_mode':'reject'}}
    request['tree']['U1']['k'] = [.31,.0001,.5]
    def forbidden(*args,**kwargs):
        raise AssertionError('CST must not be opened for a rejected geometry')
    monkeypatch.setattr(runner.exports,'open_cst_project',forbidden)
    with pytest.raises(runner.common.CaseRunError,match='sub-minimum'):
        runner.run_csv_row({'sample_id':'test','tree_json':json.dumps(request)},
                           project_path=tmp_path/'missing.cst',output_root=tmp_path/'out')
    files = list((tmp_path/'out/manufacturing_rejections').glob('*.json'))
    assert len(files) == 1
    assert json.loads(files[0].read_text())['report']['status'] == 'rejected'


def joint_fixture(width=.08):
    # Each open slab is >= .1 wide, but parent/child overlap is only `width`.
    return unary_union([box(-1,-1,1,width),box(1,0,2,.11),box(-2,0,-1,.11)])


@pytest.mark.parametrize('phase',['copper','gap'])
def test_joint_collar_preserves_min_width_and_is_idempotent(phase):
    feature = joint_fixture()
    copper = feature if phase == 'copper' else box(-3,-3,3,3).difference(feature)
    assert not mf.thin_features(copper,.1)
    before = mf._proximity_issues(copper,.1)
    assert before and all(v['phase']==phase for v in before)
    result,report = repair(copper)
    assert report['version']=='tree-manufacturing-v2'
    assert report['joint_repairs']==1
    assert any(a.get('kind')=='joint_cap' for a in report['actions'])
    assert not mf._proximity_issues(result,.1)
    assert not mf.thin_features(result,.1)
    assert not mf.thin_features(box(-5,-5,5,5).difference(result),.1)
    assert result.equals(scale(result,xfact=-1,origin=(0,0)))
    second,again = repair(result)
    assert second.equals(result) and not again['actions']


def test_small_step_not_a_throat_regardless_of_ring_orientation():
    copper = box(-1,-1,1,0).union(box(-.96,0,.96,.08))
    for shape in (copper,copper.reverse()):
        assert not mf._proximity_issues(shape,.1)
        result,report = repair(shape)
        assert result.symmetric_difference(copper).area < 1e-8 and not report['actions']


def test_joint_checker_reject_mode_and_budget_do_not_repair():
    copper = joint_fixture()
    for kwargs,message in [({'mode':'reject'},'proximity'),({'max_joint_repairs':0},'limit')]:
        with pytest.raises(mf.ManufacturingError,match=message) as caught:
            repair(copper,**kwargs)
        assert not caught.value.report['actions']


def test_tiny_joint_needs_explicit_redesign_not_forced_fill():
    with pytest.raises(mf.ManufacturingError,match='explicit redesign'):
        repair(joint_fixture(.03))


def test_joint_collar_respects_feed_mask():
    copper = joint_fixture()
    mask = box(.95,-.1,1.05,.2)
    with pytest.raises(mf.ManufacturingError,match='protected'):
        mf.repair_copper(copper,box(-5,-5,5,5),mask)


def test_joint_caps_are_bounded_and_not_global_dilation():
    copper = joint_fixture()
    result,report = repair(copper)
    cap_action = next(a for a in report['actions'] if a.get('kind')=='joint_cap')
    # Reports use Y measured from the substrate bottom (-5 for this fixture).
    caps = translate(unary_union([box(*b) for b in cap_action['cap_bounds']]),yoff=-5)
    assert result.symmetric_difference(copper).difference(caps).area < 1e-8
    assert result.bounds[:3] == pytest.approx(copper.bounds[:3])
    assert result.bounds[3]-copper.bounds[3] <= .06  # bounded local collar, not global offset


_AUDIT_CASES = json.loads((Path(__file__).parent/'fixtures/manufacturing_corner_audit.json').read_text())['cases']


@pytest.mark.parametrize('case',_AUDIT_CASES,ids=lambda case: f"audit-{case['sample']}")
def test_frozen_paper_audit_regressions(case):
    request = {k:case[k] for k in ('params','tree','build_options')}
    if case['expected'] != 'passed':
        with pytest.raises(mf.ManufacturingError,match=case['expected']):
            model.build(request['params'],request['tree'],**request['build_options'])
        return
    geometry = model.build(request['params'],request['tree'],**request['build_options'])
    copper = geometry['Copper']
    assert copper.is_valid
    assert copper.symmetric_difference(scale(copper,xfact=-1,origin=(0,0))).area < 1e-8
    assert not mf._proximity_issues(copper,.1)
    assert geometry['Manufacturing']['status'] == 'passed'
    assert not mf.thin_features(copper,.1)
    if case['sample'] in (8,22,37):
        assert geometry['Manufacturing']['joint_repairs'] > 0
    if case['sample'] in (122,147,247):
        assert geometry['Manufacturing']['joint_repairs'] == 0
