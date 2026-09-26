from dataclasses import replace
import json
import numpy as np
import pytest
from gle_impact.compact_study import CompactStudyConfig, CompactStudy, _ratio_difference, plot_compact
from gle_impact import SimulationConfig


def test_default_plan_counts():
    study = CompactStudyConfig()
    assert study.plan()["total_primary_settings"] == 74
    assert study.plan()["primary_settings"] == {
        "figure_1_impact": 60, "figure_2_schedules": 4, "figure_3_memory": 10}
    assert len(study.sizes)==15 and study.durations==(10,50)


def test_matched_reference_and_paired_ratio():
    a=np.array([11.,12.,13.,14.])
    report=_ratio_difference(2*a,a)
    assert report['mean']==1 and report['se']==0
    assert _ratio_difference(a,np.zeros(4))['resolved'] is False


def test_three_figure_pipeline(tmp_path):
    study=CompactStudyConfig.quick()
    cfg=SimulationConfig(paths=8,dt=.05,observations=11,seed=37)
    runner=CompactStudy(tmp_path,study,cfg)
    runner.run(plots=True)
    assert len(json.loads((tmp_path/'impact/curves.json').read_text()))==8
    for name in ('figure_1_impact','figure_2_schedules','figure_3_memory'):
        for suffix in ('pdf','png'):
            assert (tmp_path/'figures'/f'{name}.{suffix}').stat().st_size>1000
    rows=json.loads((tmp_path/'memory/memory_summary.json').read_text())
    assert len(rows)==6
    row=rows[0]
    assert row['elapsed_start']==study.prior_T+row['gap']
    with np.load(tmp_path/'memory/single_gap00_samples.npz') as data:
        assert np.all(data['no_prior_state'][:,0]==0)
        assert np.any(data['inherited_state'][:,0]!=0)
        effect=data['incremental']-data['no_prior_incremental']
        np.testing.assert_allclose(row['history_effect'],effect.mean())
        np.testing.assert_allclose(row['history_effect_se'],effect.std(ddof=1)/np.sqrt(cfg.paths))
    # Resume leaves completed case bytes intact and reproduces summaries.
    before=(tmp_path/'impact/single_t00_q00.npz').read_bytes()
    runner.impact()
    assert (tmp_path/'impact/single_t00_q00.npz').read_bytes()==before
    with pytest.raises(ValueError,match='different study'):
        CompactStudy(tmp_path,study,replace(cfg,dt=.01))
