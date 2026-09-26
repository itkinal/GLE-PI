from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from gle_impact import Parameters, Model, Variant, Schedule, MonteCarlo, SimulationConfig
from gle_impact.benchmarks import quadratic_impact, fresh_ode, linear_stationary_covariance
from gle_impact.diagnostics import round_trip_identity, largest_band


def test_schedules_and_memory():
    p = Parameters()
    for schedule in [Schedule.flat(2,3), Schedule.front(2,3), Schedule.back(2,3), Schedule.pause(2,3)]:
        assert_allclose(schedule.volume(), 2)
        t = 1.234
        dt = 1e-5
        derivative = (schedule.memory(t+dt,p.lam,p.c)-schedule.memory(t-dt,p.lam,p.c))/(2*dt)
        assert_allclose(derivative, -np.array(p.lam)*schedule.memory(t,p.lam,p.c)+np.array(p.c)*schedule.rate(t), atol=1e-8)
    assert_allclose(Schedule.round_trip(1,2,0.5).volume(), 0)


def test_fdt_initialization_and_linear_covariance():
    model = Model()
    z = model.finite_start(100000, np.random.default_rng(7))
    force = -z[:,model.h]@np.array(model.p.a)
    assert abs(force.var()-model.p.sigma_y**2*sum(model.p.a)) < 0.008
    covariance = linear_stationary_covariance(model.p)
    assert 0.22 < covariance[0,0] < 0.24


def test_spectrum_controls():
    p = Parameters()
    for kind in ("integrated","zero_lag"):
        other = p.spectrum(4,4,100,100,kind)
        key = "integrated" if kind == "integrated" else "zero_lag"
        for prefix in ("YY_", "YX_"):
            assert_allclose(p.strength_summary()[prefix+key], other.strength_summary()[prefix+key])
    single = Model(p,Variant.SINGLE).p
    assert_allclose(single.gamma, [np.sqrt(0.1)])
    assert_allclose(single.c, [5*np.sqrt(0.4)])


def test_kyle_and_round_trip():
    cfg = SimulationConfig(paths=16,dt=0.02,observations=31)
    q = Schedule.pause(-2,3)
    out = MonteCarlo(Model(variant=Variant.KYLE),cfg).run(q,5)
    assert_allclose(out.states[:,:,0], np.broadcast_to(q.volume(out.times)[:,None],out.states[:,:,0].shape),atol=1e-10)
    out = MonteCarlo(Model(),cfg).run(Schedule.round_trip(2,1,0.2))
    report = round_trip_identity(out,1)
    assert report['max_pathwise_identity_error'] < 1e-8
    assert report['cost']['mean'] >= 0


def test_fresh_independent_ode_and_quadratic():
    cfg=SimulationConfig(paths=2,dt=0.01,observations=31)
    p=Parameters(response="quadratic",sigma_y=0)
    q=Schedule.flat(2,3)
    out=MonteCarlo(Model(p,Variant.FRESH),cfg).run(q)
    assert_allclose(out.mean[-1,0],quadratic_impact(2,3),rtol=3e-6)
    p=replace(p,response="exponential")
    q=Schedule.pause(1,3)
    out=MonteCarlo(Model(p,Variant.FRESH),cfg).run(q,6)
    expected=fresh_ode(p,q,out.times)
    assert_allclose(out.mean[:,0],expected,atol=2e-6)


def test_small_order_symmetry_and_baseline_target():
    p=Parameters(sigma_y=0)
    cfg=SimulationConfig(paths=1,dt=0.005,observations=11)
    positive=MonteCarlo(Model(p),cfg).run(Schedule.flat(1,1))
    negative=MonteCarlo(Model(p),cfg).run(Schedule.flat(-1,1))
    assert_allclose(positive.mean, -negative.mean,atol=1e-10)
    assert 1.0 < positive.mean[-1,1] < 1.03
    Q=1e-3
    out=MonteCarlo(Model(p),cfg).run(Schedule.flat(Q,1))
    assert abs((Q-out.mean[-1,0])/Q**2-1/6) < 0.002


def test_no_order_pair_and_band_detection():
    cfg=SimulationConfig(paths=4,dt=0.02,observations=11)
    order,control=MonteCarlo(Model(),cfg).paired(Schedule.flat(0,1))
    assert_allclose(order.states,control.states,atol=0)
    V=np.logspace(-2,2,41)
    band=largest_band(V,np.sqrt(V))
    assert band is not None and band['width_decades'] > 3
    assert largest_band(V,V) is None
