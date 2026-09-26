from dataclasses import replace
import numpy as np
import torch
from numpy.testing import assert_allclose
from gle_impact import Parameters, Model, Variant, Schedule, MonteCarlo, SimulationConfig
from gle_impact.pinn import (TorchCoefficients, Domain, Observables,
                             BackwardPINN, PINNConfig, generator_value)


def test_generator_has_no_mixed_diffusion():
    model=Model()
    coef=TorchCoefficients(model,Schedule.flat(1,2),reduced=False)
    rng=np.random.default_rng(4)
    tz=coef.tensor(np.column_stack((rng.uniform(0,2,8),rng.normal(size=(8,model.dim))))).requires_grad_(True)
    z=tz[:,1:]
    drift=coef.drift(tz[:,:1],z)
    D,h=z[:,:1],z[:,2:3]
    actual=generator_value(D*h,tz,coef)
    expected=drift[:,:1]*h+D*drift[:,2:3]
    assert_allclose(actual.detach(),expected.detach(),atol=1e-12)
    actual=generator_value(h*h,tz,coef)
    expected=2*h*drift[:,2:3]+coef.sigma[0]**2
    assert_allclose(actual.detach(),expected.detach(),atol=1e-12)


def test_numpy_torch_and_reduction_agree():
    model=Model()
    schedule=Schedule.pause(-2,3)
    coef=TorchCoefficients(model,schedule)
    t=np.array([0.1,0.8,1.3,2.7,4.0])
    z=np.random.default_rng(9).normal(size=(5,coef.dim))
    g=schedule.memory(t,model.p.lam,model.p.c)
    assert_allclose(coef.memory(coef.tensor(t[:,None])).numpy(),g,atol=1e-12)
    full=np.column_stack((z,g))
    assert_allclose(coef.drift(coef.tensor(t[:,None]),coef.tensor(z)).numpy(),model.drift(t,full,schedule)[:,:coef.dim],atol=1e-12)
    p=replace(model.p,cross_b=(0.2,-0.1))
    fullcoef=TorchCoefficients(Model(p),schedule,reduced=False)
    assert_allclose(fullcoef.drift(fullcoef.tensor(t[:,None]),fullcoef.tensor(full)).numpy(),Model(p).drift(t,full,schedule),atol=1e-12)


def test_terminal_and_continuity_and_training():
    model=Model(variant=Variant.KYLE)
    schedule=Schedule.pause(1,1)
    coef=TorchCoefficients(model,schedule)
    domain=Domain(np.full(coef.dim,-1.),np.full(coef.dim,1.))
    obs=Observables(("D","cf_volume"))
    fit=BackwardPINN(coef,obs,domain,1,PINNConfig(width=8,depth=1,epochs=2,batch_size=8,validate_every=1,lbfgs_steps=2)).fit()
    z=np.zeros((3,coef.dim))
    z[:,0]=[-0.3,0.1,0.8]
    assert_allclose(fit.predict(1,z),np.column_stack((z[:,0],np.zeros(3))),atol=0)
    for left,right in zip(fit.blocks[:-1],fit.blocks[1:]):
        tz=coef.tensor(np.column_stack((np.full(3,left.end),z)))
        assert_allclose(left(tz).detach(),right(tz).detach(),atol=0)
    assert np.isfinite(fit.predict(0,z)).all()
