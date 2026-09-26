"""Examples of the research API. Run selected blocks, not all sweeps at once."""
from dataclasses import replace
import numpy as np
from gle_impact import Parameters, Model, Variant, Schedule, MonteCarlo, SimulationConfig
from gle_impact.experiments import ExperimentRunner
from gle_impact.pinn import TorchCoefficients, Observables, Domain, BackwardPINN, PINNConfig


def equilibrium_comparison():
    p=Parameters()
    cfg=SimulationConfig(paths=1024,dt=0.01,seed=42)
    model=Model(p)
    # Finite-start law is the baseline. This is only a candidate stationary ensemble.
    # Compare burn-in lengths and joint moments/covariances before calling it equilibrium.
    ensemble=MonteCarlo(model,cfg).burn_in(100)
    # Future noise must be independent of burn-in innovations.
    runner=ExperimentRunner(p,replace(cfg,seed=43),"results/equilibrium")
    return runner.case("buy",Schedule.flat(1,20),initial=ensemble,horizon=60)


def robustness_and_spectra():
    runner=ExperimentRunner(output="results/robustness")
    runner.sensitivity("u4",(0.05,0.1,0.2))
    runner.sensitivity("u3",(-0.1,0,0.1))
    runner.sensitivity("sigma_y",(0,0.5,1,2))
    runner.sensitivity("sigma_x",(0.5,1,2))
    runner.sensitivity("pi0",(0,0.01,0.1))
    runner.sensitivity("response",("exponential","saturating"))
    # Full planned size grid should follow checks on a smaller grid.
    runner.spectral(np.logspace(-2,3,26),(1,10,50),preserve="integrated")
    runner.spectral(np.logspace(-2,3,26),(1,10,50),preserve="zero_lag")


def full_state_pinn():
    p=replace(Parameters(),cross_b=(0.2,-0.1))
    model=Model(p)
    schedule=Schedule.pause(1,2)
    pilot=MonteCarlo(model,SimulationConfig(paths=256)).paired(schedule,3)
    domain=Domain.from_results(*pilot)
    coef=TorchCoefficients(model,schedule,reduced=False)
    obs=Observables(("D","Y","Y2","pool","h_0","g_0","cf_volume"))
    fit=BackwardPINN(coef,obs,domain,3,PINNConfig(epochs=2000),
                     output_scales=(1,1,1,1,2,2,1)).fit()
    # Repeat for schedule.zero() on this same domain and initial law, then subtract.
    return fit


def duration_scaled_thresholds(Q=1,T=20):
    # Choose tau_d=T explicitly; this changes the model input for each duration.
    # q_star remains fixed here; market-volume scaling is a separate assumption.
    p=replace(Parameters(),tau_d=T)
    return ExperimentRunner(p,output="results/duration_scaled").case("response",Schedule.flat(Q,T))


def inherited_example():
    return ExperimentRunner(output="results/inherited").inherited(prior_Q=1,Q=-1,T=10)
