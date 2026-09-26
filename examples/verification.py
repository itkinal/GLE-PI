"""Run from project root: python examples/verification.py"""
from dataclasses import replace
from pathlib import Path
import json
import platform
import numpy as np
import torch
from gle_impact import Parameters, Model, Variant, Schedule, MonteCarlo, SimulationConfig
from gle_impact.benchmarks import fresh_ode, linear_stationary_covariance
from gle_impact.diagnostics import round_trip_identity
from gle_impact.pinn import TorchCoefficients, Domain, Observables, BackwardPINN, PINNConfig


def main():
    torch.set_num_threads(2)
    p=Parameters()
    report={"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__}
    report['linearized_stationary_sd_Y']=float(np.sqrt(linear_stationary_covariance(p)[0,0]))
    report['fresh_pool_refinement']=[]
    q=Schedule.pause(2,3)
    for dt in (0.04,0.02,0.01):
        run=MonteCarlo(Model(replace(p,sigma_y=0),Variant.FRESH),SimulationConfig(paths=1,dt=dt,observations=31)).run(q,6)
        exact=fresh_ode(p,q,run.times)
        report['fresh_pool_refinement'].append({'dt':dt,'max_error':float(abs(run.mean[:,0]-exact).max())})
    cfg=SimulationConfig(paths=1024,dt=0.02,observations=31)
    trip=MonteCarlo(Model(p),cfg).run(Schedule.round_trip(1,1,0.5))
    report['round_trip']=round_trip_identity(trip,p.L0)
    cfg=replace(cfg,paths=512)
    schedule=Schedule.flat(1,1)
    model=Model(p)
    pilot=MonteCarlo(model,cfg).paired(schedule)
    domain=Domain.from_results(*pilot,reduced_dim=2+model.n)
    obs=Observables(('D',))
    check=MonteCarlo(model,replace(cfg,seed=9841)).paired(schedule)
    initial=check[0].states[0,:,:2+model.n]
    fits=[]
    conditional=[]
    for name,sched in [('order',schedule),('control',schedule.zero())]:
        coef=TorchCoefficients(model,sched)
        fit=BackwardPINN(coef,obs,domain,1,PINNConfig(width=32,depth=2,epochs=1500,batch_size=128,validate_every=100,lbfgs_steps=100)).fit()
        conditional.append(fit.predict(0,initial)[:,0])
        fits.append({'case':name,'training_seconds':fit.training_seconds,'last_validation':fit.history[-1]})
    mc=check[0].states[-1,:,0]-check[1].states[-1,:,0]
    estimate=conditional[0]-conditional[1]
    report['pinn_smoke']={'Q':1,'T':1,'estimate':float(estimate.mean()),
                          'initial_quadrature_se':float(estimate.std(ddof=1)/np.sqrt(len(estimate))),
                          'independent_mc':float(mc.mean()),'mc_se':float(mc.std(ddof=1)/np.sqrt(len(mc))),
                          'fits':fits,'status':'smoke comparison only; not a convergence certificate'}
    report['stochastic_timestep_check']=[]
    for dt in (0.04,0.02,0.01):
        out=MonteCarlo(model,replace(cfg,paths=2048,dt=dt,seed=912)).run(schedule)
        vals=out.states[-1,:,0]
        report['stochastic_timestep_check'].append({'dt':dt,'mean':float(vals.mean()),'se':float(vals.std(ddof=1)/np.sqrt(len(vals)))})
    Path('verification_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
