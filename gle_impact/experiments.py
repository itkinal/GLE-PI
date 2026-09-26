"""Experiment drivers share one model/schedule/simulation API."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import platform
import time
import numpy as np
from .model import Model, Parameters, Variant
from .schedules import Schedule, Segment
from .simulation import MonteCarlo, SimulationConfig
from .diagnostics import response_summary, local_exponents, largest_band, relaxation


class ExperimentRunner:
    def __init__(self, parameters=None, simulation=None, output="results"):
        self.parameters = parameters or Parameters()
        self.simulation = simulation or SimulationConfig()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def case(self, label, schedule, variant=Variant.GLE, horizon=None, initial=None):
        model = Model(self.parameters, variant)
        started = time.perf_counter()
        order, control = MonteCarlo(model, self.simulation).paired(schedule, horizon, initial)
        Jpaths = order.states[:, :, 0]-control.states[:, :, 0]
        J = Jpaths.mean(1)
        se = Jpaths.std(1, ddof=1)/np.sqrt(self.simulation.paths) if self.simulation.paths > 1 else np.full(len(J), np.nan)
        meta = {"label": label, "variant": str(variant), "parameters": asdict(model.p),
                "simulation": asdict(self.simulation), "schedule": asdict(schedule),
                "initialization": "finite_start" if initial is None else "supplied_joint_ensemble",
                "horizon": float(order.times[-1]), "seconds": time.perf_counter()-started,
                "python": platform.python_version(), "numpy": np.__version__,
                "status": "unrefined_numerical_run",
                "summary": response_summary(order, control, schedule)}
        if order.times[-1] > schedule.duration:
            meta["relaxation"] = relaxation(order.times, J, schedule.duration)
        sign = np.sign(schedule.volume())
        rho_order = model.pool(sign*order.states[:, :, 1])
        rho_control = model.pool(sign*control.states[:, :, 1])
        # Save pathwise samples at observation times for bootstrap, quantiles and diagnostics.
        np.savez_compressed(self.output/f"{label}.npz", times=order.times,
                            J=J, J_se=se, states_order=order.states, states_control=control.states,
                            cf_volume_order=order.cf_volume, cf_volume_control=control.cf_volume,
                            trading_cost=order.trading_cost, cf_cost=order.cf_cost,
                            pool_mean=rho_order.mean(1), pool_response=(rho_order-rho_control).mean(1),
                            pool_quantiles=np.quantile(rho_order, [0.05, 0.5, 0.95], axis=1),
                            min_active_pool=order.min_pool, max_active_pool=order.max_pool)
        (self.output/f"{label}.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
        return meta

    def size_duration(self, sizes=None, durations=(1, 3, 10, 20, 50, 100),
                      signs=(-1, 1), variants=tuple(Variant)):
        sizes = np.logspace(-2, 4, 61) if sizes is None else np.asarray(sizes)
        rows, bands = [], []
        for variant in variants:
            for sign in signs:
                for T in durations:
                    impacts = []
                    for i, V in enumerate(sizes):
                        label = f"size_{variant}_{sign}_{T:g}_{i:03d}"
                        meta = self.case(label, Schedule.flat(sign*V, T), variant)
                        I = meta["summary"]["terminal"]["mean"]
                        impacts.append(I)
                        rows.append({"variant": str(variant), "sign": sign, "T": float(T),
                                     "V": float(V), "I": I,
                                     "se": meta["summary"]["terminal"]["se"]})
                    bands.append({"variant": str(variant), "sign": sign, "T": float(T),
                                  "sizes": sizes.tolist(), "impacts": impacts,
                                  "exponents": [float(v) if np.isfinite(v) else None for v in local_exponents(sizes, impacts)],
                                  "square_root_candidate": largest_band(sizes, impacts),
                                  "power_law_candidate": largest_band(sizes, impacts, "power_law")})
        (self.output/"size_duration.json").write_text(json.dumps({"runs": rows, "bands": bands}, indent=2))
        return rows, bands

    def schedules(self, Q=1.0, T=20.0, fraction=0.3, relaxation_horizon=30.0):
        rows = []
        for variant in Variant:
            for schedule in (Schedule.flat(Q,T), Schedule.front(Q,T), Schedule.back(Q,T), Schedule.pause(Q,T,fraction)):
                rows.append(self.case(f"schedule_{variant}_{schedule.name}_{Q:g}_{T:g}",
                                      schedule, variant, T+relaxation_horizon))
        return rows

    def inherited(self, prior_Q, Q, T, gaps=(0, 1, 10, 50), variant=Variant.GLE):
        """Branch after a prior order: retain D,Y,h,g, then subtract no-second-order path.

        This measures incremental impact of the second order in an ongoing history.
        To study only inherited liquidity with a fresh displacement reference, explicitly
        copy the inherited ensemble and set column D to zero before case().
        """
        model = Model(self.parameters, variant)
        prior = Schedule.flat(prior_Q, T)
        mc = MonteCarlo(model, self.simulation)
        rows = []
        for gap in gaps:
            history = mc.run(prior, T+gap, times=np.array([0, T, T+gap]))
            # Use a different seed for future innovations, independent of the prior path.
            runner = ExperimentRunner(self.parameters, replace(self.simulation, seed=self.simulation.seed+1), self.output)
            rows.append(runner.case(f"inherited_{variant}_{prior_Q:g}_{Q:g}_{gap:g}",
                                    Schedule.flat(Q,T), variant, initial=history.states[-1]))
        return rows

    def spectral(self, sizes, durations, mode_counts=(1, 2, 4), breadths=(1, 10, 100), preserve="integrated"):
        summaries = []
        for count in mode_counts:
            for breadth in breadths:
                if count == 1 and breadth != breadths[0]: continue
                p = self.parameters.spectrum(count, count, breadth, breadth, preserve)
                sub = self.output/f"spectrum_{count}_{breadth}_{preserve}"
                ExperimentRunner(p, self.simulation, sub).size_duration(sizes, durations, variants=(Variant.GLE,))
                summaries.append({"modes": count, "requested_breadth": breadth, **p.strength_summary()})
        (self.output/"spectra.json").write_text(json.dumps(summaries, indent=2))
        return summaries

    def sensitivity(self, parameter, values, Q=1.0, T=20.0):
        rows = []
        for i, value in enumerate(values):
            p = replace(self.parameters, **{parameter: value})
            runner = ExperimentRunner(p, self.simulation, self.output/f"sensitivity_{parameter}_{i}")
            for sign in (-1, 1):
                rows.append(runner.case(f"response_{sign}", Schedule.flat(sign*abs(Q), T), horizon=T+30))
        return rows
