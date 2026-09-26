"""Three-figure study: 60 impact + 4 schedule + 10 inherited-state settings.

The count excludes no-order/reference branches, validation, and local refinement.
All primary figures use Monte Carlo; Kyle and fresh controls are analytic/ODE.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import json
import platform
import numpy as np
from .model import Parameters, Model, Variant
from .schedules import Schedule
from .simulation import SimulationConfig, MonteCarlo
from .experiments import ExperimentRunner
from .benchmarks import fresh_ode
from .diagnostics import estimate, local_exponents, largest_band


@dataclass(frozen=True)
class CompactStudyConfig:
    sizes: tuple[float, ...] = tuple(float(x) for x in np.logspace(-2, 4, 15))
    durations: tuple[float, ...] = (10.0, 50.0)
    schedule_Q: float = 1.0
    schedule_T: float = 50.0
    pause_fraction: float = 0.3
    relaxation_extension: float = 50.0
    prior_Q: float = 1.0
    prior_T: float = 1.0
    probe_Q: float = 1.0
    probe_T: float = 1.0
    gaps: tuple[float, ...] = (0.0, 0.5, 2.0, 10.0, 50.0)

    def __post_init__(self):
        if len(self.sizes) < 3 or min(self.sizes) <= 0 or np.any(np.diff(self.sizes) <= 0):
            raise ValueError("Need at least three increasing positive order sizes")
        if len(self.durations) < 1 or min(self.durations) <= 0 or np.any(np.diff(self.durations) <= 0):
            raise ValueError("Durations must be positive and increasing")
        if len(self.gaps) < 1 or min(self.gaps) < 0 or np.any(np.diff(self.gaps) <= 0):
            raise ValueError("Gaps must be nonnegative and increasing")
        if min(self.schedule_Q, self.schedule_T, self.prior_Q, self.prior_T,
               self.probe_Q, self.probe_T) <= 0:
            raise ValueError("This symmetric-baseline preset uses positive order sizes and durations")
        if not 0 < self.pause_fraction < 1 or self.relaxation_extension <= 0:
            raise ValueError("Require 0 < pause_fraction < 1 and positive relaxation extension")
        scalars = [*self.sizes, *self.durations, *self.gaps, self.schedule_Q, self.schedule_T,
                   self.prior_Q, self.prior_T, self.probe_Q, self.probe_T,
                   self.pause_fraction, self.relaxation_extension]
        if not np.isfinite(scalars).all():
            raise ValueError("All study settings must be finite")

    @classmethod
    def quick(cls):
        """Execution/plot check only; does not reproduce manuscript parameters."""
        return cls(sizes=(0.1, 1.0, 10.0), durations=(0.5, 1.0), schedule_T=1,
                   relaxation_extension=1, prior_T=0.5, probe_T=0.5,
                   gaps=(0.0, 0.5, 1.0))

    @classmethod
    def from_json(cls, path):
        obj = json.loads(Path(path).read_text())
        for key in ("sizes", "durations", "gaps"):
            if key in obj: obj[key] = tuple(obj[key])
        return cls(**obj)

    def plan(self):
        counts = {"figure_1_impact": 2*len(self.sizes)*len(self.durations),
                  "figure_2_schedules": 4, "figure_3_memory": 2*len(self.gaps)}
        return {"primary_settings": counts, "total_primary_settings": sum(counts.values()),
                "analytical_controls": "Kyle formula and scalar fresh-pool ODE",
                "excluded_from_count": "no-order/reference branches, refinements and validation",
                "study": asdict(self)}


def _json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False))


def _csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def _ratio_difference(a, b):
    """E[a]/E[b]-1 and paired delta-method SE; denominator resolution checked."""
    am, bm = float(np.mean(a)), float(np.mean(b))
    bse = np.std(b, ddof=1)/np.sqrt(len(b)) if len(b) > 1 else np.inf
    if bm <= max(1e-10, 5*bse):
        return {"mean": None, "se": None, "resolved": False}
    influence = (a-am)/bm-am*(b-bm)/bm**2
    return {"mean": am/bm-1, "se": float(np.std(influence, ddof=1)/np.sqrt(len(a))),
            "resolved": True}


class CompactStudy:
    def __init__(self, output="results/compact", study=None, simulation=None, parameters=None):
        self.output = Path(output)
        self.study = study or CompactStudyConfig()
        self.simulation = simulation or SimulationConfig(paths=512, dt=0.02, observations=101)
        self.parameters = parameters or Parameters()
        if self.simulation.paths < 2:
            raise ValueError("At least two paths are needed for uncertainty reporting")
        if self.parameters.u3 != 0 or any(self.parameters.cross_b):
            raise ValueError("Compact one-sign study requires the symmetric baseline; use general drivers for asymmetry")
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = {"schema_version": 1, "study": asdict(self.study),
                         "simulation": asdict(self.simulation), "parameters": asdict(self.parameters)}
        # JSON roundtrip makes tuples and list-valued input configurations comparable.
        self.manifest = json.loads(json.dumps(self.manifest))
        target = self.output/"study_manifest.json"
        if target.exists() and json.loads(target.read_text()) != self.manifest:
            raise ValueError("Output directory contains a different study. Choose a new --output directory.")
        _json(target, self.manifest)
        _json(self.output/"study_plan.json", self.study.plan())

    def _case(self, directory, label, schedule, variant, horizon=None, initial=None, config=None):
        directory = self.output/directory
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory/f"{label}.json"
        if metadata.exists() and (directory/f"{label}.npz").exists():
            return json.loads(metadata.read_text())
        print(f"Running {directory.name}/{label}", flush=True)
        return ExperimentRunner(self.parameters, config or self.simulation, directory).case(
            label, schedule, variant, horizon, initial)

    def impact(self):
        s = self.study
        directory = self.output/"impact"
        directory.mkdir(parents=True, exist_ok=True)
        rows, curves = [], []
        for ti, T in enumerate(s.durations):
            for variant in Variant:
                means, errors = [], []
                for qi, Q in enumerate(s.sizes):
                    if variant == Variant.KYLE:
                        value, error = Q/self.parameters.L0, 0.0
                        source = "analytic"
                    elif variant == Variant.FRESH:
                        value = float(fresh_ode(self.parameters, Schedule.flat(Q,T), np.array([0.0,T]))[-1])
                        error, source = 0.0, "DOP853"
                    else:
                        meta = self._case("impact", f"{variant}_t{ti:02d}_q{qi:02d}", Schedule.flat(Q,T), variant)
                        value, error = meta["summary"]["terminal"]["mean"], meta["summary"]["terminal"]["se"]
                        source = "monte_carlo"
                    means.append(value); errors.append(error)
                    rows.append({"variant": str(variant), "T": T, "Q": Q,
                                 "impact": value, "se": error, "source": source})
                curves.append({"variant": str(variant), "T": T, "sizes": list(s.sizes),
                               "impact": means, "se": errors,
                               "exponents": _finite_list(local_exponents(s.sizes, means)),
                               "candidate_square_root_band": largest_band(s.sizes, means),
                               "status": "screening_grid_requires_local_refinement_and_uncertainty_checks"})
        _json(directory/"curves.json", curves)
        _csv(directory/"terminal_impact.csv", rows)
        return curves

    def schedules(self):
        s = self.study
        rows = []
        schedules = [Schedule.flat(s.schedule_Q,s.schedule_T),
                     Schedule.front(s.schedule_Q,s.schedule_T),
                     Schedule.back(s.schedule_Q,s.schedule_T),
                     Schedule.pause(s.schedule_Q,s.schedule_T,s.pause_fraction)]
        for schedule in schedules:
            meta = self._case("schedules", schedule.name, schedule, Variant.GLE,
                              s.schedule_T+s.relaxation_extension)
            summary = meta["summary"]
            relaxation = meta["relaxation"]
            rows.append({"schedule": schedule.name, "terminal": summary["terminal"]["mean"],
                         "terminal_se": summary["terminal"]["se"],
                         "peak_mean": summary["peak_mean_response"],
                         "execution_displacement": summary["execution_displacement"]["mean"],
                         "execution_displacement_se": summary["execution_displacement"]["se"],
                         "counterflow_volume": summary["counterflow_volume"]["mean"],
                         "half_life": relaxation.get("crossings", {}).get("0.5"),
                         "residual_ratio": relaxation.get("residual_ratio")})
        _csv(self.output/"schedules"/"schedule_summary.csv", rows)
        _json(self.output/"schedules"/"schedule_summary.json", rows)
        return rows

    def memory(self):
        s, p, cfg = self.study, self.parameters, self.simulation
        rows = []
        directory = self.output/"memory"
        directory.mkdir(parents=True, exist_ok=True)
        prior, probe = Schedule.flat(s.prior_Q,s.prior_T), Schedule.flat(s.probe_Q,s.probe_T)
        for variant in (Variant.SINGLE, Variant.GLE):
            model = Model(p,variant)
            pre_mc = MonteCarlo(model,cfg)
            for gi, gap in enumerate(s.gaps):
                label = f"{variant}_gap{gi:02d}"
                report_path = directory/f"{label}_comparison.json"
                if report_path.exists():
                    rows.append(json.loads(report_path.read_text())); continue
                start = s.prior_T+gap
                # Common prehistory innovations: prior order versus no-prior reference.
                before, no_prior = pre_mc.paired(prior,start)
                future_cfg = replace(cfg,seed=cfg.seed+104729+gi)
                self._case("memory", label+"_prior",probe,variant,initial=before.states[-1],config=future_cfg)
                self._case("memory", label+"_no_prior",probe,variant,initial=no_prior.states[-1],config=future_cfg)
                with np.load(directory/f"{label}_prior.npz") as a, np.load(directory/f"{label}_no_prior.npz") as b:
                    induced_a = a["states_order"][-1,:,0]-a["states_control"][-1,:,0]
                    induced_b = b["states_order"][-1,:,0]-b["states_control"][-1,:,0]
                    rho_a = model.pool(a["states_order"][0,:,1])
                    rho_b = model.pool(b["states_order"][0,:,1])
                interaction = estimate(induced_a-induced_b)
                relative = _ratio_difference(induced_a,induced_b)
                # Store joint samples for paired resampling and conditional-Y analysis.
                np.savez_compressed(directory/f"{label}_samples.npz",incremental=induced_a,
                                    no_prior_incremental=induced_b,inherited_state=before.states[-1],
                                    no_prior_state=no_prior.states[-1])
                a_stats,b_stats = estimate(induced_a),estimate(induced_b)
                rho_diff = estimate(rho_a-rho_b)
                row = {"variant": str(variant),"gap":gap,"elapsed_start":start,
                       "incremental_impact":a_stats["mean"],"incremental_se":a_stats["se"],
                       "no_prior_impact":b_stats["mean"],"no_prior_se":b_stats["se"],
                       "history_effect":interaction["mean"],"history_effect_se":interaction["se"],
                       "relative_history_effect":relative["mean"],"relative_history_se":relative["se"],
                       "relative_resolved":relative["resolved"],
                       "initial_pool_difference":rho_diff["mean"],"initial_pool_difference_se":rho_diff["se"]}
                _json(report_path,row); rows.append(row)
        _csv(directory/"memory_summary.csv",rows)
        _json(directory/"memory_summary.json",rows)
        return rows

    def run(self, experiment="all", plots=True):
        if experiment not in {"all","impact","schedules","memory"}:
            raise ValueError("Unknown experiment")
        for name in ("impact","schedules","memory"):
            if experiment in ("all",name): getattr(self,name)()
        if plots: plot_compact(self.output,experiment)


def plot_compact(output, experiment="all"):
    """Make figure-ready PDF/PNG files from saved data; no simulations are rerun."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(output)
    manifest = json.loads((root/"study_manifest.json").read_text())
    s, p = CompactStudyConfig(**manifest["study"]), Parameters(**manifest["parameters"])
    figdir = root/"figures"
    figdir.mkdir(exist_ok=True)
    colors = {"kyle":"#68717b","fresh":"#ca842b","single":"#278881","gle":"#334fa3"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ("pdf","png"): fig.savefig(figdir/f"{name}.{ext}",dpi=200,bbox_inches="tight")
        plt.close(fig)
    if experiment in ("all","impact"):
        curves=json.loads((root/"impact"/"curves.json").read_text())
        fig,axs=plt.subplots(2,len(s.durations),figsize=(5*len(s.durations),7),squeeze=False)
        for j,T in enumerate(s.durations):
            for row in curves:
                if row["T"]!=T: continue
                x,y,se=np.array(row["sizes"]),np.array(row["impact"]),np.array(row["se"])
                label=row["variant"]; color=colors[label]
                axs[0,j].errorbar(x,y,yerr=1.96*se,label=label,color=color,marker="o",markersize=3,lw=1.2)
                d=np.array([np.nan if v is None else v for v in row["exponents"]])
                axs[1,j].plot(x,d,color=color,marker="o",markersize=3,lw=1.2)
            axs[0,j].set(xscale="log",yscale="log",title=f"T = {T:g}",ylabel="Terminal impact")
            axs[0,j].legend(frameon=False)
            axs[1,j].set(xscale="log",xlabel="Order size Q / Q0",ylabel="Local exponent")
            axs[1,j].axhspan(0.4,0.6,color="#b4beca",alpha=0.25)
            axs[1,j].axhline(0.5,color="grey",ls=":",lw=1)
            for ax in axs[:,j]: ax.grid(alpha=0.15)
        save(fig,"figure_1_impact")
    if experiment in ("all","schedules"):
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        model=Model(p)
        for name in ("flat","front","back","pause"):
            with np.load(root/"schedules"/f"{name}.npz") as data:
                t,J=data["times"],data["J"]
                line=axs[0,0].plot(t,J,label=name)[0]
                color=line.get_color()
                axs[0,0].fill_between(t,J-1.96*data["J_se"],J+1.96*data["J_se"],alpha=.12,color=color)
                axs[0,1].plot(t,data["pool_mean"],label=name,color=color)
                states=data["states_order"]
                cf=model.counterflow(states[:,:,0],states[:,:,1]).mean(1)
                axs[1,0].plot(t,cf,color=color)
                i=np.flatnonzero(np.isclose(t,s.schedule_T))[0]
                if abs(J[i])>max(1e-10,5*data["J_se"][i]):
                    axs[1,1].plot(t[i:]-s.schedule_T,J[i:]/J[i],color=color)
        axs[0,0].set(ylabel="Expected displacement",xlabel="Time")
        axs[0,1].set(ylabel="Mean sell-side pool intensity",xlabel="Time")
        axs[1,0].set(ylabel="Mean counterflow rate",xlabel="Time")
        axs[1,1].set(ylabel="Normalized residual impact",xlabel="Time since completion")
        for ax in (axs[0,0],axs[0,1],axs[1,0]): ax.axvline(s.schedule_T,color="grey",ls=":")
        axs[0,0].legend(frameon=False)
        for ax in axs.flat: ax.grid(alpha=.15)
        save(fig,"figure_2_schedules")
    if experiment in ("all","memory"):
        rows=json.loads((root/"memory"/"memory_summary.json").read_text())
        fig,axs=plt.subplots(1,3,figsize=(13,3.8))
        for variant in ("single","gle"):
            records=sorted((r for r in rows if r["variant"]==variant),key=lambda r:r["gap"])
            x=np.array([r["gap"] for r in records])
            for ax,key,error in zip(axs,("incremental_impact","relative_history_effect","initial_pool_difference"),
                                    ("incremental_se","relative_history_se","initial_pool_difference_se")):
                y=np.array([np.nan if r[key] is None else r[key] for r in records])
                se=np.array([np.nan if r[error] is None else r[error] for r in records])
                ax.errorbar(x,y,yerr=1.96*se,color=colors[variant],marker="o",label=variant,capsize=3)
                ax.set_xscale("symlog",linthresh=.5)
                ax.set_xlabel("Gap after the prior order")
                ax.grid(alpha=.15)
        axs[0].set_ylabel("Incremental second-order impact")
        axs[1].set_ylabel("Relative effect of prior history")
        axs[2].set_ylabel("Initial pool difference")
        axs[1].axhline(0,color="grey",ls=":"); axs[2].axhline(0,color="grey",ls=":")
        axs[0].legend(frameon=False)
        save(fig,"figure_3_memory")
