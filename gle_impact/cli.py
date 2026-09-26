from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .model import Parameters, Model, Variant
from .schedules import Schedule
from .simulation import MonteCarlo, SimulationConfig
from .experiments import ExperimentRunner
from .diagnostics import round_trip_identity


def main():
    parser = ArgumentParser(description="GLE impact research scaffold")
    parser.add_argument("command", choices=["paper", "paper-plan", "paper-plot", "smoke", "sweep", "schedules", "pinn"])
    parser.add_argument("--output", default="results")
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--experiment", choices=["all", "impact", "schedules", "memory"], default="all")
    parser.add_argument("--quick", action="store_true", help="Small execution/plot check, not manuscript settings")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--study-config", help="JSON overrides of CompactStudyConfig defaults")
    parser.add_argument("--observations", type=int, default=101)
    args = parser.parse_args()
    cfg = SimulationConfig(paths=args.paths, dt=args.dt, observations=args.observations, seed=args.seed)
    if args.command.startswith("paper"):
        from .compact_study import CompactStudy, CompactStudyConfig, plot_compact
        if args.quick and args.study_config:
            parser.error("Choose either --quick or --study-config")
        study = (CompactStudyConfig.from_json(args.study_config) if args.study_config
                 else CompactStudyConfig.quick() if args.quick else CompactStudyConfig())
        if args.command == "paper-plan":
            print(json.dumps(study.plan(), indent=2))
        elif args.command == "paper-plot":
            plot_compact(args.output, args.experiment)
        else:
            print(json.dumps(study.plan(), indent=2), flush=True)
            CompactStudy(args.output, study, cfg).run(args.experiment, plots=not args.no_plots)
        return
    runner = ExperimentRunner(simulation=cfg, output=args.output)
    if args.command == "smoke":
        reports = []
        for variant in Variant:
            reports.append(runner.case(f"smoke_{variant}", Schedule.flat(1, 1), variant, horizon=3))
        trip = MonteCarlo(Model(), cfg).run(Schedule.round_trip(1, 1, 0.5))
        identity = round_trip_identity(trip, 1)
        Path(args.output, "round_trip.json").write_text(json.dumps(identity, indent=2))
        print(json.dumps({r["variant"]: r["summary"]["terminal"] for r in reports}, indent=2))
        print("Round-trip identity:", identity)
    elif args.command == "sweep":
        runner.size_duration() if args.full_grid else runner.size_duration(np.logspace(-2, 2, 9), (1, 10, 20))
    elif args.command == "schedules":
        runner.schedules()
    else:
        from .pinn import TorchCoefficients, Observables, Domain, PINNConfig, BackwardPINN
        import torch
        torch.set_num_threads(4)
        model, schedule = Model(), Schedule.flat(1, 1)
        mc = MonteCarlo(model, cfg)
        order, control = mc.paired(schedule)
        dim = 2+model.n
        domain = Domain.from_results(order, control, reduced_dim=dim)
        observable = Observables(("D", "cf_volume"))
        check_order, check_control = MonteCarlo(model, replace(cfg, seed=cfg.seed+104729)).paired(schedule)
        initial = check_order.states[0, :, :dim]
        fits, means = [], []
        for name, sched in (("order", schedule), ("control", schedule.zero())):
            coef = TorchCoefficients(model, sched, device=args.device)
            fit = BackwardPINN(coef, observable, domain, 1,
                               PINNConfig(epochs=args.epochs, device=args.device, seed=args.seed, lbfgs_steps=args.lbfgs_steps)).fit()
            means.append(fit.average(initial)["mean"])
            fit.save(Path(args.output)/f"pinn_{name}.pt")
            fits.append({"case": name, "training_seconds": fit.training_seconds, "history": fit.history})
        report = {"observable_names": observable.names,
                  "pinn_order_minus_control": (means[0]-means[1]).tolist(),
                  "mc_impact": float((check_order.states[-1,:,0]-check_control.states[-1,:,0]).mean()),
                  "mc_impact_se": float((check_order.states[-1,:,0]-check_control.states[-1,:,0]).std(ddof=1)/np.sqrt(cfg.paths)),
                  "fits": fits, "status": "requires_domain_training_and_MC_refinement"}
        Path(args.output, "pinn_comparison.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != "fits"}, indent=2))


if __name__ == "__main__":
    main()
