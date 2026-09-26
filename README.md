# GLE latent-liquidity experiments

An object-oriented research implementation of **gle_price_impact7.tex**, updated to the compact plan in **gle_price_impact7_compact.tex**, “A Generalized Langevin Model of Latent Liquidity and Concave Price Impact.” Targets **standard CPython 3.13** and uses **PyTorch** for the backward-equation PINN. All baseline values are copied from the draft; they are not re-estimated.

The recommended workflow is the **three-figure study** in [COMPACT_STUDY.md](COMPACT_STUDY.md): 60 impact settings, four schedule settings, and ten repeated-order settings. It produces the three figures and CSV summary tables directly. See that guide first.

This is a working computational foundation. It does not supply completed or publication-ready experiments. The full sweep, PINN convergence studies, domain expansion, stationary-law verification, and uncertainty analysis of detected bands remain research computations.

## Installation and first runs

From the unpacked project directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
pytest -q
python -m gle_impact.cli smoke --paths 512 --dt 0.02 --output results/smoke
python -m gle_impact.cli pinn --epochs 2000 --lbfgs-steps 100 --output results/pinn
```

On Windows, create the environment using `py -3.13 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. The other commands are unchanged. For a GPU, install the appropriate PyTorch build using the official [installation selector](https://pytorch.org/get-started/locally/), then pass `--device cuda` to the PINN command. Monte Carlo uses vectorized NumPy on the CPU. Standard Python 3.13 is intended; the free-threaded interpreter is not a requirement.

```bash
# Recommended paper plan, then a small execution check and the main study:
python -m gle_impact.cli paper-plan
python -m gle_impact.cli paper --quick --paths 32 --dt 0.05 --output results/quick
python -m gle_impact.cli paper --paths 512 --dt 0.02 --output results/paper
# Optional broader exploratory grid, both signs and all four variants:
python -m gle_impact.cli sweep --paths 512 --output results/pilot
# Optional extensive grid from the earlier draft (not the compact paper plan):
python -m gle_impact.cli sweep --full-grid --paths 4096 --dt 0.01 --output results/full
python examples/plot_sweep.py results/full/size_duration.json
# Flat/front/back/interrupted schedules and subsequent relaxation:
python -m gle_impact.cli schedules --output results/schedules
# Reproduce the included validation report:
python examples/verification.py
```

The optional extensive grid stores path ensembles and can require substantial runtime and disk space. Start with a pilot, choose tolerances, and then set path count, step size and observation resolution. No default is claimed to be adequate for every size or duration.

## Design

| Module / class | Responsibility |
|---|---|
| `model.Parameters` | Immutable baseline, admissibility checks, FDT noise, spectrum normalization |
| `model.Model` | Potential, pool, aggregate counterflow, drift, finite-start law |
| `schedules.Schedule` | Piecewise-affine rates; exact volume and deterministic memory integrals |
| `simulation.MonteCarlo` | Implicit midpoint integration, common-noise order/control runs, joint burn-in |
| `pinn.TorchCoefficients` | Differentiable generator coefficients and deterministic-memory reduction |
| `pinn.Observables` | Terminal functions and running integrands |
| `pinn.BackwardPINN` | Backward PDE, exact terminal/interface constraints, held-out residuals |
| `compact_study.CompactStudy` | Three-figure preset, age-matched prior-history controls, resume checks and plots |
| `experiments.ExperimentRunner` | Size–duration, schedule, inherited-state, spectral and sensitivity runs |
| `diagnostics` | Sampling errors, local exponents, candidate bands, costs and relaxation |
| `benchmarks` | Fresh-pool ODE, global-quadratic impact, linearized stationary covariance |

The two numerical solvers share the parameter and schedule definitions, but implement the dynamics independently in NumPy and PyTorch. Tests compare these implementations.

## Model implemented

The full state is `(D, Y, h_1, ..., h_N, g_1, ..., g_M)`:

\[
F=-U'(Y)-\sum_i a_i h_i+\sum_j g_j,\qquad b=(q-q^{cf})/L_0,
\]
\[
dD=b\,dt,\quad dY=F\,dt,\quad dh_i=(F-\gamma_i h_i)dt-\sigma_{Y,i}dW_i,\quad dg_j=(-\lambda_jg_j+c_j(Y)q)dt.
\]

Here `sigma_Y_i = sigma_y * sqrt(2*gamma_i/a_i)`. There is **no direct diffusion in D or Y**, and no mixed diffusion terms. Ambient volatility enters through `dc = c_d*sigma_x*sqrt(tau_d)`. Simulating an extra independent noise for D would change the paper's model.

The baseline response is `q_star*sign(D)*(abs(D)/dc - 1 + exp(-abs(D)/dc))`, multiplied by `2/(1+exp(sign(D)*Y/y_rho))`. A small-argument series avoids cancellation. The optional atom adds `pi0*D` to the fresh response; `pi0` is in unnormalized response-intensity units. Global-quadratic and saturating controls are also implemented. Custom threshold distributions and pool functions can be added by extending the corresponding NumPy and PyTorch coefficient methods together.

The variants are Kyle (zero counterflow), fresh pool, single-mode pool, and GLE pool. Single-mode rates use the baseline geometric means and preserve integrated kernel strengths. When the dimension changes, each model uses its own FDT-consistent memory initial law with the same `Y0=0`; arbitrary memory vectors from different dimensions are not equated. A state-dependent cross amplitude must be specified explicitly after a spectrum change.

## Monte Carlo details

The baseline finite-start initialization is `D0=Y0=g0=0` and independent `h_i0 ~ N(0, sigma_y²/a_i)`. This makes the colored force stationary; it does **not** make the joint latent process stationary. `burn_in()` preserves the joint `(Y,h,g)` ensemble and is an alternative initialization. Check longer burn-in and moment/covariance stability before treating that ensemble as stationary. Use a new simulation seed for innovations after burn-in or prior-order conditioning.

The drift-implicit midpoint scheme solves a scalar equation for midpoint Y and jointly updates h and g. A second monotone scalar solve gives midpoint D. This avoids explicit-Euler instability from quartic confinement at large order sizes. It is not an exact SDE integrator. Nonmonotone or failed latent solves raise an error rather than clipping states; decrease `dt`, particularly with asymmetric/nonconvex potentials or state-dependent coupling. All schedule switches are integration boundaries. The trapezoidal treatment of deterministic decay also requires time-step refinement for fast modes.

Common random numbers are used for each order/no-order pair. The same seed is reused across sizes, giving correlated sampling errors; uncertainty in fitted exponents should respect that correlation. Different mode counts do not supply a canonical common-noise coupling; do not infer a paired variance for GLE-minus-single from seed equality alone. The supplied dt-refinement runs are not nested Brownian refinements and should be assessed with their reported sampling errors.

Trading cost and counterflow volume use the same midpoint values as the state update. This gives an unusually strong algebraic check of mass balance and of the round-trip cost identity. It does not certify path accuracy by itself. For inherited displacement, the no-second-order control is also integrated against the measured order's rate when computing execution displacement.

## PINN details and a minimal example

For each observable, the solver approximates `u_t + L^q u + r = 0`. It uses `float64`, smooth `tanh` networks, Adam with optional fixed-collocation L-BFGS polishing, and automatic differentiation with `create_graph=True`. The implementation computes only the diagonal second derivatives in h; see [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad/).

With constant `c_j` and deterministic `g_j0`, analytic convolutions remove g from the PINN input. The baseline becomes a **four-state PDE plus time**, rather than six states plus time. For random inherited g or state-dependent amplitudes use `reduced=False`. For a deterministic nonzero inherited g, supply its vector via `g0`.

```python
from gle_impact import Model, Schedule, MonteCarlo, SimulationConfig
from gle_impact.pinn import (
    TorchCoefficients, Observables, Domain, PINNConfig, BackwardPINN,
)

model = Model()
schedule = Schedule.flat(Q=1.0, T=1.0)
pilot_order, pilot_control = MonteCarlo(
    model, SimulationConfig(paths=512)
).paired(schedule)

state_dim = 2 + model.n
domain = Domain.from_results(
    pilot_order, pilot_control, reduced_dim=state_dim, expansion=1.5,
)
observables = Observables(("D", "Y", "pool", "cf_volume"), order_sign=1)
coef = TorchCoefficients(model, schedule, reduced=True)
solver = BackwardPINN(
    coef, observables, domain, horizon=1.0,
    config=PINNConfig(epochs=2000),
    output_scales=(1, 1, 1, 1),
).fit()
initial = pilot_order.states[0, :, :state_dim]
print(solver.average(initial))
```

Solve again with `schedule.zero()` on the **same domain and initial ensemble**, retaining `order_sign`, then subtract. The CLI PINN example does this and compares against a Monte Carlo ensemble with an independent seed. For an independent scientific check, also refine that ensemble's step size and path count.

Supported observables: `D`, `D2`, `Y`, `Y2`, `pool`, zero-indexed `h_0`/`g_0`, `cf_volume`, `cost`, and `cf_cost`. The last three have zero terminal condition and a running source. Add other G/r pairs by subclassing `Observables`. Distributional quantiles and expectations of pathwise maxima require simulation or another formulation; a few PINN moments do not determine them.

Time is split at all rate switches and the end of execution. Each block has the exact form `terminal(z) + (end-t)/(end-start)*output_scale*network(t,z)`. Its terminal function is the next block evaluated at the interface, or G at the observation horizon. Later blocks are frozen during earlier-block training while retaining derivatives with respect to the state. This imposes exact continuity but carries later approximation errors backward. There is no requirement of time-derivative continuity across switches.

The domain combines a pilot cloud with an expanded bounding box. Half of collocation points come from each. No zero/reflecting boundary is invented. A finite domain and low residual do not establish uniqueness or adequate tail coverage. Expand domains, improve collocation, vary networks/seeds, and compare response functionals independently. Cloud rows may recur in residual validation; the box draws are independent, and external MC is the stronger check. For many time segments the exact continuation construction increases autograd work; long schedules may benefit from a separately validated interface approximation.

`average()` reports Monte Carlo **initial-state quadrature error only**, not PINN approximation error. Output scaling and residual scaling are explicit; choose scales for each observable and order range. This first implementation trains per schedule, parameter vector, and observation horizon. It does not provide a global parametric PINN or assert a speed advantage. Repeated horizons are needed for a PINN response curve. `save()` writes weights plus problem metadata; reconstruct identical blocks to apply their state dictionaries.

## Experiments, metrics, and scope

`CompactStudy` generates the three primary figures using Monte Carlo for the stochastic pools and analytical/ODE controls for Kyle and the fresh pool. Its repeated-order comparison includes a no-prior-order reference at the same elapsed time from initialization. `ExperimentRunner` retains optional broader sweeps using Monte Carlo. `BackwardPINN` supplies the general expectation solver and order/control comparison for selected verified cases. The two have not been wrapped in a single interchangeable sweep backend because PINN peak/trajectory diagnostics require training at several horizons. This separation makes the initial implementation usable while keeping the computational cost explicit.

Per-case JSON files record the parameter vector, schedule, numerical settings, seed, runtime, and summary statistics. NPZ files retain time grids, path states, control states, counterflow, costs, pool quantiles, and response sampling errors. Reusing an output label overwrites that case; use separate directories for each refinement. Parameter sweeps are explicit and exceptions remain visible. No results are silently discarded.

The primary exponent uses the draft's centered difference of log impact versus log size. `largest_band()` searches connected intervals using the specified 0.1 tolerance and one-decade minimum, also testing positive slopes and decreasing marginal slopes where required. Boundary exponents are unavailable, so band endpoints lie inside the size grid. Bands are **candidates**: no sampling-error tolerance or bootstrap uncertainty is automatically assigned. Refine the grid and examine whether bands widen across durations before calling them regimes. Keep the full trajectories for paired depletion and memory comparisons.

Peak expected impact is distinct from the mean of pathwise peaks; both are reported on the observation grid. Refine that grid if a maximum matters. Relaxation crossings use linear interpolation; a missing crossing is `null`, not permanent impact. A terminal impact too small to resolve yields an unresolved relaxation diagnostic. The default resolution threshold is numerical, and should be replaced by an uncertainty-based threshold in the final analysis.

`examples/advanced_experiments.py` demonstrates burn-in, state-dependent cross-memory, spectral normalizations, potential/noise/atom sensitivity, duration-scaled thresholds, and inherited order histories. The inherited-history example retains residual D and subtracts a no-second-order control. Reset D to zero only if the intended experiment defines a fresh displacement reference while preserving latent liquidity. For equilibrium conditional impact versus initial Y, bin the joint initial ensemble and compare responses with the corresponding control; do not redraw h independently within Y bins.

Not implemented as dedicated experiment drivers: alternative continuous threshold densities, alternative pool-map shapes, optional reference-only drift, automatic conditional-Y binning, automatic pathwise pool-bound certificates, bootstrap confidence bands, adaptive residual refinement, empirical calibration, and a parameter-amortized PINN. They are extension points rather than hidden assumptions. The reference-only drift diagnostic requires changing D's drift in both the model and the control and is intentionally not folded into the balanced-market baseline.

## Verification

Run `pytest -q` for model algebra, exact schedule volumes/convolutions, FDT variance, spectrum normalization, Kyle impact, round-trip identity, independent fresh-pool ODE, the quadratic benchmark, baseline targeting, small-order asymptotics, symmetry, generator diffusion structure, NumPy/PyTorch agreement, terminal/interface constraints, and a short training execution.

The included `verification_report.json` records an additional fresh-pool step refinement, linearized covariance, stochastic estimates, and a modest trained PINN versus independent Monte Carlo. Consult its actual errors before using the solver for research. These checks validate the foundation; they do not establish all of the draft's proposed numerical conclusions.
