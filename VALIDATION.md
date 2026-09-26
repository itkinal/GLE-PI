# Validation performed for this delivery

Environment: CPython 3.13.15, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.0+cpu.

- Ten tests cover model algebra, analytical benchmarks, independent ODE comparison, generator structure, terminal/interface constraints, and short Adam/L-BFGS execution.
- The CLI smoke run completed for all four variants.
- A five-size sweep and two inherited-state cases completed.
- Noise-free GLE simulations at Q=0.01 and Q=10,000, T=1, remained finite. This is only an execution check, not a large-order convergence study.

Selected measured results from `verification_report.json`:

| Check | Result |
|---|---:|
| Linearized no-order stationary standard deviation of Y | 0.4797857 |
| Fresh-pool maximum error against independent DOP853, dt=0.04 | 3.64e-5 |
| Same, dt=0.02 | 9.45e-6 |
| Same, dt=0.01 | 2.44e-6 |
| Maximum pathwise round-trip identity discrepancy | 7.00e-12 |
| GLE MC terminal impact, Q=T=1, 512 independent check paths | 0.902496 |
| MC standard error | 0.000321 |
| PINN estimate after 1,500 Adam epochs and 100 L-BFGS iterations | 0.901246 |

The refined PINN differs from Monte Carlo by about **0.14%**. That gap still exceeds the MC standard error, so this is not a claim of convergence to Monte Carlo sampling precision. The reported initial-state quadrature error also excludes PDE approximation error.

For comparison, the initial 600-epoch Adam-only run produced 0.886296 (about 1.8% low). Its output is retained in `verification_report_initial.json`. The improvement establishes that additional optimization mattered in this case; it does not replace domain-expansion, architecture, seed, and tail-coverage checks.

The midpoint scheme's round-trip identity is close to algebraically exact because the state and cost use the same quadrature. Its independent fresh-pool ODE errors and stochastic step-refinement checks provide separate information about path accuracy.

No full-grid scientific conclusions, empirical calibration, broad square-root-band claims, or PINN efficiency advantage are asserted by this delivery.

## Compact-study update (version 0.2.0)

All **13 tests passed** under CPython 3.13.15. Three added tests check the 74-setting plan, the paired ratio statistic, and the complete reduced study including all three PDF/PNG figures, age-matched history controls, CSV/JSON summaries, restart behavior, and rejection of mixed output configurations.

The CLI was also run with `paper --quick --paths 16 --dt 0.05 --observations 31`. Its deliberately shortened horizons and small ensembles test execution and presentation only; they are not manuscript results. The full 74-setting preset has not been run for this update. The original PINN comparison above remains a separate single-case validation, not evidence for the three proposed figures.
