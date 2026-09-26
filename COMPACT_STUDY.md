# Three-figure study

This is now the recommended paper workflow. The broad sweep remains an optional API.

```bash
# Print the plan without running simulations:
python -m gle_impact.cli paper-plan
# Check all drivers and plotting on a deliberately small problem:
python -m gle_impact.cli paper --quick --paths 32 --dt 0.05 --output results/quick
# Run the manuscript preset and produce three PDF/PNG figures:
python -m gle_impact.cli paper --paths 512 --dt 0.02 --output results/paper
# Run one experiment, or regenerate plots without simulation:
python -m gle_impact.cli paper --experiment memory --paths 512 --dt 0.02 --output results/paper
python -m gle_impact.cli paper-plot --output results/paper
```

The initial path count and step size are starting settings, not accepted error tolerances. Keep separate directories for refinements. Completed cases are reused only under an identical study manifest; changed settings require a different output directory.

| Figure | Preset | Primary stochastic settings |
|---|---|---:|
| 1. Impact and concavity | 15 logarithmic sizes from 0.01 to 10,000; T=10 and 50; single-mode and GLE; buy orders | 60 |
| 2. Schedules and relaxation | Q=1, T=50; flat/front/back/pause; pause fraction 0.3; GLE; observe to 100 | 4 |
| 3. Prior-history effect | Prior Q=T=1; second Q=T=1; gaps 0, 0.5, 2, 10, 50; single-mode and GLE | 10 |

**74 counts distinct primary settings, not solver calls.** Order/no-order branches, age-matched references, initialization histories, and numerical checks require additional computation. Kyle and fresh-pool curves use the exact formula and scalar DOP853 ODE. The preset does not train 74 PINNs. The PyTorch solver remains available for selected expectation checks.

The schedule pair (Q=1,T=50) is fixed before inspecting stochastic curves. If it lies outside a resolved candidate band, report that outcome; any added schedule case must be identified separately. The 15-point size grid is deliberately coarse across six decades. It locates crossovers; it cannot by itself certify band width or derivative accuracy. Refine near the candidate interval and its edges before quoting a square-root range. Two durations test a directional change, not a general law of width versus duration.

## Saved outputs

- `study_manifest.json`: full immutable run settings.
- `study_plan.json`: case counts and experimental choices.
- `impact/terminal_impact.csv`: impact and standard error for all four variants.
- `impact/curves.json`: exponents and provisional candidate bands.
- `schedules/schedule_summary.csv`: terminal/peak impact, execution-weighted displacement, counterflow and relaxation.
- `memory/memory_summary.csv`: incremental second-order impact, reference, history effect and uncertainty.
- `figures/figure_1_impact.pdf` / `.png` and corresponding Figures 2 and 3.
- Per-case `.npz` files: underlying path ensembles and controls for later refinement and paired resampling.

Figure 1 has impact and exponent panels at each duration. Figure 2 has displacement, mean opposing pool, counterflow, and post-execution normalized displacement. Figure 3 has incremental second-order impact, its relative prior-history effect, and the inherited pool difference. Plotted error bars are approximate 95% pointwise sampling intervals. They exclude time-discretization error; local-exponent uncertainty is not plotted automatically.

## Why Figure 3 has four branches

For each model and gap, evolve a prior-order path and a no-prior-order reference using common innovations from the same finite-start law. At the second-order start, preserve the full joint state, including residual displacement D. From each ensemble, branch into a second-order path and a no-second-order path using common future noise independent of prehistory.

Let A be the terminal displacement difference with versus without the second order, conditional on prior history; let B be the analogous difference after the no-prior history. Report E[A], E[B], E[A-B], and E[A]/E[B]-1. The reference has the **same elapsed time** from initialization. This prevents finite-start transients from being interpreted as a prior-order effect. A ratio is withheld if its denominator is too small relative to its sampling error. The ratio SE uses paired path samples and the delta method. Cross-model contrasts do not receive a paired SE merely because the same seed was used.

This measures the effect of the complete prior state, including residual D. It is not a pure causal attribution to Y alone. The single-mode/GLE comparison isolates a model difference in relaxation spectra at fixed integrated strengths, but can include differences in both inherited D and latent states. A study that resets D would answer a different question and is not the primary preset.

## Refinements and optional extensions

Use `--study-config path.json` to override selected `CompactStudyConfig` fields; other fields retain their defaults. For example:

```json
{"sizes": [0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2, 2, 3, 5, 8, 12, 20],
 "durations": [10, 50]}
```

This is an example of a refinement grid, not a claim that it contains the actual band. Run it with `--experiment impact` in a new output directory. Retain the original broad screening curves to show both crossovers.

The paper should report step/path/observation-grid checks for representative conclusions, selected buy/sell symmetry checks, analytical controls, and the round-trip identity. Add a third duration or further gaps only when needed to resolve an observed ambiguity. Broader spectral, asymmetry, threshold, pool-map, and noise sweeps are optional follow-up studies. No universal robustness claim follows from the compact design.
