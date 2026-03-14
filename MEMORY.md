# MEMORY

## Project purpose

Article draft on a parametrized benchmark for Bayesian samplers:

- Ackley-inspired but explicitly separable,
- exact or efficiently computable target-side Gaussian-kernel MMD,
- generalized anisotropy through $f(DQx)$,
- practical Python implementation and benchmark figures.

## Current manuscript decisions

- Main manuscript file: `main.tex`.
- Bibliography uses `biblatex` with `references.bib`.
- The basic benchmark is now multiplicative, not additive.
- Basic case:
  - isotropic prior $\pi_0(x)=\mathcal N(x\mid x_0,s_0^2 I)$,
  - common shift $x_0$ for prior and modifier,
  - quadratic central term,
  - common oscillation frequency in the basic case,
  - anisotropy through `diag(lambda)`.
- Generalized case:
  - rotated coordinates $z = Q(x-x_0)$,
  - vector-valued parameters `\vc\lambda`, `\vc p`, `\vc\kappa`, `\vc\omega`,
  - still separable coordinate-wise,
  - prior intentionally kept isotropic so rotation does not break Gaussian separability.
- Additive-form discussion was removed as the primary formulation; multiplicative form is the actual benchmark definition.

## Analytical/computational stance

- Exact target-side MMD remains based on one-dimensional factors.
- Quadratic Gaussian factors are handled analytically.
- Oscillatory factors are expressed via Fourier-Bessel expansion.
- Nonquadratic exponents require one-dimensional quadrature.
- Main theoretical claim: exact separability is preserved when the relevant factors are expressed in the same rotated coordinates.

## Current manuscript structure focus

- Goal and benchmark requirements.
- Related work and positioning.
- Parametrized likelihood/posterior family.
- MMD and relation to nearby metrics.
- Efficient MMD computation by separation.
- Kernel choice.
- Practical implementation in `numpy`/`scipy`.
- Parameter regimes and concrete benchmark templates.

## Physical interpretation currently used

The benchmark examples are now framed consistently around Darcy-flow-type inverse problems:

- low-rank informed subspace,
- multimodal ambiguity in leading modes,
- mixed regularity / sharp vs smooth directions.

Avoid mixing Darcy and elasticity examples unless deliberately reintroduced.

## Repository layout

- `main.tex`: manuscript.
- `references.bib`: bibliography.
- `scripts/benchmark_reference.py`: prototype/reference implementation.
- `scripts/plot_test_function.py`: manuscript figure generator.
- `figures/testcase_pair_matrices.pdf`: main current testcase figure included in the paper.
- `figures/*_pair_matrix.pdf`: per-case figure outputs.

## Figure status

- Old pseudo-3D modifier plots were replaced by 2D pair-matrix contour plots.
- Current main figure in the text: `figures/testcase_pair_matrices.pdf`.
- Plot layout compares active-active, active-inactive, and inactive-inactive coordinate pairs.

## AGENTS.md workflow rules currently important

- Remove previous `%DONE:` comments at the start of new work after a prompt.
- Interpret `%CODEX: ...` comments as actionable todo notes.
- Turn addressed `%CODEX:` notes into `%DONE:` comments during the current prompt.
- Keep those new `%DONE:` comments during that prompt.

## Current residual issues

- `main.tex` still has a tiny overfull heading warning for the long inverse-problem section title.
- The plotting/manuscript pipeline builds successfully.

## Verified commands

- `python3 scripts/benchmark_reference.py`
- `python3 scripts/plot_test_function.py --output-dir figures`
- `latexmk -pdf -interaction=nonstopmode main.tex`

All succeeded in the current workspace state.
