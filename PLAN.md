# PLAN

## Current objective

Draft the article section on a parametrized Bayesian-sampler benchmark with:

- an Ackley-inspired separable likelihood/posterior family,
- a rotated anisotropy extension $f(DQx)$,
- Gaussian-kernel MMD as the main evaluation metric,
- efficient exact or low-dimensional target-side computation.

## Key decision

Use rotated coordinates $z = Qx$ as the main derivation frame. This keeps the rotated case mathematically clean and preserves separability when the Gaussian prior and Gaussian MMD kernel are aligned with the same basis.

## Work sequence

1. major changes according to the review in the source file -> multiplicative form
2. changes in praticular benchmarks and plots
3. review the PDF and plots in and continue with fixes and improvements 
4. new research of targeted to the benchmark multimodal high dimensional problems


## Deliverables

- manuscript-ready section structure matching the requested seven topics,
- equations for the rotated benchmark and MMD computation,
- bibliography entries in `references.bib`,
- implementation-oriented text for `numpy` and `scipy`.

## Acceptance criteria

- Benchmark admits seprable evaluation of integrals invovled in MMD
- Benchmark likelihood well defined (finite moments)
- Have comparison to published benchmark problems.
- All parameters of the general case are dicussed and used in particular source of difficulter test case. 
- Figures provides real insight to the structure of the likelihood.
