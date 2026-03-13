# Parametrized benchmark for Bayesian inversion

## Goal
Following the discussion notes in @benchmark_mmd_bayesian_sampler_notes.txt you are supposed to
describe the parametrized likelyhood and resulting analytical posterior inspired by Ackley function scalled anisotropicaly.
Generalize to the case with a rotation matirx, i.e. anisotropy has form  f(DQx), for a rotation matrix Q and diagonal matrix D, while f is isotropic.
Idea is to use as general function as possible while preserving its separability, allowing for fast evaluation of the MMD matric. 

Write it as a bit more detailet paper:
1. goal
2. reasearch on the topic
3. introduction of the parametrized liekelihood
4. introduce MMD, and its relation to other known metrics
5. efficient calculation of MMD by separation
6. Choosing the MMD kernel.
7. Practical efficient implementation using numpy (and scipy)

## Hard policy
- remove any previous `%DONE: ..` comments in the manuscript when you start your work after given prompt
- interpret comments in form `%CODEX: ... ` as todo notes for your further changes. Turn them into `%DONE: ...` once you take the note into account. 
- keep the newly generated %DONE: comments during processing the single prompt
- Use `biblatex`: `\addbibresource{references.bib}` in `main.tex` and `\printbibliography` at the end.
- Do not ask the user to build the project. If PDF review is needed, inspect the existing generated PDF with read-only Linux command-line tools and then edit the sources.
- Assume the user reviews source changes in `git-cola` before commit; keep patches clean, localized, and easy to inspect.

## Technical scope and contribution boundaries
- Treat the benchmark as a methodology paper: benchmark construction, exact target-side evaluation, and implementation guidance.
- Do not claim novelty for MMD itself, synthetic multimodal targets, or anisotropic benchmark families in general.
- A defensible contribution claim is the combination of a controllable target family with exact or efficiently computable Gaussian-kernel MMD against the exact target, without a reference MCMC chain.
- State assumptions explicitly whenever exact separability, exact quadrature reduction, or closed-form Gaussian manipulations depend on aligned coordinates.

## Writing style
- This is a part of an article draft. Prefer precise, neutral, evidence-based prose over promotional or speculative phrasing.
- Keep language consistent with the existing manuscript: technical English, domain-specific terminology, and explicit statement of assumptions, limitations, and interpretation scope.
- Prefer improving clarity, internal consistency, and evidential support over aggressive shortening. Some chapters are intentionally detailed reference chapters.
- When editing conclusions or summaries, make them traceable to the body text. Do not add claims that are not supported somewhere in the included chapters or cited sources.
- Preserve the current style of chapter-local summaries and interpretation paragraphs; several chapters close with explicit summary sections and `main.tex` contains synthesized conclusion paragraphs.

## Math and notation
- Use `$...$` for inline math and `\[...\]` for displayed math.
- Reuse macros from `main.tex` for notation, especially `\grad`, `\div`, `\vc{}`, `\tn{}`, `\norm{}`, 
- Do not reintroduce raw notation where a project macro already exists.
- Keep notation consistent across chapters, especially for HM, transport, DFN/DFM, EDZ/EIZ, pore pressure, conductivity, and stress quantities.

## Benchmark-specific mathematical guidance
- The manuscript must cover:
  1. benchmark goal and design requirements,
  2. related work,
  3. parametrized likelihood and posterior,
  4. MMD and relations to other metrics,
  5. efficient MMD calculation by separability,
  6. kernel choice,
  7. practical implementation in `numpy` and `scipy`.
- Start from the Ackley-inspired idea, but prefer a separable coordinate-wise construction over the classical radial Ackley term when exact target-side MMD computation is required.
- For the rotated anisotropy extension, use the form $f(DQx)$ with $Q^\top Q = I$ and diagonal $D$.
- Make clear that separability is preserved exactly in rotated coordinates only when the remaining Gaussian factors are expressed compatibly in the same basis.
- If exact one-dimensional quadrature is a core claim, prefer the derivation in coordinates $z = Qx$ and align the Gaussian terms accordingly, e.g. with covariance and kernel bandwidth matrices of the form $Q^\top \operatorname{diag}(\cdot) Q$.
- If a more general prior covariance or kernel orientation is discussed, present it as an extension and state explicitly that full product factorization is then generally lost.

## Citation requirements
- Support MMD definitions and properties with primary literature, especially kernel mean embedding and characteristic-kernel sources.
- Support benchmark-positioning claims with literature on Bayesian sampler benchmarks such as `posteriordb` and modern benchmark suites using MMD or related metrics.
- When comparing against Stein discrepancies or transport-based metrics, keep the comparison limited to supported statements and cite the corresponding primary or standard references.
