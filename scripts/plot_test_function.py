from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from benchmark_reference import BenchmarkParams, modifier_from_rotated


def darcy_low_rank() -> BenchmarkParams:
    d = 100
    return BenchmarkParams(
        x0=np.zeros(d),
        prior_scales=np.ones(d),
        anisotropy_scales=np.concatenate([np.full(8, 6.0), np.full(d - 8, 0.6)]),
        power_exponents=np.full(d, 2.0),
        cosine_amplitudes=np.ones(d),
        cosine_frequencies=np.full(d, np.pi),
        cosine_phases=np.zeros(d),
        rotation=np.eye(d),
        modifier="multiplicative",
    )


def darcy_multimodal() -> BenchmarkParams:
    d = 128
    lam = np.concatenate([3.0 / np.sqrt(np.arange(1, 21)), np.full(d - 20, 0.15)])
    kappa = np.concatenate([np.full(8, 2.4), np.zeros(d - 8)])
    omega = np.concatenate([np.linspace(1.4, 3.2, 8) * np.pi, np.full(d - 8, np.pi)])
    return BenchmarkParams(
        x0=np.zeros(d),
        prior_scales=np.ones(d),
        anisotropy_scales=lam,
        power_exponents=np.full(d, 2.0),
        cosine_amplitudes=kappa,
        cosine_frequencies=omega,
        cosine_phases=np.zeros(d),
        rotation=np.eye(d),
        modifier="multiplicative",
    )


def darcy_mixed_regularity() -> BenchmarkParams:
    d = 80
    return BenchmarkParams(
        x0=np.zeros(d),
        prior_scales=np.ones(d),
        anisotropy_scales=np.concatenate([np.full(6, 4.5), np.full(6, 2.4), np.full(d - 12, 0.4)]),
        power_exponents=np.concatenate([np.full(6, 2.0), np.full(6, 1.2), np.full(d - 12, 2.0)]),
        cosine_amplitudes=np.concatenate([np.full(6, 1.1), np.zeros(d - 6)]),
        cosine_frequencies=np.concatenate([np.linspace(1.0, 2.2, 6) * np.pi, np.full(d - 6, np.pi)]),
        cosine_phases=np.zeros(d),
        rotation=np.eye(d),
        modifier="multiplicative",
    )


def build_cases():
    return [
        ("darcy_low_rank", "Darcy low-rank", darcy_low_rank(), [0, 1, 8, 9]),
        ("darcy_multimodal", "Darcy multimodal", darcy_multimodal(), [0, 1, 8, 9]),
        ("darcy_mixed_regularity", "Darcy mixed-regularity", darcy_mixed_regularity(), [0, 1, 12, 13]),
    ]


def evaluate_pair(params: BenchmarkParams, dim_i: int, dim_j: int, lim_i: float, lim_j: float, n: int = 80):
    xi = np.linspace(-lim_i, lim_i, n)
    xj = np.linspace(-lim_j, lim_j, n)
    xx, yy = np.meshgrid(xi, xj)
    points = np.zeros((n * n, params.x0.shape[0]))
    points[:, dim_i] = xx.ravel()
    points[:, dim_j] = yy.ravel()
    values = modifier_from_rotated(points, params).reshape(n, n)
    return xi, xj, xx, yy, values


def pair_limits(params: BenchmarkParams, dim_i: int, dim_j: int) -> tuple[float, float]:
    scale_i = 2.5 / max(params.anisotropy_scales[dim_i], 0.5)
    scale_j = 2.5 / max(params.anisotropy_scales[dim_j], 0.5)
    return scale_i, scale_j


def plot_case_matrix(fig, gs, title: str, params: BenchmarkParams, dims: list[int]) -> None:
    pairs = [(dims[0], dims[1]), (dims[0], dims[2]), (dims[1], dims[3]), (dims[2], dims[3])]
    labels = ["active-active", "active-inactive", "active-inactive", "inactive-inactive"]
    for idx, ((i, j), label) in enumerate(zip(pairs, labels)):
        ax = fig.add_subplot(gs[idx])
        lim_i, lim_j = pair_limits(params, i, j)
        xi, xj, xx, yy, values = evaluate_pair(params, i, j, lim_i, lim_j)
        mesh = ax.pcolormesh(xi, xj, values, shading="auto", cmap="YlOrRd")
        ax.contour(xx, yy, values, levels=8, colors="black", linewidths=0.5, alpha=0.8)
        ax.set_xlabel(rf"$z_{{{i+1}}}$", fontsize=8)
        ax.set_ylabel(rf"$z_{{{j+1}}}$", fontsize=8)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.text(
                -0.08,
                1.18,
                title,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)


def build_all_plots(output_dir: Path) -> None:
    cases = build_cases()
    for stem, title, params, dims in cases:
        fig = plt.figure(figsize=(8.2, 7.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.28)
        plot_case_matrix(fig, gs, title, params, dims)
        fig.savefig(output_dir / f"{stem}_pair_matrix.pdf", bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(11.5, 11.5), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, hspace=0.38)
    for row, (_, title, params, dims) in enumerate(cases):
        gs = outer[row].subgridspec(2, 2, wspace=0.28, hspace=0.28)
        plot_case_matrix(fig, gs, title, params, dims)
    fig.savefig(output_dir / "testcase_pair_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pair-matrix contour figures for the benchmark testcases.")
    parser.add_argument("--output-dir", default="figures", help="directory for generated PDF figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_all_plots(output_dir)
