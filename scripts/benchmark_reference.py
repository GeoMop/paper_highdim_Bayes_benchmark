from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.special import iv


ModifierKind = Literal["additive", "multiplicative"]
ComponentKind = Literal["power", "cos"]


@dataclass(frozen=True)
class BenchmarkParams:
    x0: np.ndarray
    prior_scales: np.ndarray
    anisotropy_scales: np.ndarray
    power_exponents: np.ndarray
    cosine_amplitudes: np.ndarray
    cosine_frequencies: np.ndarray
    cosine_phases: np.ndarray
    rotation: np.ndarray
    a: float = 1.0
    modifier: ModifierKind = "additive"
    bessel_terms: int = 40

    def __post_init__(self) -> None:
        d = self.x0.shape[0]
        for name, value in (
            ("prior_scales", self.prior_scales),
            ("anisotropy_scales", self.anisotropy_scales),
            ("power_exponents", self.power_exponents),
            ("cosine_amplitudes", self.cosine_amplitudes),
            ("cosine_frequencies", self.cosine_frequencies),
            ("cosine_phases", self.cosine_phases),
        ):
            if value.shape != (d,):
                raise ValueError(f"{name} must have shape (d,)")
        if self.rotation.shape != (d, d):
            raise ValueError("rotation must have shape (d, d)")


def standard_normal_density(x: float, mean: float, scale: float) -> float:
    inv = 1.0 / scale
    return inv * np.exp(-0.5 * ((x - mean) * inv) ** 2) / np.sqrt(2.0 * np.pi)


def transform_to_rotated(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points)
    return points @ rotation.T


def aligned_bandwidth(rotation: np.ndarray, kernel_scales: np.ndarray) -> np.ndarray:
    return rotation.T @ np.diag(kernel_scales ** 2) @ rotation


def gaussian_kernel_matrix(points: np.ndarray, bandwidth_matrix: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    precision = np.linalg.inv(bandwidth_matrix)
    quad_form = np.einsum("...i,ij,...j->...", diff, precision, diff)
    return np.exp(-0.5 * quad_form)


def rotated_mean(params: BenchmarkParams) -> np.ndarray:
    return params.rotation @ params.x0


def gaussian_product_parameters(mean: float, scale: float, probe: float, kernel_scale: float) -> tuple[float, float, float]:
    sigma_sq = 1.0 / (kernel_scale ** -2 + scale ** -2)
    mu = sigma_sq * (probe / (kernel_scale ** 2) + mean / (scale ** 2))
    prefactor = kernel_scale / np.sqrt(kernel_scale ** 2 + scale ** 2)
    prefactor *= np.exp(-0.5 * (probe - mean) ** 2 / (kernel_scale ** 2 + scale ** 2))
    return prefactor, mu, np.sqrt(sigma_sq)


def quadratic_power_expectation(mean: float, scale: float, lam: float, d: int) -> float:
    alpha = lam ** 2 / d
    denom = 1.0 + 2.0 * alpha * scale ** 2
    return np.exp(-(alpha * mean ** 2) / denom) / np.sqrt(denom)


def cosine_expectation_series(
    mean: float,
    scale: float,
    amplitude: float,
    frequency: float,
    phase: float,
    bessel_terms: int,
) -> float:
    beta = amplitude
    series = iv(0, beta)
    for n in range(1, bessel_terms + 1):
        series += 2.0 * iv(n, beta) * np.cos(n * (frequency * mean + phase)) * np.exp(
            -0.5 * (n * frequency * scale) ** 2
        )
    return float(series)


def _power_expectation(mean: float, scale: float, lam: float, exponent: float, d: int) -> float:
    if np.isclose(exponent, 2.0):
        return quadratic_power_expectation(mean, scale, lam, d)
    return quad(
        lambda x: standard_normal_density(x, mean, scale) * np.exp(-(abs(lam * x) ** exponent) / d),
        -np.inf,
        np.inf,
        limit=200,
    )[0]


def _cos_expectation(
    mean: float,
    scale: float,
    lam: float,
    amplitude: float,
    frequency: float,
    phase: float,
    d: int,
    bessel_terms: int,
) -> float:
    beta = amplitude / d
    omega = frequency * lam
    return cosine_expectation_series(mean, scale, beta, omega, phase, bessel_terms)


def modifier_from_rotated(z: np.ndarray, params: BenchmarkParams) -> np.ndarray:
    scaled = z * params.anisotropy_scales
    power_term = np.exp(-(1.0 / z.shape[-1]) * np.sum(np.abs(scaled) ** params.power_exponents, axis=-1))
    cos_term = np.exp(
        (1.0 / z.shape[-1])
        * np.sum(
            params.cosine_amplitudes * np.cos(params.cosine_frequencies * scaled + params.cosine_phases),
            axis=-1,
        )
    )
    if params.modifier == "additive":
        return params.a * power_term + cos_term
    if params.modifier == "multiplicative":
        return power_term * cos_term
    raise ValueError(f"unknown modifier kind: {params.modifier}")


def log_prior_density(points: np.ndarray, params: BenchmarkParams) -> np.ndarray:
    z = transform_to_rotated(points, params.rotation)
    mu = rotated_mean(params)
    inv_s = 1.0 / params.prior_scales
    centered = (z - mu) * inv_s
    log_det = np.sum(np.log(params.prior_scales))
    d = points.shape[-1]
    return -0.5 * np.sum(centered ** 2, axis=-1) - log_det - 0.5 * d * np.log(2.0 * np.pi)


def unnormalized_density(points: np.ndarray, params: BenchmarkParams) -> np.ndarray:
    z = transform_to_rotated(points, params.rotation)
    return np.exp(log_prior_density(points, params)) * modifier_from_rotated(z, params)


def component_expectation(params: BenchmarkParams, component: ComponentKind) -> float:
    mu = rotated_mean(params)
    values = []
    d = params.x0.shape[0]
    for i in range(d):
        if component == "power":
            val = _power_expectation(
                mu[i], params.prior_scales[i], params.anisotropy_scales[i], params.power_exponents[i], d
            )
        else:
            val = _cos_expectation(
                mu[i],
                params.prior_scales[i],
                params.anisotropy_scales[i],
                params.cosine_amplitudes[i],
                params.cosine_frequencies[i],
                params.cosine_phases[i],
                d,
                params.bessel_terms,
            )
        values.append(val)
    return float(np.prod(values))


def normalizing_constant(params: BenchmarkParams) -> float:
    if params.modifier == "additive":
        return params.a * component_expectation(params, "power") + component_expectation(params, "cos")

    mu = rotated_mean(params)
    d = params.x0.shape[0]
    factors = []
    for i in range(d):
        factors.append(
            quad(
                lambda x: standard_normal_density(x, mu[i], params.prior_scales[i])
                * np.exp(-(abs(params.anisotropy_scales[i] * x) ** params.power_exponents[i]) / d)
                * np.exp(
                    (params.cosine_amplitudes[i] / d)
                    * np.cos(params.cosine_frequencies[i] * params.anisotropy_scales[i] * x + params.cosine_phases[i])
                ),
                -np.inf,
                np.inf,
                limit=200,
            )[0]
        )
    return float(np.prod(factors))


def exact_probe_expectation(probe: np.ndarray, kernel_scales: np.ndarray, params: BenchmarkParams) -> float:
    probe_z = (params.rotation @ probe.reshape(-1, 1)).ravel()
    mu = rotated_mean(params)
    d = params.x0.shape[0]
    z_const = normalizing_constant(params)

    if params.modifier == "additive":
        power_factors = []
        cos_factors = []
        for i in range(d):
            pref, merged_mu, merged_scale = gaussian_product_parameters(
                mu[i], params.prior_scales[i], probe_z[i], kernel_scales[i]
            )
            power_factors.append(
                pref
                * _power_expectation(
                    merged_mu,
                    merged_scale,
                    params.anisotropy_scales[i],
                    params.power_exponents[i],
                    d,
                )
            )
            cos_factors.append(
                pref
                * _cos_expectation(
                    merged_mu,
                    merged_scale,
                    params.anisotropy_scales[i],
                    params.cosine_amplitudes[i],
                    params.cosine_frequencies[i],
                    params.cosine_phases[i],
                    d,
                    params.bessel_terms,
                )
            )
        return (params.a * float(np.prod(power_factors)) + float(np.prod(cos_factors))) / z_const

    factors = []
    for i in range(d):
        pref, merged_mu, merged_scale = gaussian_product_parameters(
            mu[i], params.prior_scales[i], probe_z[i], kernel_scales[i]
        )
        factors.append(
            pref
            * quad(
                lambda x: standard_normal_density(x, merged_mu, merged_scale)
                * np.exp(-(abs(params.anisotropy_scales[i] * x) ** params.power_exponents[i]) / d)
                * np.exp(
                    (params.cosine_amplitudes[i] / d)
                    * np.cos(params.cosine_frequencies[i] * params.anisotropy_scales[i] * x + params.cosine_phases[i])
                ),
                -np.inf,
                np.inf,
                limit=200,
            )[0]
        )
    return float(np.prod(factors)) / z_const


def _component_1d(x: float, mean: float, scale: float, i: int, params: BenchmarkParams, component: ComponentKind) -> float:
    base = standard_normal_density(x, mean, scale)
    if component == "power":
        return base * np.exp(-(abs(params.anisotropy_scales[i] * x) ** params.power_exponents[i]) / params.x0.shape[0])
    if component == "cos":
        return base * np.exp(
            (params.cosine_amplitudes[i] / params.x0.shape[0])
            * np.cos(params.cosine_frequencies[i] * params.anisotropy_scales[i] * x + params.cosine_phases[i])
        )
    raise ValueError(component)


def exact_pp_term(kernel_scales: np.ndarray, params: BenchmarkParams) -> float:
    mu = rotated_mean(params)
    z_const = normalizing_constant(params)
    d = params.x0.shape[0]

    if params.modifier == "additive":
        total = 0.0
        for coeff_x, component_x in ((params.a, "power"), (1.0, "cos")):
            for coeff_y, component_y in ((params.a, "power"), (1.0, "cos")):
                factors = []
                for i in range(d):
                    factors.append(
                        quad(
                            lambda y: quad(
                                lambda x: np.exp(-0.5 * ((x - y) / kernel_scales[i]) ** 2)
                                * _component_1d(x, mu[i], params.prior_scales[i], i, params, component_x)
                                * _component_1d(y, mu[i], params.prior_scales[i], i, params, component_y),
                                -np.inf,
                                np.inf,
                                limit=120,
                            )[0],
                            -np.inf,
                            np.inf,
                            limit=120,
                        )[0]
                    )
                total += coeff_x * coeff_y * float(np.prod(factors))
        return total / (z_const * z_const)

    factors = []
    for i in range(d):
        factors.append(
            quad(
                lambda y: quad(
                    lambda x: np.exp(-0.5 * ((x - y) / kernel_scales[i]) ** 2)
                    * _component_1d(x, mu[i], params.prior_scales[i], i, params, "power")
                    * _component_1d(y, mu[i], params.prior_scales[i], i, params, "power")
                    * np.exp(
                        (params.cosine_amplitudes[i] / d)
                        * (
                            np.cos(params.cosine_frequencies[i] * params.anisotropy_scales[i] * x + params.cosine_phases[i])
                            + np.cos(params.cosine_frequencies[i] * params.anisotropy_scales[i] * y + params.cosine_phases[i])
                        )
                    ),
                    -np.inf,
                    np.inf,
                    limit=120,
                )[0],
                -np.inf,
                np.inf,
                limit=120,
            )[0]
        )
    return float(np.prod(factors)) / (z_const * z_const)


def exact_mmd_squared(samples: np.ndarray, kernel_scales: np.ndarray, params: BenchmarkParams) -> float:
    bandwidth = aligned_bandwidth(params.rotation, kernel_scales)
    empirical = gaussian_kernel_matrix(np.atleast_2d(samples), bandwidth)
    t_emp = float(np.mean(empirical))
    t_mix = float(np.mean([exact_probe_expectation(sample, kernel_scales, params) for sample in np.atleast_2d(samples)]))
    t_pp = exact_pp_term(kernel_scales, params)
    return t_pp - 2.0 * t_mix + t_emp


def make_rotation_2d(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)


def demo(exact_mmd: bool = False) -> None:
    rng = np.random.default_rng(4)
    params = BenchmarkParams(
        x0=np.zeros(2),
        prior_scales=np.ones(2),
        anisotropy_scales=np.array([3.0, 0.8]),
        power_exponents=np.array([2.0, 2.0]),
        cosine_amplitudes=np.ones(2),
        cosine_frequencies=np.full(2, np.pi),
        cosine_phases=np.zeros(2),
        rotation=make_rotation_2d(np.pi / 6.0),
        modifier="multiplicative",
    )
    samples = rng.normal(size=(24, 2))
    kernel_scales = np.array([0.8, 0.8])

    print(f"normalizing constant: {normalizing_constant(params):.6f}")
    print(f"probe expectation at x0: {exact_probe_expectation(params.x0, kernel_scales, params):.6f}")
    if exact_mmd:
        print(f"exact MMD^2: {exact_mmd_squared(samples, kernel_scales, params):.6f}")
    else:
        bandwidth = aligned_bandwidth(params.rotation, kernel_scales)
        print(f"empirical kernel mean: {np.mean(gaussian_kernel_matrix(samples, bandwidth)):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference implementation for the generalized separable Bayesian benchmark.")
    parser.add_argument(
        "--exact-mmd",
        action="store_true",
        help="also evaluate the full exact MMD^2 term; this is slower because T_PP still uses nested quadrature",
    )
    return parser.parse_args()


if __name__ == "__main__":
    demo(exact_mmd=parse_args().exact_mmd)
