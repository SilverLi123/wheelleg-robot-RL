"""Reproducible aperiodic height fields, independent of the simulator runtime."""
import hashlib

import numpy as np
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter


def field_rng(seed, difficulty, kind):
    key = f'{seed}:{float(difficulty).hex()}:{kind}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))


def _random_phase(values, rng, gap_range):
    """Smooth phase with nonuniform distances between successive full cycles."""
    low, high = float(values.min()) - 3., float(values.max()) + 3.
    knots = [low]
    while knots[-1] <= high:
        knots.append(knots[-1] + rng.uniform(*gap_range))
    knots = np.asarray(knots)
    phase = PchipInterpolator(knots, np.arange(len(knots)) * 2. * np.pi)(values)
    amplitude = PchipInterpolator(knots, rng.uniform(.4, 1., len(knots)))(values)
    return phase, amplitude, np.diff(knots)


def _noise(shape, scale, rng, spacing):
    size = np.asarray(shape) * scale
    nx, ny = np.maximum(np.ceil(size / spacing).astype(int) + 1, 4)
    x, y = np.linspace(0., size[0], nx), np.linspace(0., size[1], ny)
    # Irregularly spaced control points remove the regular interpolation lattice.
    x[1:-1] += rng.uniform(-.23, .23, nx - 2) * (x[1] - x[0])
    y[1:-1] += rng.uniform(-.23, .23, ny - 2) * (y[1] - y[0])
    z = rng.uniform(-1., 1., (nx, ny))
    result = RectBivariateSpline(x, y, z)(np.arange(shape[0]) * scale, np.arange(shape[1]) * scale)
    return result / max(float(np.abs(result).max()), 1e-8)


def irregular_wave(shape, scale, vertical_scale, difficulty, seed, return_metadata=False):
    rng = field_rng(seed, difficulty, 'aperiodic-wave-v2')
    x, y = np.meshgrid(np.arange(shape[0]) * scale, np.arange(shape[1]) * scale, indexing='ij')
    angle = rng.uniform(-np.pi, np.pi)
    u, v = x * np.cos(angle) + y * np.sin(angle), -x * np.sin(angle) + y * np.cos(angle)
    warp = _noise(shape, scale, rng, spacing=1.6)
    p1, a1, gaps1 = _random_phase(u + .22 * warp, rng, (.45, 1.65))
    p2, a2, gaps2 = _random_phase(v - .16 * warp, rng, (.65, 2.1))
    heights = .64 * a1 * np.sin(p1) + .23 * a2 * np.sin(p2)
    heights += .13 * _noise(shape, scale, rng, spacing=.65)
    # Scattered broad bumps and dips break up uninterrupted ridge lines.
    for _ in range(12):
        cx, cy = rng.uniform(0., shape[0] * scale), rng.uniform(0., shape[1] * scale)
        wx, wy = rng.uniform(.25, 1.1, 2)
        heights += rng.uniform(-.3, .3) * np.exp(-.5 * (((x - cx) / wx) ** 2 + ((y - cy) / wy) ** 2))
    heights /= max(float(np.abs(heights).max()), 1e-8)
    peak = .003 + .037 * difficulty
    heights = np.rint(np.clip(heights * peak, -peak, peak) / vertical_scale).astype(np.int16)
    metadata = dict(direction_rad=float(angle), nominal_cycle_gap_range_m=[.45, 1.65],
        sampled_cycle_gap_cv=float(np.std(gaps1) / np.mean(gaps1)),
        sampled_cycle_gaps_m=gaps1.tolist(), secondary_cycle_gaps_m=gaps2.tolist(),
        nominal_abs_height_bound_m=float(peak), version='aperiodic-wave-v2')
    return (heights, metadata) if return_metadata else heights


def irregular_rough(shape, scale, vertical_scale, difficulty, seed):
    rng = field_rng(seed, difficulty, 'multiscale-rough-v2')
    coarse = _noise(shape, scale, rng, spacing=rng.uniform(.55, 1.3))
    fine = _noise(shape, scale, rng, spacing=rng.uniform(.24, .45))
    envelope = .4 + .6 * (_noise(shape, scale, rng, spacing=2.) + 1.) / 2.
    heights = gaussian_filter(.62 * coarse + .38 * fine, sigma=.5) * envelope
    heights /= max(float(np.abs(heights).max()), 1e-8)
    peak = .003 + .022 * difficulty
    return np.rint(np.clip(heights * peak, -peak, peak) / vertical_scale).astype(np.int16)
