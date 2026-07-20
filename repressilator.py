"""Deterministic and stochastic repressilator models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


@dataclass(frozen=True)
class RepressilatorParameters:
    transcription_rate: float = 50.0
    basal_transcription: float = 0.1
    translation_rate: float = 5.0
    mrna_degradation: float = 0.1
    protein_degradation: float = 0.5
    repression_threshold: float = 40.0
    hill_coefficient: float = 2.0


def hill_repression(protein: float, threshold: float, hill: float) -> float:
    return 1.0 / (1.0 + (protein / threshold) ** hill)


def repressilator_rhs(_t: float, state: np.ndarray, p: RepressilatorParameters) -> np.ndarray:
    m1, m2, m3, p1, p2, p3 = state
    transcription = p.transcription_rate
    basal = p.basal_transcription
    dm1 = transcription * hill_repression(p3, p.repression_threshold, p.hill_coefficient) + basal - p.mrna_degradation * m1
    dm2 = transcription * hill_repression(p1, p.repression_threshold, p.hill_coefficient) + basal - p.mrna_degradation * m2
    dm3 = transcription * hill_repression(p2, p.repression_threshold, p.hill_coefficient) + basal - p.mrna_degradation * m3
    dp1 = p.translation_rate * m1 - p.protein_degradation * p1
    dp2 = p.translation_rate * m2 - p.protein_degradation * p2
    dp3 = p.translation_rate * m3 - p.protein_degradation * p3
    return np.array([dm1, dm2, dm3, dp1, dp2, dp3], dtype=float)


def simulate_deterministic(times: np.ndarray, initial_state: Iterable[float], p: RepressilatorParameters) -> np.ndarray:
    solution = solve_ivp(
        fun=lambda t, y: repressilator_rhs(t, y, p),
        t_span=(float(times[0]), float(times[-1])),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=times,
        method="LSODA",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def stochastic_propensities(state: np.ndarray, p: RepressilatorParameters) -> np.ndarray:
    m1, m2, m3, p1, p2, p3 = state
    return np.array([
        p.transcription_rate * hill_repression(p3, p.repression_threshold, p.hill_coefficient) + p.basal_transcription,
        p.transcription_rate * hill_repression(p1, p.repression_threshold, p.hill_coefficient) + p.basal_transcription,
        p.transcription_rate * hill_repression(p2, p.repression_threshold, p.hill_coefficient) + p.basal_transcription,
        p.mrna_degradation * m1,
        p.mrna_degradation * m2,
        p.mrna_degradation * m3,
        p.translation_rate * m1,
        p.translation_rate * m2,
        p.translation_rate * m3,
        p.protein_degradation * p1,
        p.protein_degradation * p2,
        p.protein_degradation * p3,
    ], dtype=float)


STOICHIOMETRY = np.array([
    [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
    [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, -1],
], dtype=int)


def simulate_gillespie(
    max_time: float,
    initial_state: Iterable[int],
    p: RepressilatorParameters,
    seed: int | None = None,
    max_events: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = np.asarray(initial_state, dtype=int).copy()
    if state.shape != (6,) or np.any(state < 0):
        raise ValueError("initial_state must contain six non-negative integers")
    time = 0.0
    times = [time]
    states = [state.copy()]
    for _ in range(max_events):
        rates = stochastic_propensities(state, p)
        total = rates.sum()
        if total <= 0 or time >= max_time:
            break
        time += rng.exponential(1.0 / total)
        if time > max_time:
            break
        reaction = rng.choice(len(rates), p=rates / total)
        candidate = state + STOICHIOMETRY[reaction]
        if np.any(candidate < 0):
            continue
        state = candidate
        times.append(time)
        states.append(state.copy())
    return np.asarray(times), np.asarray(states)


def oscillation_metrics(times: np.ndarray, signal: np.ndarray, prominence: float | None = None) -> tuple[float, float]:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 3:
        return np.nan, np.nan
    if prominence is None:
        prominence = max(1e-12, 0.1 * np.ptp(signal))
    peaks, _ = find_peaks(signal, prominence=prominence)
    period = float(np.mean(np.diff(times[peaks]))) if len(peaks) >= 2 else np.nan
    amplitude = float(np.mean(signal[peaks]) - np.min(signal)) if len(peaks) else np.nan
    return period, amplitude


def parameter_sweep(alpha_values, threshold_values, times, initial_state, base_params):
    periods = np.full((len(alpha_values), len(threshold_values)), np.nan)
    amplitudes = np.full_like(periods, np.nan)
    for i, alpha in enumerate(alpha_values):
        for j, threshold in enumerate(threshold_values):
            p = RepressilatorParameters(**{**base_params.__dict__, "transcription_rate": float(alpha), "repression_threshold": float(threshold)})
            solution = simulate_deterministic(times, initial_state, p)
            periods[i, j], amplitudes[i, j] = oscillation_metrics(times, solution[:, 3])
    return periods, amplitudes
