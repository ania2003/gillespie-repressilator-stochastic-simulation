"""Gillespie SSA for a two-species mRNA-protein birth-death model."""
from __future__ import annotations
import numpy as np

UPDATE = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)


def propensities(population, transcription_rate, mrna_degradation, translation_rate, protein_degradation):
    mrna, protein = population
    return np.array([
        transcription_rate,
        mrna_degradation * mrna,
        translation_rate * mrna,
        protein_degradation * protein,
    ], dtype=float)


def simulate_trajectory(time_points, initial_population, params, rng):
    population = np.asarray(initial_population, dtype=int).copy()
    output = np.empty((len(time_points), 2), dtype=int)
    output[0] = population
    current_time = float(time_points[0])
    output_index = 1
    while output_index < len(time_points):
        rates = propensities(population, *params)
        total = rates.sum()
        if total <= 0:
            output[output_index:] = population
            break
        next_time = current_time + rng.exponential(1.0 / total)
        while output_index < len(time_points) and time_points[output_index] <= next_time:
            output[output_index] = population
            output_index += 1
        reaction = rng.choice(4, p=rates / total)
        candidate = population + UPDATE[reaction]
        if np.all(candidate >= 0):
            population = candidate
        current_time = next_time
    return output


def simulate_ensemble(size, time_points, initial_population=(0, 0), params=(10.0, 1.0, 10.0, 0.4), seed=42):
    rng = np.random.default_rng(seed)
    samples = np.empty((size, len(time_points), 2), dtype=int)
    for i in range(size):
        samples[i] = simulate_trajectory(time_points, initial_population, params, rng)
    return samples
