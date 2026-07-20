from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

from src.repressilator import RepressilatorParameters, simulate_deterministic, simulate_gillespie, oscillation_metrics, parameter_sweep
from src.gene_expression import simulate_ensemble

FIG = Path("figures"); RES = Path("results")
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

params = RepressilatorParameters()
times = np.linspace(0, 200, 2001)
initial_det = [1, 0, 2, 0, 0, 0]
initial_stoch = [1, 1, 1, 0, 0, 0]

det = simulate_deterministic(times, initial_det, params)
st_t, st_x = simulate_gillespie(200, initial_stoch, params, seed=42)

det_period, det_amp = oscillation_metrics(times, det[:, 3])
st_period, st_amp = oscillation_metrics(st_t, st_x[:, 3])

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
for i, label in enumerate(["Protein 1", "Protein 2", "Protein 3"]):
    axes[0].plot(times, det[:, 3+i], label=label)
    axes[1].step(st_t, st_x[:, 3+i], where="post", label=label)
axes[0].set(title="Deterministic repressilator", ylabel="Concentration")
axes[1].set(title="Stochastic repressilator", xlabel="Time", ylabel="Molecule count")
for ax in axes: ax.legend()
fig.tight_layout(); fig.savefig(FIG/"deterministic_vs_stochastic.png", dpi=300); plt.close(fig)

alphas = np.linspace(20, 100, 6)
thresholds = np.linspace(10, 100, 8)
periods, amplitudes = parameter_sweep(alphas, thresholds, times, initial_det, params)
for values, title, fname in [(periods, "Oscillation period", "period_heatmap.png"), (amplitudes, "Oscillation amplitude", "amplitude_heatmap.png")]:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(values, origin="lower", aspect="auto", extent=[thresholds[0], thresholds[-1], alphas[0], alphas[-1]])
    fig.colorbar(image, ax=ax, label=title)
    ax.set(xlabel="Repression threshold K", ylabel="Transcription rate alpha", title=title)
    fig.tight_layout(); fig.savefig(FIG/fname, dpi=300); plt.close(fig)

pd.DataFrame(periods, index=alphas, columns=thresholds).to_csv(RES/"deterministic_period_sweep.csv")
pd.DataFrame(amplitudes, index=alphas, columns=thresholds).to_csv(RES/"deterministic_amplitude_sweep.csv")

simple_times = np.linspace(0, 50, 101)
samples = simulate_ensemble(100, simple_times, params=(10.0, 1.0, 10.0, 0.4), seed=42)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for species, label in enumerate(["mRNA", "Protein"]):
    for trajectory in samples[:, :, species]:
        axes[species].plot(simple_times, trajectory, linewidth=0.3, alpha=0.15)
    axes[species].plot(simple_times, samples[:, :, species].mean(axis=0), linewidth=3, label="Mean")
    axes[species].set(xlabel="Time", ylabel="Copy number", title=label); axes[species].legend()
fig.tight_layout(); fig.savefig(FIG/"gene_expression_ensemble.png", dpi=300); plt.close(fig)

summary = {
    "deterministic_period": det_period,
    "deterministic_amplitude": det_amp,
    "stochastic_period": st_period,
    "stochastic_amplitude": st_amp,
    "final_mean_mrna": float(samples[:, -1, 0].mean()),
    "final_mean_protein": float(samples[:, -1, 1].mean()),
}
(RES/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
