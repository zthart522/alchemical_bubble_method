#!/usr/bin/env python3
"""
analyze_bubble.py

Parse LAMMPS thermo output from a lambda-windowed TI (thermodynamic
integration) run, laid out as:

    bubble_on/lambda_0.000/log.lammps
    bubble_on/lambda_0.250/log.lammps
    ...
    bubble_off/lambda_*/log.lammps
    ti_cross/lambda_*/log.lammps

For each lambda directory in each leg, this script:
  1. Extracts the thermo table and the f_dUdL_avg column.
  2. Computes its mean (and std) over the run.

Per leg, it then builds two grid figures (one PNG each) with one subplot
per lambda window, arranged in a grid with GRID_NCOLS columns:
  3. A grid of histograms of f_dUdL_avg, one panel per window.
  4. A grid of timeseries of f_dUdL_avg (vs. Step), one panel per window.

Then, per leg, it:
  5. Plots mean f_dUdL_avg vs. lambda.
  6. Integrates that curve over lambda using Simpson's rule and the
     trapezoidal rule, reporting free-energy contributions.

Usage:
    python analyze_bubble.py --root . --legs bubble_on bubble_off ti_cross
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import simpson, trapezoid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THERMO_COL = "f_dUdL_avg"   # column name to extract from thermo output
LOG_NAME = "log.lammps"     # LAMMPS stdout/log filename inside each lambda dir
LAMBDA_DIR_RE = re.compile(r"lambda_([0-9]*\.?[0-9]+)")
GRID_NCOLS = 3               # number of columns in the per-leg subplot grids


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_lammps_thermo(log_path: Path, column: str = THERMO_COL) -> "np.ndarray | None":
    """
    Extract a named thermo column from a LAMMPS log file as a 1D array.

    LAMMPS prints one or more thermo blocks per log (one per `run`
    command), each starting with a header line of column names and
    ending at a line starting with "Loop time". This grabs every
    thermo block found and concatenates the requested column across
    all of them (steps are not de-duplicated/re-indexed; if you only
    want the last run block, filter on Step resets instead).

    Returns None if the column is never found.
    """
    text = log_path.read_text(errors="ignore").splitlines()

    values = []
    steps = []
    header = None
    col_idx = None

    for line in text:
        stripped = line.strip()

        # Detect a thermo header line (contains "Step" and our column name)
        if stripped.startswith("Step") and column in stripped:
            header = stripped.split()
            col_idx = header.index(column)
            step_idx = header.index("Step")
            continue

        if header is not None:
            # End of this thermo block
            if stripped.startswith("Loop time") or stripped == "":
                header = None
                col_idx = None
                continue

            parts = stripped.split()
            if len(parts) != len(header):
                # Not a data row (e.g. a warning line interleaved in log)
                continue
            try:
                step_val = float(parts[step_idx])
                col_val = float(parts[col_idx])
            except ValueError:
                continue
            steps.append(step_val)
            values.append(col_val)

    if not values:
        return None

    return np.array(steps), np.array(values)


def discover_lambda_dirs(leg_root: Path):
    """Return sorted list of (lambda_value, directory_path) under a leg root."""
    found = []
    if not leg_root.is_dir():
        return found
    for d in leg_root.iterdir():
        if not d.is_dir():
            continue
        m = LAMBDA_DIR_RE.search(d.name)
        if m:
            found.append((float(m.group(1)), d))
    found.sort(key=lambda t: t[0])
    return found


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_grid(n_panels: int, ncols: int, figsize_per_panel=(4.2, 3.4)):
    """Create a grid of axes sized to fit n_panels, hiding unused cells."""
    ncols = min(ncols, max(n_panels, 1))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    return fig, axes_flat


def plot_histogram_grid(windows: list, leg: str, outdir: Path):
    """windows: list of (lam, steps, values) tuples, one per lambda window."""
    fig, axes = _make_grid(len(windows), GRID_NCOLS)

    for ax, (lam, steps, values) in zip(axes, windows):
        ax.hist(values, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(values.mean(), color="firebrick", linestyle="--",
                   label=f"mean = {values.mean():.4f}")
        ax.set_xlabel(THERMO_COL)
        ax.set_ylabel("Count")
        ax.set_title(f"\u03bb = {lam:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle(f"{leg}: {THERMO_COL} histograms by window", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / f"hist_grid_{leg}.png", dpi=150)
    plt.close(fig)


def plot_timeseries_grid(windows: list, leg: str, outdir: Path):
    """windows: list of (lam, steps, values) tuples, one per lambda window."""
    fig, axes = _make_grid(len(windows), GRID_NCOLS)

    for ax, (lam, steps, values) in zip(axes, windows):
        ax.plot(steps, values, color="darkorange", linewidth=1)
        ax.axhline(values.mean(), color="firebrick", linestyle="--",
                   label=f"mean = {values.mean():.4f}")
        ax.set_xlabel("Step")
        ax.set_ylabel(THERMO_COL)
        ax.set_title(f"\u03bb = {lam:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle(f"{leg}: {THERMO_COL} timeseries by window", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / f"timeseries_grid_{leg}.png", dpi=150)
    plt.close(fig)


def plot_leg_curve(lams: np.ndarray, means: np.ndarray, sems: np.ndarray, leg: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(lams, means, yerr=sems, marker="o", capsize=3, color="teal")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle dU/d\lambda \rangle$")
    ax.set_title(f"{leg}: dU/d\u03bb vs \u03bb")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"dUdL_vs_lambda_{leg}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def analyze_leg(leg: str, root: Path, outdir: Path):
    leg_root = root / leg
    lambda_dirs = discover_lambda_dirs(leg_root)

    if not lambda_dirs:
        print(f"[{leg}] no lambda_* directories found under {leg_root}, skipping.")
        return None

    lams, means, stds, sems = [], [], [], []
    windows = []  # (lam, steps, values) for windows successfully parsed

    for lam, d in lambda_dirs:
        log_path = d / LOG_NAME
        if not log_path.exists():
            print(f"[{leg}] missing {log_path}, skipping this window.")
            continue

        result = parse_lammps_thermo(log_path)
        if result is None:
            print(f"[{leg}] column '{THERMO_COL}' not found in {log_path}, skipping.")
            continue
        
        # Discard the first 30% as equilibration
        steps, values = result
        N = int(0.3 * len(values))
        steps = steps[N:]
        values = values[N:]

        windows.append((lam, steps, values))

        mean, std, sem, n_eff = estimate_mean_error(values)

        lams.append(lam)
        means.append(mean)
        stds.append(std)
        sems.append(sem)

        print(
            f"[{leg}] lambda={lam:.3f}  "
            f"mean={mean:.6f}  "
            f"std={std:.6f}  "
            f"SEM={sem:.6f}  "
            f"n={len(values)}  "
            f"n_eff={n_eff:.1f}"
        )


    if not lams:
        return None

    # One grid figure per leg for histograms, and one for timeseries,
    # instead of a separate PNG per lambda window.
    plot_histogram_grid(windows, leg, outdir)
    plot_timeseries_grid(windows, leg, outdir)

    lams = np.array(lams)
    means = np.array(means)
    stds = np.array(stds)
    sems = np.array(sems)

    # Sort by lambda (discover_lambda_dirs already sorts, but be safe)
    order = np.argsort(lams)
    lams = lams[order]
    means = means[order]
    stds = stds[order]
    sems = sems[order]


    plot_leg_curve(lams, means, sems, leg, outdir)

    simpson_val = simpson(means, x=lams)
    trapz_val = trapezoid(means, x=lams)

    trapz_val, trapz_err = compute_integral(lams, means, sems, method='trapz')
    simpson_val, simpson_err = compute_integral(lams, means, sems, method='simpson')
    

    print(
        f"[{leg}] Trapezoid integral = "
        f"{trapz_val:.6f} ± {trapz_err:.6f}"
    )

    print(
        f"[{leg}] Simspson integral = "
        f"{simpson_val:.6f} ± {simpson_err:.6f}"
    )

    return {
        "leg": leg,
        "lams": lams,
        "means": means,
        "stds": stds,
        "sems": sems,
        "simpson": simpson_val,
        "trapz": trapz_val,
        "trapz_err": trapz_err,
        "simpson_err": simpson_err

    }


def estimate_mean_error(values: np.ndarray, max_lag: int = 1000):
    """
    Estimate the mean, standard deviation, and standard error of the mean
    for a correlated time series.

    The standard error accounts approximately for autocorrelation using
    the integrated autocorrelation time:

        var(mean) = var(x) * (1 + 2 * sum rho_k) / N

    The summation stops when the autocorrelation becomes non-positive or
    max_lag is reached.

    Returns
    -------
    mean : float
        Sample mean.
    std : float
        Sample standard deviation.
    sem : float
        Estimated standard error of the mean.
    n_eff : float
        Effective number of independent samples.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n < 2:
        return np.mean(values), np.nan, np.nan, 1.0

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    # Remove mean
    x = values - mean

    # Autocorrelation using FFT
    fft = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(fft * np.conjugate(fft))[:n]
    acov /= np.arange(n, 0, -1)

    if acov[0] <= 0:
        return mean, std, std / np.sqrt(n), n

    rho = acov / acov[0]

    # Integrated autocorrelation time.
    # Stop at the first non-positive autocorrelation.
    positive = rho[1:min(max_lag + 1, n)] > 0

    tau_int = 0.5  # convention: tau_int = 1/2 + sum rho_k

    for r, keep in zip(rho[1:], positive):
        if not keep:
            break
        tau_int += r

    # Effective sample size
    n_eff = n / (2.0 * tau_int)

    # Guard against pathological numerical values
    n_eff = max(1.0, min(float(n), n_eff))

    sem = std / np.sqrt(n_eff)

    return mean, std, sem, n_eff

def trapezoidal_weights(x):
    """
    Compute global weights for trapezoidal rule integration estimate,
    such that: I = sum_i w_i * y_i

    :param x: a list of discrete x-values where a corresponding
        y-value is measured
    :type x: array of floats
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = np.zeros(n)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    for i in range(1, n - 1):
        w[i] = 0.5 * (x[i + 1] - x[i - 1])
    return w

def simpson_weights(x):
    """
    Compute global weights for composite Simpson's 1/3 rule
    on a potentially nonuniform grid.

    If an even number of points are passed, Simpson's rule is
    applied through the second-to-last point and the final interval
    is handled with the trapezoidal rule.

    :param x: a list of discrete x-values where a corresponding
        y-value is measured
    :type x: array of floats

    :returns: weights as an ndarray the same length as x
    :rtype: ndarray
    """
    n = len(x)
    if n < 3:
        raise ValueError("Simpson's rule requires at least 3 points.")

    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")

    weights = np.zeros(n)
    h = np.diff(x)
    simpson_n = n if n % 2 == 1 else n - 1

    # Apply Simpson's 1/3 rule to consecutive triplets.
    for j in range(1, simpson_n - 1, 2):
        h1 = h[j - 1]
        h2 = h[j]
        weights[j - 1] += (2 * h1**2 + h1 * h2 - h2**2) / (6 * h1)
        weights[j] += (h1 + h2)**3 / (6 * h1 * h2)
        weights[j + 1] += (2 * h2**2 + h1 * h2 - h1**2) / (6 * h2)

    # If n is even, use trapz for last interval
    if n % 2 == 0:
        h_last = x[n - 1] - x[n - 2]
        weights[n - 2] += 0.5 * h_last
        weights[n - 1] += 0.5 * h_last

    return weights

def compute_integral(x, y, y_err, method='simpson'):
    """
    Compute an integral using either trapezoidal rule or Simpson's
    1/3 rule.

    :param x: a list of discrete x-values where a corresponding
        y-value is measured
    :type x: array of floats
    :param y: a list of y-values, each sampled as the corresponding
        x-value
    :type y: array of floats
    :param y_err: the sampling uncertainty of the corresponding y-value
    :type y_err: array of floats
    :param method: integration method to use, either "simpson" for
        composite Simpson's 1/3 rule or "trapz" for the trapezoidal rule.
    :type method: str, optional

    :returns: A tuple ``(I, error)`` containing the estimated integral
        value and its propagated uncertainty.
    :rtype: tuple
    """
    if method.lower() == 'simpson':
        w = simpson_weights(x)
    elif method.lower() == 'trapz':
        w = trapezoidal_weights(x)
    else:
        raise ValueError(f"Unknown method {method}")
    integral = np.dot(w, y)
    error = np.sqrt(np.sum((w * y_err)**2))
    return integral, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                         help="Root directory containing the leg subdirectories.")
    parser.add_argument("--legs", nargs="+",
                         default=["bubble_on", "bubble_off", "ti_cross"],
                         help="Leg subdirectory names to process.")
    parser.add_argument("--outdir", type=Path, default=Path("plots"),
                         help="Directory to write plots/summary into.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    results = {}

    for leg in args.legs:
        res = analyze_leg(leg, args.root, args.outdir)
        if res is not None:
            results[leg] = res

    # Combined overview plot (all legs on one axis)
    if results:
        fig, ax = plt.subplots(figsize=(7, 5))
        for leg, res in results.items():
            ax.errorbar(
                res["lams"],
                res["means"],
                yerr=res["sems"],
                marker="o",
                capsize=3,
                label=leg
            )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\langle dU/d\lambda \rangle$")
        ax.set_title("dU/d\u03bb vs \u03bb, all legs")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.outdir / "dUdL_vs_lambda_all_legs.png", dpi=150)
        plt.close(fig)

    # Write a plain-text summary
    summary_path = args.outdir / "integration_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"{'Leg':<15}{'Simpson':>15}{'Simpson Error':>15}{'Trapz':>15}{'Trapz Error':>15}\n")
        for leg, res in results.items():
            f.write(f"{leg:<15}{res['simpson']:>15.6f}{res['simpson_err']:>15.6f}{res['trapz']:>15.6f}{res['trapz_err']:>15.6f}\n")
        if results:
            total_simpson = -(
                results["bubble_on"]["simpson"]
                - results["bubble_off"]["simpson"]
                + results["ti_cross"]["simpson"]
            )
            total_simpson_err = np.sqrt(
                results["bubble_on"]["simpson_err"]**2
                + results["bubble_off"]["simpson_err"]**2
                + results["ti_cross"]["simpson_err"]**2
            )

            total_trapz = -(
                results["bubble_on"]["trapz"]
                - results["bubble_off"]["trapz"]
                + results["ti_cross"]["trapz"]
            )
            total_trapz_err = np.sqrt(
                results["bubble_on"]["trapz_err"]**2
                + results["bubble_off"]["trapz_err"]**2
                + results["ti_cross"]["trapz_err"]**2
            )

            f.write(f"{'NET':<15}{total_simpson:>15.6f}{total_simpson_err:>15.6f}{total_trapz:>15.6f}{total_trapz_err:>15.6f}\n")

    print(f"\nSummary written to {summary_path}")
    print(f"Plots written to {args.outdir}/")


if __name__ == "__main__":
    main()