"""BPM noise tables and variance helpers for simulated measurements.

This module loads packaged noise tables, derives variance maps for individual
BPMs or BPM families, and applies those variances when generating noisy
tracking-style datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NOISE_FILES_DIR = Path(__file__).with_name("noise_files")

_ACCELERATOR_NOISE_FILES = {
    "lhc": NOISE_FILES_DIR / "lhc_bpm_noise.txt",
    "psb": NOISE_FILES_DIR / "psb_bpm_noise.txt",
}


def _normalise_accelerator_type(accelerator_type: str) -> str:
    accelerator_key = accelerator_type.strip().lower()
    if accelerator_key not in _ACCELERATOR_NOISE_FILES:
        supported = ", ".join(sorted(_ACCELERATOR_NOISE_FILES))
        raise ValueError(
            f"Unsupported accelerator_type '{accelerator_type}'. Supported types: {supported}"
        )
    return accelerator_key


def get_noise_file_for_accelerator(accelerator_type: str) -> Path:
    """Return the packaged BPM noise file for a supported accelerator."""
    return _ACCELERATOR_NOISE_FILES[_normalise_accelerator_type(accelerator_type)]


def load_bpm_noise_table(
    accelerator_type: str,
    noise_file: Path | None = None,
) -> pd.DataFrame:
    """Load a packaged BPM noise table, converting std values from mm to m."""
    accelerator_key = _normalise_accelerator_type(accelerator_type)
    noise_data = pd.read_csv(
        noise_file or get_noise_file_for_accelerator(accelerator_key),
        sep=r"\s+",
        header=0,
        names=["name", "Horizontal_STD", "Vertical_STD"],
    )
    noise_data["name"] = noise_data["name"].str.upper()
    noise_data["Horizontal_STD"] /= 1000.0
    noise_data["Vertical_STD"] /= 1000.0
    return noise_data


def build_bpm_variance_maps(
    accelerator_type: str,
    noise_file: Path | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build per-BPM variance maps for a supported accelerator."""
    accelerator_key = _normalise_accelerator_type(accelerator_type)
    noise_data = load_bpm_noise_table(accelerator_key, noise_file)

    noise_std_x = noise_data.set_index("name")["Horizontal_STD"].to_dict()
    noise_std_y = noise_data.set_index("name")["Vertical_STD"].to_dict()

    def _variance_for_std(std: float) -> float:
        return float("inf") if std == 0 else float(std**2)

    var_x = {bpm_name: _variance_for_std(std) for bpm_name, std in noise_std_x.items()}
    var_y = {bpm_name: _variance_for_std(std) for bpm_name, std in noise_std_y.items()}
    return var_x, var_y


def resolve_bpm_variance(
    bpm_name: str,
    per_bpm: dict[str, float],
) -> float:
    """Resolve the variance for one BPM using exact-name lookup only."""
    bpm_name = bpm_name.upper()
    if bpm_name in per_bpm:
        return per_bpm[bpm_name]

    raise ValueError(f"No noise variance found for BPM {bpm_name}")


def assign_bpm_variances(
    df: pd.DataFrame,
    accelerator_type: str,
    bad_bpms: list[str] | None = None,
    noise_file: Path | None = None,
) -> pd.DataFrame:
    """Assign `var_x` and `var_y` using the packaged BPM noise table for an accelerator."""
    accelerator_key = _normalise_accelerator_type(accelerator_type)
    var_x_by_bpm, var_y_by_bpm = build_bpm_variance_maps(accelerator_key, noise_file)
    df = df.copy()
    reset_index = False
    if df.index.name is None:
        reset_index = True
        if "name" in df.columns:
            df.set_index("name", inplace=True)
        else:
            raise ValueError("DataFrame must have an index or a 'name' column for BPM names")
    df.index = df.index.astype(str).str.upper()
    if accelerator_key == "lhc":
        # remove rows BPMCS.
        df = df[~df.index.str.startswith("BPMCS.")]
    df.index.name = "name"
    df["var_x"] = df.index.map(lambda bpm: resolve_bpm_variance(bpm, var_x_by_bpm))
    df["var_y"] = df.index.map(lambda bpm: resolve_bpm_variance(bpm, var_y_by_bpm))

    if bad_bpms:
        bad_bpm_set = {bpm.upper() for bpm in bad_bpms}
        df.loc[df.index.isin(bad_bpm_set), "var_x"] = float("inf")
        df.loc[df.index.isin(bad_bpm_set), "var_y"] = float("inf")

    if reset_index:
        df = df.reset_index()

    return df


def apply_bpm_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    accelerator_type: str,
    bad_bpms: list[str] | None = None,
    noise_file: Path | None = None,
) -> pd.DataFrame:
    """Apply BPM-dependent Gaussian noise to `x` and `y` and assign matching variances."""
    df = assign_bpm_variances(
        df,
        accelerator_type=accelerator_type,
        bad_bpms=bad_bpms,
        noise_file=noise_file,
    )
    std_x = np.sqrt(df["var_x"].to_numpy(dtype=float))
    std_y = np.sqrt(df["var_y"].to_numpy(dtype=float))
    noise_x = np.zeros_like(std_x)
    noise_y = np.zeros_like(std_y)
    finite_x = np.isfinite(std_x)
    finite_y = np.isfinite(std_y)
    noise_x[finite_x] = rng.normal(0.0, std_x[finite_x])
    noise_y[finite_y] = rng.normal(0.0, std_y[finite_y])
    df["x"] += noise_x
    df["y"] += noise_y
    return df
