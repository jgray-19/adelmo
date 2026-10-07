"""Variance assignment helpers for measurement data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.measurements.noise import assign_bpm_variances

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def assign_uniform_variances(
    df: pd.DataFrame,
    bad_bpms: list[str],
    *,
    var_value: float = (1e-4) ** 2,
) -> pd.DataFrame:
    """Assign the same variance to every coordinate and zero-weight bad BPMs."""
    result = df.copy()
    for column in ("var_x", "var_y", "var_px", "var_py"):
        result[column] = var_value
        result.loc[result["name"].isin(bad_bpms), column] = float("inf")
    return result


def assign_known_noise_variances(
    df: pd.DataFrame,
    bad_bpms: list[str],
    *,
    nan_variance_patterns: str | Sequence[str] | None = None,
    accelerator_type: str = "lhc",
    noise_file: Path | None = None,
) -> pd.DataFrame:
    """Assign known BPM noise variances, allowing selected names to become NaN-weight rows.

    ``nan_variance_patterns`` is interpreted as one or more case-insensitive regex
    patterns matched against the BPM names held in the index or ``name`` column.
    Matching rows receive ``NaN`` in ``var_x`` and ``var_y`` instead of raising when
    they are absent from the packaged noise table.

    ``noise_file`` overrides the packaged table, for callers holding a resolution
    measured on the campaign being analysed rather than the shipped default.
    """
    patterns = _normalise_patterns(nan_variance_patterns)
    if not patterns:
        return assign_bpm_variances(
            df, accelerator_type=accelerator_type, bad_bpms=bad_bpms, noise_file=noise_file
        )

    indexed = _ensure_name_index(df)
    nan_mask = _build_nan_variance_mask(indexed.index.astype(str), patterns)

    included = assign_bpm_variances(
        indexed.loc[~nan_mask].copy(),
        accelerator_type=accelerator_type,
        bad_bpms=bad_bpms,
        noise_file=noise_file,
    )
    excluded = indexed.loc[nan_mask].copy()
    excluded["var_x"] = float("nan")
    excluded["var_y"] = float("nan")

    combined = pd.concat([included, excluded]).loc[indexed.index]
    if df.index.name is None and "name" in df.columns:
        return combined.reset_index()
    return combined


def _normalise_patterns(patterns: str | Sequence[str] | None) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [patterns]
    return [pattern for pattern in patterns if pattern]


def _ensure_name_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name is not None:
        return df.copy()
    if "name" not in df.columns:
        raise ValueError("DataFrame must have an index or a 'name' column for BPM names")
    return df.set_index("name").copy()


def _build_nan_variance_mask(index: pd.Index, patterns: list[str]) -> pd.Series:
    mask = pd.Series(data=False, index=index)
    for pattern in patterns:
        mask = mask | index.to_series().str.contains(pattern, case=False, regex=True)
    return mask
