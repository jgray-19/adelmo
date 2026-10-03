"""Per-dataframe momentum reconstruction helpers."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd
from pymadng_utils.physics import beta_from_energy, dp2pt
from tmom_recon import ACDipoleConfig, ModelDetails, OpticsInput, calculate_pz
from tmom_recon.svd import weighted_svd_clean_measurements

from aba_optimiser.measurements.preprocessing import trim_measurement_to_kick
from aba_optimiser.measurements.variances import (
    assign_known_noise_variances,
    assign_uniform_variances,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from aba_optimiser.measurements.reference import ReconstructionFrame

LOGGER = logging.getLogger(__name__)


def process_single_dataframe(
    df_with_index: tuple[int, pd.DataFrame],
    twiss: pd.DataFrame,
    bad_bpms: list[str],
    analysis_dir: Path,
    use_uniform_vars: bool,
    beam: int,
    model_details: ModelDetails,
    frame: ReconstructionFrame,
    ac_dipole_inputs_factory: Callable[[int], tuple[ModelDetails, ACDipoleConfig, float] | None]
    | None = None,
    machine_deltap: float | None = None,
    trim_to_kick: bool = False,
    n_turns_free: int = 1000,
    kicker_name: str | None = None,
    nan_variance_patterns: str | list[str] | None = None,
    accelerator_type: str = "lhc",
) -> tuple[int, pd.DataFrame]:
    """Preprocess, weight, and reconstruct one measurement dataframe.

    ``frame`` owns the measured orbit-zero transformation. Raw BPM positions are
    passed to tmom-recon; this wrapper never subtracts an orbit itself.
    """
    index, df = df_with_index
    # Each input dataframe is a single bunch; preserve its identifier across the
    # reconstruction transforms (some of which rebuild the frame and drop columns).
    bunch_number = int(df["bunch_number"].iloc[0])
    # Check every row has the same bunch number, otherwise the input dataframe is malformed.
    if not (df["bunch_number"] == bunch_number).all():
        raise ValueError(f"Input dataframe {index} contains multiple bunch numbers")
    ac_dipole_inputs = (
        ac_dipole_inputs_factory(index) if ac_dipole_inputs_factory is not None else None
    )
    call_model_details, ac_dipole_config, barrier_s = (
        ac_dipole_inputs if ac_dipole_inputs is not None else (model_details, None, None)
    )

    if trim_to_kick:
        df = trim_measurement_to_kick(
            df,
            twiss,
            n_turns_free=n_turns_free,
            kicker_name=kicker_name,
        )
    df = df[df["name"].isin(twiss.index)]
    df = _assign_variances(
        df,
        bad_bpms,
        use_uniform_vars=use_uniform_vars,
        nan_variance_patterns=nan_variance_patterns,
        accelerator_type=accelerator_type,
    )
    cleaned = weighted_svd_clean_measurements(df)
    svd_ranks = (cleaned.attrs.get("svd_rank_x"), cleaned.attrs.get("svd_rank_y"))
    n_bpms = int(cleaned["name"].nunique())
    # Scale before calculate_pz: it builds var_px/var_py from var_x/var_y, so the
    # momentum weights must see the cleaned-data position variance too.
    df = _scale_position_variances_after_svd(cleaned, n_bpms=n_bpms, svd_ranks=svd_ranks)

    # tmom-recon takes the measurement pt on the ModelDetails and the measured
    # optics through OpticsInput. Phase, beta and alpha come from the omc3
    # analysis directory; dispersion stays on the model because a single kick
    # measures no dispersion, and OpticsInput refuses to substitute silently.
    if machine_deltap is not None:
        call_model_details = replace(
            call_model_details, pt=_machine_deltap_to_pt(machine_deltap, twiss)
        )
    df = calculate_pz(
        df,
        call_model_details,
        closed_orbit_at_zero=frame.closed_orbit_at_zero,
        orbit_mode=frame.orbit_mode,
        optics=OpticsInput(
            measurement_dir=analysis_dir,
            sources=dict.fromkeys(("phase", "beta", "alpha"), "measurement"),
            reverse_measurement_order=beam == 2,
        ),
        acd=ac_dipole_config,
        barrier_s=barrier_s,
    )
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Reconstruction returned unexpected type {type(df)} for dataframe")
    df = _drop_nan_momenta(df, dataframe_index=index)
    if ac_dipole_config is not None:
        df.attrs["ac_dipole_barrier_s"] = barrier_s
    df["bunch_number"] = bunch_number
    return index, df


def _machine_deltap_to_pt(machine_deltap: float | None, twiss: pd.DataFrame) -> float:
    """Convert machine ``dp/p`` metadata to MAD-NG ``pt`` for dispersion-aware reconstruction."""
    if machine_deltap is None:
        return 0.0
    headers = {str(key).lower(): value for key, value in getattr(twiss, "headers", {}).items()}
    beta0 = headers.get("beta")
    if beta0 is not None:
        return dp2pt(machine_deltap, float(beta0))
    energy = headers.get("energy")
    if energy is not None:
        beta0 = beta_from_energy(float(energy), particle="proton")
        return dp2pt(machine_deltap, beta0)
    LOGGER.warning(
        "Twiss headers do not contain beam beta or energy; converting machine_deltap to pt with beta=1."
    )
    return dp2pt(machine_deltap)


def _assign_variances(
    df: pd.DataFrame,
    bad_bpms: list[str],
    *,
    use_uniform_vars: bool,
    nan_variance_patterns: str | list[str] | None,
    accelerator_type: str,
) -> pd.DataFrame:
    if use_uniform_vars:
        return assign_uniform_variances(df, bad_bpms)

    noise_input = df.set_index("name")
    weighted = assign_known_noise_variances(
        noise_input,
        bad_bpms,
        nan_variance_patterns=nan_variance_patterns,
        accelerator_type=accelerator_type,
    )
    return weighted.reset_index()


def _scale_position_variances_after_svd(
    df: pd.DataFrame,
    *,
    n_bpms: int,
    svd_ranks: tuple[int | None, int | None],
) -> pd.DataFrame:
    """Reduce the declared position variances by the gain the SVD actually achieved.

    Projecting an ``n_bpms``-dimensional noise vector onto ``rank`` retained modes
    leaves ``rank / n_bpms`` of its variance, so the cleaned data is more precise
    than the resolution the BPM table declares and its weight has to rise to match.

    This used to be a hardcoded factor of 100, which is the LHC value: ~500 BPMs
    over an auto-selected rank of ~5. On the PSB's 16 BPMs at rank 2 the true gain
    is 8, so the constant over-weighted PSB positions by more than an order of
    magnitude. Verified against simulation at 16/32/128 BPMs (measured 7.97, 16.41
    and 60.85 against 8, 16 and 64) and against PSB tracking data, where the
    measured gain is 9.2.
    """
    result = df.copy()
    for plane, rank in zip(("x", "y"), svd_ranks, strict=True):
        if not rank or rank <= 0 or n_bpms <= 0:
            LOGGER.warning(
                "No SVD rank recorded for plane %s; leaving var_%s unscaled.", plane, plane
            )
            continue
        result[f"var_{plane}"] = result[f"var_{plane}"] * (rank / n_bpms)
    return result


def _drop_nan_momenta(df: pd.DataFrame, *, dataframe_index: int) -> pd.DataFrame:
    if df["px"].isna().any() or df["py"].isna().any():
        LOGGER.warning(
            "NaN values found in px or py for dataframe %s, dropping rows.", dataframe_index
        )
        return df.dropna(subset=["px", "py"])
    return df


def append_acd_marker_rows(frame: pd.DataFrame, acd_result: pd.DataFrame) -> pd.DataFrame:
    """Append tmom-recon's ``<acd>_before`` / ``<acd>_after`` marker rows to ``frame``.

    The ACD fits start bidirectional tracking at these markers, so the
    measurement parquet must carry their reconstructed phase space per turn.
    Without them the worker setup drops the marker as a missing BPM and no
    observation plan survives. Row order does not matter: the data loader
    reorders every file into ring order. Marker rows are reindexed to
    ``frame``'s columns; ``frame`` is returned unchanged if there are none.
    """
    is_marker = acd_result["name"].astype(str).str.lower().str.endswith(("_before", "_after"))
    marker_rows = acd_result.loc[is_marker]
    if marker_rows.empty:
        LOGGER.warning("ACD reconstruction carried no _before/_after marker rows to append.")
        return frame
    LOGGER.info(
        "Appending %d AC-dipole marker row(s) (%s).",
        len(marker_rows),
        ", ".join(sorted(marker_rows["name"].astype(str).unique())),
    )
    return pd.concat([frame, marker_rows.reindex(columns=frame.columns)], ignore_index=True)
