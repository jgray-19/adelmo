"""Measurement preprocessing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tmom_recon.kicker.core import find_kick

if TYPE_CHECKING:
    import pandas as pd


def trim_measurement_to_kick(
    df: pd.DataFrame,
    twiss: pd.DataFrame,
    *,
    n_turns_free: int = 1000,
    kicker_name: str | None = None,
) -> pd.DataFrame:
    """Trim raw BPM data to the kick without changing its coordinate frame."""
    if _starts_at_kicker(df, kicker_name):
        return df.copy()
    indexed = df.set_index("name", drop=False)
    kick_bpm, kick_turn = find_kick(indexed.copy(), n_turns_free=n_turns_free)
    trimmed = df[df["turn"] >= kick_turn].copy()
    trimmed = _drop_upstream_rows_on_kick_turn(trimmed, twiss, kick_bpm, kick_turn)
    trimmed["turn"] = (trimmed["turn"] - kick_turn + 1).astype(df["turn"].dtype)
    return trimmed.reset_index(drop=True)


def _starts_at_kicker(df: pd.DataFrame, kicker_name: str | None) -> bool:
    return kicker_name is not None and not df.empty and str(df.iloc[0]["name"]).upper() == kicker_name.upper()


def _drop_upstream_rows_on_kick_turn(
    df: pd.DataFrame,
    twiss: pd.DataFrame,
    kick_bpm: str,
    kick_turn: int,
) -> pd.DataFrame:
    kick_s = float(twiss.loc[kick_bpm, "s"])
    bpm_s = df["name"].map(twiss["s"])
    keep_mask = ~((df["turn"] == kick_turn) & (bpm_s < kick_s))
    return df.loc[keep_mask].copy()
