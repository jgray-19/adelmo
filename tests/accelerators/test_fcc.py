"""FCC accelerator knob selection (no MAD-NG)."""

from aba_optimiser.accelerators import FCC


def test_quad_k1_knob_spec(tmp_path):
    acc = FCC(sequence_file=tmp_path / "fccee_z.madx", errors={"quad": {"k1"}})
    (spec,) = acc.get_knob_specs()
    assert (spec.kind, spec.attribute, spec.pattern, spec.nonzero_attr) == ("quadrupole", "k1", "^Q[DF][123]AM?%.%d+$", "k1")
    assert acc.seq_name == "fccee_p_ring"
    assert acc.infer_monitor_plane("bpm_qd0ar.0") == "HV"


def test_copy_with_keeps_selection(tmp_path):
    acc = FCC(sequence_file=tmp_path / "fccee_z.madx", errors={"quad": {"k1"}})
    copy = acc.copy_with(bpm_pattern="^BPM_Q.*$")
    assert copy.errors == acc.errors
    assert copy.bpm_pattern == "^BPM_Q.*$"
    assert copy.mode == "z"
