"""Decoder fall-through hygiene tests (mood-degeneracy closeout).

The SIGNED_INT16 AttributeError was silently swallowed by the decoder
chain's except, invisibly disabling miniaudio (ffmpeg won that run).
Contract now: an AVAILABLE decoder that RAISES logs a WARNING before
falling through; an ABSENT decoder (ImportError / missing binary) stays
debug-level — warning there would spam every track in HA core.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import _stub_homeassistant

_stub_homeassistant()

# conftest registers `ax_bpm` as a package entry that SKIPS __init__.py
# execution (the full integration __init__ pulls the HA runtime surface).
from ax_bpm import decode as integration_decode


def _fake_miniaudio_module(decode_fn):
    mod = types.ModuleType("miniaudio")
    mod.SampleFormat = types.SimpleNamespace(SIGNED16=1, FLOAT32=3)
    mod.decode = decode_fn
    return mod


class _Decoded:
    sample_format = 1  # SIGNED16
    samples = np.zeros(16, dtype=np.float32)


def test_analyzer_available_decoder_raise_warns_and_falls_through(
    monkeypatch, caplog
):
    """miniaudio raises → WARNING emitted → soundfile result returned."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyzer"))
    from ax_bpm_analyzer import decode as analyzer_decode

    def boom(data, **kwargs):
        raise AttributeError(
            "'SampleFormat' object has no attribute 'SIGNED_INT16'"
        )

    monkeypatch.setitem(
        sys.modules, "miniaudio", _fake_miniaudio_module(boom)
    )
    fallback = (np.ones(16, dtype=np.float32), 16000)
    monkeypatch.setattr(
        analyzer_decode, "_decode_with_soundfile", lambda d, t: fallback
    )

    with caplog.at_level("WARNING", logger="ax_bpm_analyzer.decode"):
        result = analyzer_decode.decode_traced(b"fake-mp3-bytes", 16000)

    assert result is not None
    samples, sr, decoder = result
    assert decoder == "soundfile"
    assert sr == 16000
    assert any(
        "miniaudio decode failed" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    )


def test_analyzer_absent_decoder_stays_debug(monkeypatch, caplog):
    """miniaudio ImportError → NO warning (absent is expected, debug-level)."""
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyzer"))
    from ax_bpm_analyzer import decode as analyzer_decode

    monkeypatch.setitem(sys.modules, "miniaudio", None)  # import → ImportError
    importlib.reload(analyzer_decode)
    fallback = (np.ones(16, dtype=np.float32), 16000)
    monkeypatch.setattr(
        analyzer_decode, "_decode_with_soundfile", lambda d, t: fallback
    )

    with caplog.at_level("WARNING", logger="ax_bpm_analyzer.decode"):
        result = analyzer_decode.decode_traced(b"fake-mp3-bytes", 16000)

    assert result is not None
    assert result[2] == "soundfile"
    assert not any(
        "miniaudio" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    )


def test_integration_available_decoder_raise_warns_and_falls_through(
    monkeypatch, caplog, tmp_path
):
    """Integration copy: miniaudio raises → WARNING → soundfile result."""
    path = str(tmp_path / "track.mp3")
    Path(path).write_bytes(b"fake")

    # The REAL _decode_with_miniaudio must run so its except branch (the
    # code under test) emits the warning: inject a fake miniaudio module
    # whose decode_file raises the SIGNED_INT16-class AttributeError.
    def boom(path_arg, **kwargs):
        raise AttributeError(
            "'SampleFormat' object has no attribute 'SIGNED_INT16'"
        )

    monkeypatch.setitem(
        sys.modules, "miniaudio", _fake_miniaudio_module(boom)
    )
    fallback = (np.ones(16, dtype=np.float32), 16000)
    # _DECODERS captures function references at import time — patch the
    # tuple itself (patching module attributes would not affect it). The
    # real miniaudio decoder stays in; only soundfile is stubbed.
    monkeypatch.setattr(
        integration_decode,
        "_DECODERS",
        (integration_decode._decode_with_miniaudio, lambda p: fallback),
    )

    with caplog.at_level("WARNING", logger="custom_components.ax_bpm.decode"):
        result = integration_decode.decode_mono(path)

    assert result is not None
    samples, sr = result
    assert sr == 16000
    assert any(
        "miniaudio decode failed" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    )


def test_integration_decoder_inventory_logged_once(caplog):
    """log_decoder_inventory emits the one-time live/absent summary."""
    with caplog.at_level("INFO", logger="ax_bpm.decode"):
        integration_decode.log_decoder_inventory()

    inventory = [
        rec.message for rec in caplog.records if "AX BPM decoders" in rec.message
    ]
    assert len(inventory) == 1
    assert "live=" in inventory[0]
    assert "absent=" in inventory[0]
