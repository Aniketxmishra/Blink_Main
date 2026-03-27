"""
tests/test_blink_sdk.py
=======================
Fast, GPU-free tests for the installable blink-gpu SDK.
Designed to run in GitHub Actions CI (no GPU, no real models needed).
All tests mock or use tiny synthetic models so the suite completes in < 30 s.
"""
import os
import sys

import pytest

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_model():
    """A tiny nn.Linear model — lightweight, no GPU required."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _models_exist() -> bool:
    """True only if the trained .joblib files are present."""
    return os.path.exists(os.path.join(ROOT, "models", "random_forest_model.joblib"))


# ── Package import tests ──────────────────────────────────────────────────────

class TestPackageImports:
    def test_import_blink(self):
        import blink
        assert blink is not None

    def test_version_string(self):
        from blink import __version__
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3, "Version should be semver X.Y.Z"

    def test_blink_predictor_importable(self):
        from blink import BlinkPredictor
        assert BlinkPredictor is not None

    def test_blink_analyzer_importable(self):
        from blink import BlinkAnalyzer
        assert BlinkAnalyzer is not None

    def test_public_api_completeness(self):
        """All __all__ names must be importable."""
        import blink
        for name in blink.__all__:
            assert hasattr(blink, name), f"{name!r} in __all__ but not importable"


# ── BlinkAnalyzer tests ───────────────────────────────────────────────────────

class TestBlinkAnalyzer:
    def test_instantiation(self):
        from blink import BlinkAnalyzer
        az = BlinkAnalyzer()
        assert az is not None

    def test_analyze_returns_dict(self):
        from blink import BlinkAnalyzer
        az = BlinkAnalyzer()
        feats = az.analyze(_tiny_model(), input_shape=(1, 16))
        assert isinstance(feats, dict)

    def test_analyze_has_required_keys(self):
        from blink import BlinkAnalyzer
        feats = BlinkAnalyzer().analyze(_tiny_model(), input_shape=(1, 16))
        required = {"total_parameters", "model_size_mb", "flops"}
        missing = required - feats.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_analyze_parameter_count_correct(self):
        import torch.nn as nn

        from blink import BlinkAnalyzer
        model = nn.Linear(10, 5)   # 10*5 + 5 = 55 parameters
        feats = BlinkAnalyzer().analyze(model, input_shape=(1, 10))
        assert feats["total_parameters"] == 55

    def test_summary_returns_string(self):
        from blink import BlinkAnalyzer
        summary = BlinkAnalyzer().summary(_tiny_model(), input_shape=(1, 16))
        assert isinstance(summary, str)
        assert "Parameters" in summary

    def test_summary_contains_model_size(self):
        from blink import BlinkAnalyzer
        summary = BlinkAnalyzer().summary(_tiny_model(), input_shape=(1, 16))
        assert "Size" in summary

    def test_analyze_conv_model(self):
        """Ensure conv layers are counted correctly."""
        import torch.nn as nn

        from blink import BlinkAnalyzer
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
        )
        feats = BlinkAnalyzer().analyze(model, input_shape=(3, 32, 32))
        assert feats.get("num_conv_layers", 0) >= 2


# ── BlinkPredictor tests (model-file-gated) ───────────────────────────────────

@pytest.mark.skipif(not _models_exist(), reason="Trained model files not present in CI")
class TestBlinkPredictor:
    def test_instantiation(self):
        from blink import BlinkPredictor
        p = BlinkPredictor()
        assert p is not None

    def test_predict_named_model_returns_dict(self):
        from blink import BlinkPredictor
        r = BlinkPredictor().predict("resnet18", batch_size=1)
        assert isinstance(r, dict)

    def test_predict_required_keys(self):
        from blink import BlinkPredictor
        r = BlinkPredictor().predict("resnet18", batch_size=1)
        for key in ("exec_time_ms", "memory_mb"):
            assert key in r, f"Key '{key}' missing from prediction result"

    def test_predict_exec_time_positive(self):
        from blink import BlinkPredictor
        r = BlinkPredictor().predict("resnet18", batch_size=4)
        assert r["exec_time_ms"] > 0

    def test_predict_memory_positive(self):
        from blink import BlinkPredictor
        r = BlinkPredictor().predict("resnet18", batch_size=4)
        assert r["memory_mb"] > 0

    def test_predict_confidence_interval_ordering(self):
        """Lower bound must be ≤ prediction ≤ upper bound."""
        from blink import BlinkPredictor
        r = BlinkPredictor().predict("resnet18", batch_size=8)
        assert r["exec_time_lower"] <= r["exec_time_ms"] <= r["exec_time_upper"]

    def test_predict_batch_monotone(self):
        """Execution time should grow with larger batch sizes (generally)."""
        from blink import BlinkPredictor
        p = BlinkPredictor()
        results = p.predict_batch("resnet18", [1, 8, 32])
        times = [r["exec_time_ms"] for r in results]
        # Not strictly required but a sanity check
        assert times[0] <= times[2], "Exec time should generally increase with batch size"

    def test_predict_batch_length_matches(self):
        from blink import BlinkPredictor
        batch_sizes = [1, 2, 4, 8]
        results = BlinkPredictor().predict_batch("resnet18", batch_sizes)
        assert len(results) == len(batch_sizes)

    def test_predict_unknown_model_raises(self):
        from blink import BlinkPredictor
        with pytest.raises(ValueError, match="Unknown model"):
            BlinkPredictor().predict("not_a_real_model_xyz", batch_size=1)

    def test_predict_custom_module(self):
        from blink import BlinkPredictor
        r = BlinkPredictor().predict(_tiny_model(), batch_size=4)
        assert r["exec_time_ms"] > 0


# ── Version format ────────────────────────────────────────────────────────────

class TestVersion:
    def test_version_semver(self):
        from blink._version import __version__
        major, minor, patch = __version__.split(".")
        assert int(major) >= 0
        assert int(minor) >= 0
        assert int(patch) >= 0

    def test_version_is_not_placeholder(self):
        from blink._version import __version__
        assert __version__ != "0.0.0"
        assert __version__ != "VERSION"
