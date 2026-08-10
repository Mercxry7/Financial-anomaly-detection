"""
Smoke tests: every module should import cleanly and Settings should build
with defaults. These exist to catch import-time regressions (syntax errors,
broken imports, config schema breaks) in CI — they are not a substitute for
real unit/integration tests against the detectors and feature pipeline.
"""


def test_schemas_import():
    import schemas  # noqa: F401


def test_data_loader_import():
    import data_loader  # noqa: F401


def test_feature_engineer_import():
    import feature_engineer  # noqa: F401


def test_models_import():
    import models  # noqa: F401


def test_evaluator_import():
    import evaluator  # noqa: F401


def test_explainer_import():
    import explainer  # noqa: F401


def test_pipeline_import():
    import pipeline  # noqa: F401


def test_config_loads_with_defaults():
    from config import Settings

    settings = Settings()
    assert settings.app_name == "financial-anomaly-detection"
    assert settings.iforest.contamination > 0
    assert settings.lstm.sequence_length > 0
