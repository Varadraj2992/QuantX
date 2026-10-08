from config.settings import settings


def test_settings_default_values() -> None:
    assert settings.app_env in {"development", "production", "testing"}
    assert settings.api_port > 0
    assert settings.streamlit_port > 0
