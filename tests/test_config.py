"""
Tests for the configuration module.
"""

import pytest
from pathlib import Path

from config import (
    ProcessingSettings,
    AppSettings,
    SettingsManager,
    validate_api_key_format,
    QUALITY_PRESETS,
    WHISPER_MODELS,
)


class TestValidateApiKeyFormat:
    """Tests for API key format validation."""

    def test_valid_api_key(self, mock_api_key: str):
        """Test validation of a valid API key."""
        is_valid, error = validate_api_key_format(mock_api_key)
        assert is_valid is True
        assert error == ""

    def test_empty_api_key(self):
        """Test that empty API key is rejected."""
        is_valid, error = validate_api_key_format("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_api_key_wrong_prefix(self):
        """Test that wrong prefix is rejected."""
        is_valid, error = validate_api_key_format("sk-wrong-prefix-key-12345")
        assert is_valid is False
        assert "sk-ant-" in error

    def test_api_key_too_short(self):
        """Test that short API key is rejected."""
        is_valid, error = validate_api_key_format("sk-ant-short")
        assert is_valid is False
        assert "too short" in error.lower()

    def test_api_key_whitespace(self):
        """Test that whitespace in API key is detected."""
        is_valid, error = validate_api_key_format("  sk-ant-api03-test-key  ")
        assert is_valid is False
        assert "whitespace" in error.lower()

    def test_api_key_invalid_chars(self):
        """Test that invalid characters are rejected."""
        # Key must be long enough (80+ chars) to pass length check first
        invalid_key = "sk-ant-api03-" + "a" * 70 + "$invalid!"
        is_valid, error = validate_api_key_format(invalid_key)
        assert is_valid is False
        assert "invalid characters" in error.lower()


class TestProcessingSettings:
    """Tests for ProcessingSettings dataclass."""

    def test_default_values(self):
        """Test default processing settings values."""
        settings = ProcessingSettings()
        assert settings.auto_detect is True
        assert settings.whisper_model == "base"
        assert settings.enable_chunking is True
        assert settings.chunk_size_minutes == 10
        assert settings.use_gpu is True
        assert settings.quality_preset == "balanced"

    def test_to_dict(self):
        """Test converting settings to dictionary."""
        settings = ProcessingSettings(whisper_model="small")
        data = settings.to_dict()
        assert data["whisper_model"] == "small"
        assert "auto_detect" in data
        assert "enable_chunking" in data

    def test_from_dict(self):
        """Test creating settings from dictionary."""
        data = {"whisper_model": "medium", "use_gpu": False}
        settings = ProcessingSettings.from_dict(data)
        assert settings.whisper_model == "medium"
        assert settings.use_gpu is False
        # Defaults should be preserved
        assert settings.auto_detect is True


class TestAppSettings:
    """Tests for AppSettings dataclass."""

    def test_default_values(self):
        """Test default app settings values."""
        settings = AppSettings()
        assert settings.theme == "dark"
        assert settings.api_key == ""
        assert settings.window_width == 1000
        assert settings.window_height == 700

    def test_to_dict(self):
        """Test converting app settings to dictionary."""
        settings = AppSettings(theme="light")
        data = settings.to_dict()
        assert data["theme"] == "light"
        assert "api_key" in data

    def test_from_dict(self):
        """Test creating app settings from dictionary."""
        data = {"theme": "light", "window_width": 1200}
        settings = AppSettings.from_dict(data)
        assert settings.theme == "light"
        assert settings.window_width == 1200


class TestSettingsManager:
    """Tests for SettingsManager."""

    def test_initialization_creates_directory(self, test_settings_dir: Path):
        """Test that settings manager creates settings directory."""
        manager = SettingsManager(settings_dir=str(test_settings_dir))
        assert test_settings_dir.exists()

    def test_save_and_load(self, test_settings_dir: Path):
        """Test saving and loading settings."""
        manager = SettingsManager(settings_dir=str(test_settings_dir))
        manager.processing_settings.whisper_model = "medium"
        manager.app_settings.theme = "light"
        manager.save()

        # Create new manager and load
        manager2 = SettingsManager(settings_dir=str(test_settings_dir))
        assert manager2.processing_settings.whisper_model == "medium"
        assert manager2.app_settings.theme == "light"

    def test_get_api_key_from_settings(self, test_settings_dir: Path, mock_api_key: str):
        """Test getting API key from settings."""
        manager = SettingsManager(settings_dir=str(test_settings_dir))
        manager.app_settings.api_key = mock_api_key
        assert manager.get_api_key() == mock_api_key

    def test_get_api_key_from_env(self, test_settings_dir: Path, mock_env_api_key, mock_api_key: str):
        """Test getting API key from environment."""
        manager = SettingsManager(settings_dir=str(test_settings_dir))
        manager.app_settings.api_key = ""  # No key in settings
        assert manager.get_api_key() == mock_api_key

    def test_apply_quality_preset(self, test_settings_dir: Path):
        """Test applying quality presets."""
        manager = SettingsManager(settings_dir=str(test_settings_dir))

        manager.apply_quality_preset("speed")
        assert manager.processing_settings.whisper_model == "tiny"
        assert manager.processing_settings.quality_preset == "speed"

        manager.apply_quality_preset("quality")
        assert manager.processing_settings.whisper_model == "small"
        assert manager.processing_settings.quality_preset == "quality"

    def test_reset_to_defaults(self, test_settings_dir: Path, mock_api_key: str):
        """Test resetting to defaults preserves API key."""
        manager = SettingsManager(settings_dir=str(test_settings_dir))
        manager.app_settings.api_key = mock_api_key
        manager.processing_settings.whisper_model = "large-v3"

        manager.reset_to_defaults()

        assert manager.processing_settings.whisper_model == "base"  # Default
        assert manager.app_settings.api_key == mock_api_key  # Preserved


class TestQualityPresets:
    """Tests for quality preset configurations."""

    def test_all_presets_defined(self):
        """Test that all expected presets are defined."""
        expected_presets = ["speed", "balanced", "quality"]
        for preset in expected_presets:
            assert preset in QUALITY_PRESETS

    def test_preset_has_required_fields(self):
        """Test that presets have required configuration fields."""
        required_fields = ["whisper_model", "chunk_size_minutes", "description"]
        for preset_name, preset_config in QUALITY_PRESETS.items():
            for field in required_fields:
                assert field in preset_config, f"Preset '{preset_name}' missing field '{field}'"


class TestWhisperModels:
    """Tests for Whisper model configurations."""

    def test_all_models_defined(self, valid_whisper_models):
        """Test that all expected models are defined."""
        for model in valid_whisper_models:
            assert model in WHISPER_MODELS

    def test_model_has_required_fields(self):
        """Test that model configs have required fields."""
        required_fields = ["name", "vram_gb", "speed", "quality", "description"]
        for model_name, model_config in WHISPER_MODELS.items():
            for field in required_fields:
                assert field in model_config, f"Model '{model_name}' missing field '{field}'"
