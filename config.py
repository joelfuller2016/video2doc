"""
Configuration Module - Processing tier settings and application configuration
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from pathlib import Path
import json
import os


class WhisperModel(Enum):
    """Available Whisper model sizes"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v3"


class QualityPreset(Enum):
    """Processing quality presets"""
    SPEED = "speed"        # Fastest, lower quality
    BALANCED = "balanced"  # Default balance
    QUALITY = "quality"    # Highest quality, slower


@dataclass
class ProcessingSettings:
    """User-configurable processing settings"""
    # Auto-detection
    auto_detect: bool = True

    # Whisper settings
    whisper_model: str = "base"
    language: str = ""  # Empty = auto-detect

    # Chunking settings
    enable_chunking: bool = True
    chunk_size_minutes: int = 10
    chunk_overlap_seconds: int = 30

    # GPU settings
    use_gpu: bool = True  # Will fall back to CPU if unavailable
    compute_type: str = "auto"  # auto, float16, int8

    # Checkpoint settings
    enable_checkpoints: bool = True
    checkpoint_interval_chunks: int = 5

    # Claude settings
    claude_strategy: str = "auto"  # auto, single-pass, hierarchical
    claude_model: str = "claude-sonnet-4-20250514"
    custom_prompt: str = ""

    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["docx"])
    include_transcript: bool = False
    keep_screenshots: bool = False
    super_detailed_output: bool = False

    # Quality preset
    quality_preset: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auto_detect": self.auto_detect,
            "whisper_model": self.whisper_model,
            "language": self.language,
            "enable_chunking": self.enable_chunking,
            "chunk_size_minutes": self.chunk_size_minutes,
            "chunk_overlap_seconds": self.chunk_overlap_seconds,
            "use_gpu": self.use_gpu,
            "compute_type": self.compute_type,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval_chunks": self.checkpoint_interval_chunks,
            "claude_strategy": self.claude_strategy,
            "claude_model": self.claude_model,
            "custom_prompt": self.custom_prompt,
            "output_formats": self.output_formats,
            "include_transcript": self.include_transcript,
            "keep_screenshots": self.keep_screenshots,
            "super_detailed_output": self.super_detailed_output,
            "quality_preset": self.quality_preset
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Quality preset configurations
QUALITY_PRESETS = {
    "speed": {
        "whisper_model": "tiny",
        "chunk_size_minutes": 5,
        "claude_strategy": "single-pass",
        "description": "Fastest processing, good for previews"
    },
    "balanced": {
        "whisper_model": "base",
        "chunk_size_minutes": 10,
        "claude_strategy": "hierarchical",
        "description": "Good balance of speed and quality"
    },
    "quality": {
        "whisper_model": "small",
        "chunk_size_minutes": 15,
        "claude_strategy": "hierarchical",
        "description": "Best quality, slower processing"
    }
}


# Whisper model information
WHISPER_MODELS = {
    "tiny": {
        "name": "Tiny",
        "vram_gb": 1,
        "speed": "Fastest",
        "quality": "Lower",
        "description": "Quick drafts, short videos"
    },
    "base": {
        "name": "Base",
        "vram_gb": 1,
        "speed": "Fast",
        "quality": "Good",
        "description": "General use (recommended)"
    },
    "small": {
        "name": "Small",
        "vram_gb": 2,
        "speed": "Medium",
        "quality": "Better",
        "description": "Better accuracy"
    },
    "medium": {
        "name": "Medium",
        "vram_gb": 5,
        "speed": "Slower",
        "quality": "High",
        "description": "Professional use"
    },
    "large-v3": {
        "name": "Large v3",
        "vram_gb": 10,
        "speed": "Slowest",
        "quality": "Highest",
        "description": "Maximum accuracy"
    }
}


# Supported video formats
SUPPORTED_VIDEO_FORMATS = [
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    ".flv", ".webm", ".m4v", ".mpeg", ".mpg"
]


# Language codes for transcription
LANGUAGE_CODES = {
    "": "Auto-detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi"
}


# Application settings
APP_NAME = "Video2Doc"
APP_VERSION = "2.0.0"
APP_TITLE = f"{APP_NAME} - Documentation Generator"


# UI Theme colors (for CustomTkinter)
THEME_COLORS = {
    "dark": {
        "bg_primary": "#1a1a2e",
        "bg_secondary": "#16213e",
        "accent": "#0f3460",
        "accent_hover": "#1a4a7a",
        "text_primary": "#ffffff",
        "text_secondary": "#a0a0a0",
        "success": "#4ade80",
        "warning": "#fbbf24",
        "error": "#ef4444"
    },
    "light": {
        "bg_primary": "#ffffff",
        "bg_secondary": "#f3f4f6",
        "accent": "#3b82f6",
        "accent_hover": "#2563eb",
        "text_primary": "#1f2937",
        "text_secondary": "#6b7280",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "error": "#dc2626"
    }
}


# Claude AI models
CLAUDE_MODELS = {
    "claude-sonnet-4-20250514": {
        "name": "Claude Sonnet 4",
        "description": "Latest model, best balance of speed and quality (recommended)"
    },
    "claude-3-5-sonnet-20241022": {
        "name": "Claude 3.5 Sonnet",
        "description": "Previous generation, still very capable"
    },
    "claude-3-haiku-20240307": {
        "name": "Claude 3 Haiku",
        "description": "Fastest model, good for quick processing"
    }
}


# Tooltip descriptions for settings
SETTING_TOOLTIPS = {
    "whisper_model": "Transcription model size. Larger models are more accurate but slower and use more memory.",
    "language": "Video language for transcription. Auto-detect works well for most content.",
    "enable_chunking": "Process long videos in segments. Recommended for videos over 30 minutes.",
    "chunk_size_minutes": "Length of each audio segment for processing. Smaller chunks use less memory.",
    "use_gpu": "Use GPU acceleration if available. Significantly faster than CPU processing.",
    "quality_preset": "Quick setting to adjust multiple options at once for desired quality/speed balance.",
    "claude_model": "AI model for generating documentation. Newer models generally produce better results.",
    "claude_strategy": "How to analyze long videos. Auto selects the best strategy based on video length.",
    "output_formats": "File formats to generate. DOCX is recommended for editing, PDF for sharing.",
    "include_transcript": "Include the full timestamped transcript in the output document.",
    "keep_screenshots": "Save extracted screenshots to a folder alongside the document.",
    "super_detailed_output": "Generate comprehensive documentation with detailed chapter breakdowns, "
                           "complete technical explanations, step-by-step procedures, and extended summaries.",
    "api_key": "Your Anthropic API key for Claude AI. Get one at console.anthropic.com.",
    "theme": "Application color theme. Dark mode is easier on the eyes in low light."
}


@dataclass
class AppSettings:
    """Application-wide settings (non-processing)"""
    theme: str = "dark"  # dark, light, system
    api_key: str = ""
    last_output_dir: str = ""
    last_video_dir: str = ""
    window_width: int = 1000
    window_height: int = 700

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme": self.theme,
            "api_key": self.api_key,
            "last_output_dir": self.last_output_dir,
            "last_video_dir": self.last_video_dir,
            "window_width": self.window_width,
            "window_height": self.window_height
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SettingsManager:
    """
    Manages application settings with JSON persistence.
    Settings are stored in the user's home directory.
    """

    DEFAULT_SETTINGS_FILE = "video2doc_settings.json"

    def __init__(self, settings_dir: Optional[str] = None):
        """
        Initialize settings manager.

        Args:
            settings_dir: Directory to store settings. Defaults to user's home directory.
        """
        if settings_dir:
            self.settings_dir = Path(settings_dir)
        else:
            self.settings_dir = Path.home() / ".video2doc"

        self.settings_file = self.settings_dir / self.DEFAULT_SETTINGS_FILE
        self.app_settings = AppSettings()
        self.processing_settings = ProcessingSettings()

        # Ensure settings directory exists
        self.settings_dir.mkdir(parents=True, exist_ok=True)

        # Load settings if they exist
        self.load()

    def load(self) -> bool:
        """
        Load settings from JSON file.

        Returns:
            True if settings were loaded successfully, False otherwise.
        """
        if not self.settings_file.exists():
            return False

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "app" in data:
                self.app_settings = AppSettings.from_dict(data["app"])
            if "processing" in data:
                self.processing_settings = ProcessingSettings.from_dict(data["processing"])

            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load settings: {e}")
            return False

    def save(self) -> bool:
        """
        Save settings to JSON file.

        Returns:
            True if settings were saved successfully, False otherwise.
        """
        try:
            data = {
                "app": self.app_settings.to_dict(),
                "processing": self.processing_settings.to_dict()
            }

            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            return True
        except IOError as e:
            print(f"Warning: Could not save settings: {e}")
            return False

    def get_api_key(self) -> str:
        """Get the Claude API key."""
        # First check settings, then environment variable
        if self.app_settings.api_key:
            return self.app_settings.api_key
        return os.environ.get("ANTHROPIC_API_KEY", "")

    def set_api_key(self, key: str) -> None:
        """Set the Claude API key."""
        self.app_settings.api_key = key
        self.save()

    def apply_quality_preset(self, preset: str) -> None:
        """Apply a quality preset to processing settings."""
        if preset not in QUALITY_PRESETS:
            return

        preset_config = QUALITY_PRESETS[preset]
        self.processing_settings.quality_preset = preset
        self.processing_settings.whisper_model = preset_config["whisper_model"]
        self.processing_settings.chunk_size_minutes = preset_config["chunk_size_minutes"]
        self.processing_settings.claude_strategy = preset_config["claude_strategy"]

    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        api_key = self.app_settings.api_key  # Preserve API key
        self.app_settings = AppSettings()
        self.app_settings.api_key = api_key
        self.processing_settings = ProcessingSettings()
        self.save()


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
