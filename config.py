"""
Configuration Module - Processing tier settings and application configuration
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum


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
    custom_prompt: str = ""

    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["docx"])
    include_transcript: bool = False
    keep_screenshots: bool = False

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
            "custom_prompt": self.custom_prompt,
            "output_formats": self.output_formats,
            "include_transcript": self.include_transcript,
            "keep_screenshots": self.keep_screenshots,
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
