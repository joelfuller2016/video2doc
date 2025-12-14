"""
Video Analyzer Module - Intelligent auto-detection of optimal processing parameters
Analyzes video files and recommends processing tier based on size, duration, and system capabilities
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import shutil


class ProcessingTier(Enum):
    """Processing tier based on file characteristics"""
    SMALL = "small"      # <500MB, <30min
    MEDIUM = "medium"    # 500MB-2GB, 30-90min
    LARGE = "large"      # >2GB, >90min


@dataclass
class VideoInfo:
    """Video file metadata"""
    path: str
    file_size_bytes: int
    file_size_mb: float
    duration_seconds: float
    duration_formatted: str
    width: int
    height: int
    fps: float
    codec: str
    bitrate_kbps: int
    has_audio: bool


@dataclass
class ProcessingConfig:
    """Recommended processing configuration based on video analysis"""
    tier: ProcessingTier
    whisper_model: str
    chunking_enabled: bool
    chunk_size_minutes: int
    chunk_overlap_seconds: int
    claude_strategy: str  # 'single-pass', 'hierarchical', 'hierarchical-parallel'
    checkpointing_enabled: bool
    use_gpu: bool
    estimated_time_minutes: float
    recommendations: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "whisper_model": self.whisper_model,
            "chunking_enabled": self.chunking_enabled,
            "chunk_size_minutes": self.chunk_size_minutes,
            "chunk_overlap_seconds": self.chunk_overlap_seconds,
            "claude_strategy": self.claude_strategy,
            "checkpointing_enabled": self.checkpointing_enabled,
            "use_gpu": self.use_gpu,
            "estimated_time_minutes": self.estimated_time_minutes,
            "recommendations": self.recommendations
        }


class VideoAnalyzer:
    """
    Analyzes video files and determines optimal processing parameters.
    Uses FFprobe for video metadata extraction.
    """

    # Tier thresholds
    SMALL_MAX_SIZE_MB = 500
    SMALL_MAX_DURATION_MIN = 30
    MEDIUM_MAX_SIZE_MB = 2048  # 2GB
    MEDIUM_MAX_DURATION_MIN = 90

    # Processing estimates (minutes per minute of video)
    TRANSCRIPTION_RATE_CPU = 0.3  # 3 min video = ~1 min processing
    TRANSCRIPTION_RATE_GPU = 0.1  # 3 min video = ~0.3 min processing
    ANALYSIS_RATE = 0.05  # Claude analysis per minute of video

    def __init__(self):
        self._ffprobe_path = self._find_ffprobe()
        self._gpu_available = self._check_gpu()

    def _find_ffprobe(self) -> Optional[str]:
        """Find FFprobe executable"""
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            return ffprobe
        # Try common locations on Windows
        common_paths = [
            r"C:\ffmpeg\bin\ffprobe.exe",
            r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None

    def _check_gpu(self) -> bool:
        """Check if CUDA GPU is available for acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def gpu_available(self) -> bool:
        """Whether GPU acceleration is available"""
        return self._gpu_available

    def analyze(self, video_path: str) -> Tuple[VideoInfo, ProcessingConfig]:
        """
        Analyze video file and return metadata and recommended processing config.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (VideoInfo, ProcessingConfig)

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If FFprobe fails or is not installed
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video info
        video_info = self._get_video_info(video_path)

        # Determine processing config
        config = self._determine_config(video_info)

        return video_info, config

    def _get_video_info(self, video_path: Path) -> VideoInfo:
        """Extract video metadata using FFprobe"""
        if not self._ffprobe_path:
            raise RuntimeError(
                "FFprobe not found. Please install FFmpeg and ensure it's in PATH."
            )

        cmd = [
            self._ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            data = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFprobe timed out analyzing video")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFprobe failed: {e.stderr}")
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse FFprobe output")

        # Extract video stream info
        video_stream = None
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream

        if not video_stream:
            raise RuntimeError("No video stream found in file")

        format_info = data.get("format", {})

        # File size
        file_size_bytes = int(format_info.get("size", 0))
        if file_size_bytes == 0:
            file_size_bytes = video_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Duration
        duration_seconds = float(format_info.get("duration", 0))
        duration_formatted = self._format_duration(duration_seconds)

        # Video properties
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # FPS
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den else 0
        except (ValueError, ZeroDivisionError):
            fps = 0

        # Codec
        codec = video_stream.get("codec_name", "unknown")

        # Bitrate
        bitrate_kbps = int(format_info.get("bit_rate", 0)) // 1000

        return VideoInfo(
            path=str(video_path),
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            duration_seconds=duration_seconds,
            duration_formatted=duration_formatted,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            has_audio=audio_stream is not None
        )

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _determine_config(self, video_info: VideoInfo) -> ProcessingConfig:
        """Determine optimal processing configuration based on video info"""
        duration_min = video_info.duration_seconds / 60
        size_mb = video_info.file_size_mb

        # Determine tier
        if size_mb < self.SMALL_MAX_SIZE_MB and duration_min < self.SMALL_MAX_DURATION_MIN:
            tier = ProcessingTier.SMALL
        elif size_mb < self.MEDIUM_MAX_SIZE_MB and duration_min < self.MEDIUM_MAX_DURATION_MIN:
            tier = ProcessingTier.MEDIUM
        else:
            tier = ProcessingTier.LARGE

        # Build config based on tier
        recommendations = []

        if tier == ProcessingTier.SMALL:
            config = ProcessingConfig(
                tier=tier,
                whisper_model="base",
                chunking_enabled=False,
                chunk_size_minutes=0,
                chunk_overlap_seconds=0,
                claude_strategy="single-pass",
                checkpointing_enabled=False,
                use_gpu=self._gpu_available,
                estimated_time_minutes=0,
                recommendations=recommendations
            )
            recommendations.append("Small file detected - using single-pass processing")
            if self._gpu_available:
                recommendations.append("GPU acceleration will be used")

        elif tier == ProcessingTier.MEDIUM:
            config = ProcessingConfig(
                tier=tier,
                whisper_model="base",
                chunking_enabled=True,
                chunk_size_minutes=10,
                chunk_overlap_seconds=30,
                claude_strategy="hierarchical",
                checkpointing_enabled=False,
                use_gpu=self._gpu_available,
                estimated_time_minutes=0,
                recommendations=recommendations
            )
            recommendations.append("Medium file detected - using chunked processing")
            recommendations.append("Audio will be processed in 10-minute segments")
            recommendations.append("Claude will analyze chapters separately then synthesize")
            if self._gpu_available:
                recommendations.append("GPU acceleration will speed up transcription")

        else:  # LARGE
            # For very large files, consider using smaller model for speed
            model = "base" if self._gpu_available else "tiny"
            config = ProcessingConfig(
                tier=tier,
                whisper_model=model,
                chunking_enabled=True,
                chunk_size_minutes=5,  # Smaller chunks for memory
                chunk_overlap_seconds=30,
                claude_strategy="hierarchical-parallel",
                checkpointing_enabled=True,
                use_gpu=self._gpu_available,
                estimated_time_minutes=0,
                recommendations=recommendations
            )
            recommendations.append("Large file detected - using aggressive chunking")
            recommendations.append("Checkpointing enabled for crash recovery")
            recommendations.append(f"Using '{model}' model for optimal speed/quality balance")
            if not self._gpu_available:
                recommendations.append("Consider using GPU for faster processing of large files")

        # Estimate processing time
        transcription_rate = self.TRANSCRIPTION_RATE_GPU if self._gpu_available else self.TRANSCRIPTION_RATE_CPU
        estimated_time = (duration_min * transcription_rate) + (duration_min * self.ANALYSIS_RATE)
        config.estimated_time_minutes = round(estimated_time, 1)

        return config

    def get_tier_display(self, tier: ProcessingTier) -> str:
        """Get human-readable tier description"""
        descriptions = {
            ProcessingTier.SMALL: "Small (Quick Processing)",
            ProcessingTier.MEDIUM: "Medium (Chunked Processing)",
            ProcessingTier.LARGE: "Large (Advanced Processing)"
        }
        return descriptions.get(tier, "Unknown")

    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def analyze_video(video_path: str) -> Tuple[VideoInfo, ProcessingConfig]:
    """
    Convenience function to analyze a video file.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (VideoInfo, ProcessingConfig)
    """
    analyzer = VideoAnalyzer()
    return analyzer.analyze(video_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_analyzer.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        analyzer = VideoAnalyzer()
        print(f"GPU Available: {analyzer.gpu_available}")
        print()

        video_info, config = analyzer.analyze(video_path)

        print("=" * 50)
        print("VIDEO INFORMATION")
        print("=" * 50)
        print(f"File: {video_info.path}")
        print(f"Size: {analyzer.format_file_size(video_info.file_size_bytes)} ({video_info.file_size_mb:.1f} MB)")
        print(f"Duration: {video_info.duration_formatted} ({video_info.duration_seconds:.1f}s)")
        print(f"Resolution: {video_info.width}x{video_info.height}")
        print(f"FPS: {video_info.fps:.2f}")
        print(f"Codec: {video_info.codec}")
        print(f"Bitrate: {video_info.bitrate_kbps} kbps")
        print(f"Has Audio: {video_info.has_audio}")

        print()
        print("=" * 50)
        print("PROCESSING RECOMMENDATION")
        print("=" * 50)
        print(f"Detected Tier: {analyzer.get_tier_display(config.tier)}")
        print(f"Whisper Model: {config.whisper_model}")
        print(f"Chunking: {'Enabled' if config.chunking_enabled else 'Disabled'}")
        if config.chunking_enabled:
            print(f"  - Chunk Size: {config.chunk_size_minutes} minutes")
            print(f"  - Overlap: {config.chunk_overlap_seconds} seconds")
        print(f"Claude Strategy: {config.claude_strategy}")
        print(f"Checkpointing: {'Enabled' if config.checkpointing_enabled else 'Disabled'}")
        print(f"GPU Acceleration: {'Yes' if config.use_gpu else 'No'}")
        print(f"Estimated Time: ~{config.estimated_time_minutes} minutes")

        print()
        print("Recommendations:")
        for rec in config.recommendations:
            print(f"  • {rec}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
