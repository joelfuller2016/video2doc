"""
Screenshotter Module - Extracts frames from video at specified timestamps
Uses FFmpeg for reliable frame extraction across all video formats
"""

import subprocess
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from PIL import Image

from logger import get_logger, LogContext
from utils.ffmpeg_builder import (
    FFmpegCommandBuilder,
    FFmpegError,
    FFmpegNotFoundError,
    FFmpegTimeoutError,
    FFmpegExecutionError,
    PathValidationError
)
from utils.path_validator import (
    is_safe_path,
    validate_output_directory,
    validate_file_output_path,
)

# Module logger
logger = get_logger(__name__)

# Shared FFmpeg command builder instance
_ffmpeg_builder: Optional[FFmpegCommandBuilder] = None


def _get_builder() -> FFmpegCommandBuilder:
    """Get or create the shared FFmpegCommandBuilder instance."""
    global _ffmpeg_builder
    if _ffmpeg_builder is None:
        _ffmpeg_builder = FFmpegCommandBuilder()
    return _ffmpeg_builder


@dataclass
class Screenshot:
    """Represents a captured screenshot"""
    timestamp: float      # Timestamp in seconds
    filepath: str         # Path to the saved image
    width: int           # Image width
    height: int          # Image height


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using FFprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If FFprobe fails or path validation fails
    """
    logger.debug(f"Getting video duration for: {video_path}")

    try:
        builder = _get_builder()
        cmd = builder.get_duration(video_path, timeout=30)
        result = cmd.execute()
        duration = float(result.stdout.strip())
        logger.debug(f"Video duration: {duration:.1f}s")
        return duration
    except PathValidationError as e:
        logger.error(f"Invalid video path: {e}")
        raise RuntimeError(f"Invalid video path: {e}")
    except FFmpegTimeoutError:
        logger.error("FFprobe timed out getting video duration")
        raise RuntimeError("FFprobe timed out getting video duration")
    except FFmpegNotFoundError as e:
        logger.error(str(e))
        raise RuntimeError(str(e))
    except (FFmpegExecutionError, ValueError) as e:
        logger.error(f"Failed to get video duration: {e}")
        raise RuntimeError(f"Failed to get video duration: {e}")


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Get video resolution using FFprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height)

    Raises:
        RuntimeError: If FFprobe fails or path validation fails
    """
    logger.debug(f"Getting video resolution for: {video_path}")

    try:
        builder = _get_builder()
        cmd = builder.get_resolution(video_path, timeout=30)
        result = cmd.execute()
        width, height = result.stdout.strip().split("x")
        logger.debug(f"Video resolution: {width}x{height}")
        return int(width), int(height)
    except PathValidationError as e:
        logger.error(f"Invalid video path: {e}")
        raise RuntimeError(f"Invalid video path: {e}")
    except FFmpegTimeoutError:
        logger.error("FFprobe timed out getting video resolution")
        raise RuntimeError("FFprobe timed out getting video resolution")
    except FFmpegNotFoundError as e:
        logger.error(str(e))
        raise RuntimeError(str(e))
    except (FFmpegExecutionError, ValueError) as e:
        logger.error(f"Failed to get video resolution: {e}")
        raise RuntimeError(f"Failed to get video resolution: {e}")


def capture_frame(
    video_path: str,
    timestamp: float,
    output_path: str,
    quality: int = 2
) -> Screenshot:
    """
    Capture a single frame from video at specified timestamp.

    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds
        output_path: Path for output image
        quality: JPEG quality (2=best, 31=worst for FFmpeg)

    Returns:
        Screenshot object with metadata

    Raises:
        FileNotFoundError: If video file not found
        RuntimeError: If FFmpeg fails or path validation fails
    """
    video_path_obj = Path(video_path)

    # Validate output path for security
    output_result = validate_file_output_path(
        output_path,
        allowed_extensions=[".png", ".jpg", ".jpeg"],
        create_parent_dirs=True
    )
    if not output_result.is_valid:
        raise ValueError(f"Invalid output path: {output_result.error_message}")
    output_path_obj = Path(output_result.sanitized_value)

    logger.debug(f"Capturing frame at {timestamp}s from {video_path_obj}")

    try:
        builder = _get_builder()
        cmd = builder.capture_frame(
            str(video_path_obj),
            str(output_path_obj),
            timestamp=timestamp,
            quality=quality,
            timeout=60
        )
        cmd.execute()
    except PathValidationError as e:
        if "does not exist" in str(e):
            logger.error(f"Video file not found: {video_path_obj}")
            raise FileNotFoundError(f"Video file not found: {video_path_obj}")
        logger.error(f"Invalid path: {e}")
        raise RuntimeError(f"Invalid path: {e}")
    except FFmpegTimeoutError:
        logger.error(f"Frame extraction timed out at {timestamp}s")
        raise RuntimeError(f"Frame extraction timed out at {timestamp}s")
    except FFmpegNotFoundError as e:
        logger.error(str(e))
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH.")
    except FFmpegExecutionError as e:
        logger.error(f"Frame extraction failed at {timestamp}s: {e}")
        raise RuntimeError(f"Frame extraction failed at {timestamp}s: {e}")

    # Verify output was created
    if not output_path_obj.exists():
        logger.error(f"Frame extraction failed - output not created: {output_path_obj}")
        raise RuntimeError(f"Frame extraction failed - output not created: {output_path_obj}")

    # Get image dimensions
    with Image.open(output_path_obj) as img:
        width, height = img.size

    return Screenshot(
        timestamp=timestamp,
        filepath=str(output_path_obj),
        width=width,
        height=height
    )


def capture_screenshots(
    video_path: str,
    timestamps: List[float],
    output_dir: str,
    prefix: str = "screenshot",
    format: str = "png"
) -> List[Screenshot]:
    """
    Capture multiple screenshots from video.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps in seconds
        output_dir: Directory for output images
        prefix: Filename prefix for screenshots
        format: Output format (png, jpg)

    Returns:
        List of Screenshot objects
    """
    # Validate video path for security
    is_safe, error = is_safe_path(video_path)
    if not is_safe:
        raise ValueError(f"Invalid video path: {error}")
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Validate output directory for security
    output_result = validate_output_directory(output_dir, create_if_missing=True)
    if not output_result.is_valid:
        raise ValueError(f"Invalid output directory: {output_result.error_message}")
    output_dir = Path(output_result.sanitized_value)

    # Get video duration for validation
    duration = get_video_duration(str(video_path))

    screenshots = []
    valid_timestamps = []

    # Filter invalid timestamps
    for ts in timestamps:
        if 0 <= ts <= duration:
            valid_timestamps.append(ts)
        else:
            logger.warning(f"Skipping invalid timestamp {ts}s (video duration: {duration:.1f}s)")

    # Sort timestamps
    valid_timestamps.sort()

    logger.info(f"Capturing {len(valid_timestamps)} screenshots from video")

    for i, timestamp in enumerate(valid_timestamps):
        # Generate filename with index and timestamp
        filename = f"{prefix}_{i+1:03d}_{timestamp:.1f}s.{format}"
        output_path = output_dir / filename

        try:
            screenshot = capture_frame(
                str(video_path),
                timestamp,
                str(output_path),
                quality=2 if format == "jpg" else 1
            )
            screenshots.append(screenshot)
            logger.debug(f"Captured screenshot: {filename}")
        except RuntimeError as e:
            logger.error(f"Failed to capture frame at {timestamp}s: {e}")

    logger.info(f"Successfully captured {len(screenshots)}/{len(valid_timestamps)} screenshots")

    return screenshots


def create_thumbnail(
    image_path: str,
    output_path: str,
    max_size: Tuple[int, int] = (800, 600)
) -> str:
    """
    Create a thumbnail version of an image.

    Args:
        image_path: Path to source image
        output_path: Path for thumbnail
        max_size: Maximum dimensions (width, height)

    Returns:
        Path to thumbnail
    """
    # Validate input path for security
    is_safe, error = is_safe_path(image_path)
    if not is_safe:
        raise ValueError(f"Invalid image path: {error}")

    # Validate output path for security
    output_result = validate_file_output_path(
        output_path,
        allowed_extensions=[".png", ".jpg", ".jpeg", ".gif", ".webp"],
        create_parent_dirs=True
    )
    if not output_result.is_valid:
        raise ValueError(f"Invalid output path: {output_result.error_message}")
    output_path = output_result.sanitized_value

    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(output_path, quality=85, optimize=True)

    return output_path


def format_timestamp_filename(seconds: float) -> str:
    """Convert seconds to filename-safe timestamp string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    else:
        return f"{minutes:02d}m{secs:02d}s"


if __name__ == "__main__":
    import sys
    from logger import init_logging
    init_logging(verbose=True)

    if len(sys.argv) < 2:
        logger.error("Usage: python screenshotter.py <video_path> [timestamp1] [timestamp2] ...")
        sys.exit(1)

    video = sys.argv[1]
    timestamps = [float(t) for t in sys.argv[2:]] if len(sys.argv) > 2 else [0, 10, 30, 60]

    logger.info(f"Video: {video}")
    logger.info(f"Timestamps: {timestamps}")

    try:
        duration = get_video_duration(video)
        resolution = get_video_resolution(video)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Resolution: {resolution[0]}x{resolution[1]}")

        screenshots = capture_screenshots(
            video,
            timestamps,
            output_dir="./screenshots",
            prefix="frame"
        )

        logger.info("Captured Screenshots:")
        for ss in screenshots:
            logger.info(f"  {ss.filepath} ({ss.width}x{ss.height})")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
