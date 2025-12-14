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
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return float(result.stdout.strip())
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFprobe timed out getting video duration")
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get video duration: {e}")


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Get video resolution using FFprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height)
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        width, height = result.stdout.strip().split("x")
        return int(width), int(height)
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFprobe timed out getting video resolution")
    except (subprocess.CalledProcessError, ValueError) as e:
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
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to extract single frame
    # Using -ss before -i for faster seeking
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),         # Seek to timestamp
        "-i", str(video_path),         # Input file
        "-vframes", "1",               # Extract 1 frame
        "-q:v", str(quality),          # Quality setting
        "-y",                          # Overwrite output
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=60  # 1 minute timeout per frame
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Frame extraction timed out at {timestamp}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Frame extraction failed at {timestamp}s: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH.")

    # Verify output was created
    if not output_path.exists():
        raise RuntimeError(f"Frame extraction failed - output not created: {output_path}")

    # Get image dimensions
    with Image.open(output_path) as img:
        width, height = img.size

    return Screenshot(
        timestamp=timestamp,
        filepath=str(output_path),
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
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration for validation
    duration = get_video_duration(str(video_path))

    screenshots = []
    valid_timestamps = []

    # Filter invalid timestamps
    for ts in timestamps:
        if 0 <= ts <= duration:
            valid_timestamps.append(ts)
        else:
            print(f"Warning: Skipping invalid timestamp {ts}s (video duration: {duration:.1f}s)")

    # Sort timestamps
    valid_timestamps.sort()

    print(f"Capturing {len(valid_timestamps)} screenshots...")

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
            print(f"  Captured: {filename}")
        except RuntimeError as e:
            print(f"  Failed at {timestamp}s: {e}")

    print(f"Successfully captured {len(screenshots)} screenshots")

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

    if len(sys.argv) < 2:
        print("Usage: python screenshotter.py <video_path> [timestamp1] [timestamp2] ...")
        sys.exit(1)

    video = sys.argv[1]
    timestamps = [float(t) for t in sys.argv[2:]] if len(sys.argv) > 2 else [0, 10, 30, 60]

    print(f"Video: {video}")
    print(f"Timestamps: {timestamps}")

    try:
        duration = get_video_duration(video)
        resolution = get_video_resolution(video)
        print(f"Duration: {duration:.1f}s")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")

        screenshots = capture_screenshots(
            video,
            timestamps,
            output_dir="./screenshots",
            prefix="frame"
        )

        print("\nCaptured Screenshots:")
        for ss in screenshots:
            print(f"  {ss.filepath} ({ss.width}x{ss.height})")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
