"""
Transcriber Module - Extracts audio and generates timestamped transcript
Uses faster-whisper for efficient speech-to-text conversion
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from faster_whisper import WhisperModel

from utils.ffmpeg_builder import (
    FFmpegCommandBuilder,
    FFmpegError,
    FFmpegNotFoundError,
    FFmpegTimeoutError,
    FFmpegExecutionError,
    PathValidationError
)

# Shared FFmpeg command builder instance
_ffmpeg_builder: Optional[FFmpegCommandBuilder] = None


def _get_builder() -> FFmpegCommandBuilder:
    """Get or create the shared FFmpegCommandBuilder instance."""
    global _ffmpeg_builder
    if _ffmpeg_builder is None:
        _ffmpeg_builder = FFmpegCommandBuilder()
    return _ffmpeg_builder


@dataclass
class TranscriptSegment:
    """Represents a single segment of transcribed speech"""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str     # Transcribed text

    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }


def _check_gpu_available() -> bool:
    """
    Check if CUDA GPU is available for Whisper acceleration.

    Returns:
        True if CUDA GPU is available, False otherwise.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            return True
        return False
    except ImportError:
        return False
    except Exception:
        return False


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from video file using FFmpeg.

    Args:
        video_path: Path to the video file
        output_path: Optional path for output audio file

    Returns:
        Path to the extracted audio file

    Raises:
        FileNotFoundError: If video file not found
        RuntimeError: If FFmpeg fails or path validation fails
    """
    # Create secure temp file if no output path specified
    # Using mkstemp for atomic creation with unpredictable name to prevent race conditions
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="framenotes_audio_")
        os.close(fd)  # Close fd since FFmpeg will write to this path

    try:
        builder = _get_builder()
        cmd = builder.extract_audio(
            str(video_path),
            output_path,
            sample_rate=16000,  # Optimal for Whisper
            channels=1,         # Mono
            timeout=600         # 10 minute timeout for large videos
        )
        cmd.execute()
    except PathValidationError as e:
        if "does not exist" in str(e):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        raise RuntimeError(f"Invalid path: {e}")
    except FFmpegTimeoutError:
        raise RuntimeError("FFmpeg audio extraction timed out. Video may be too large or corrupt.")
    except FFmpegNotFoundError as e:
        raise RuntimeError(str(e))
    except FFmpegExecutionError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e}")

    return output_path


def transcribe(
    video_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto"
) -> List[TranscriptSegment]:
    """
    Transcribe video/audio file to text with timestamps.

    Args:
        video_path: Path to video or audio file
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        language: Optional language code (auto-detected if None)
        device: Compute device (auto, cpu, cuda)

    Returns:
        List of TranscriptSegment objects with timestamps
    """
    video_path = Path(video_path)

    # Check if input is audio or video
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    is_audio = video_path.suffix.lower() in audio_extensions

    # Extract audio if video file
    if is_audio:
        audio_path = str(video_path)
        cleanup_audio = False
    else:
        print(f"Extracting audio from video...")
        audio_path = extract_audio(str(video_path))
        cleanup_audio = True

    try:
        # Determine actual device (with GPU auto-detection)
        if device == "auto":
            if _check_gpu_available():
                actual_device = "cuda"
                compute_type = "float16"
            else:
                actual_device = "cpu"
                compute_type = "int8"
        elif device == "cuda":
            actual_device = "cuda"
            compute_type = "float16"
        else:
            actual_device = "cpu"
            compute_type = "int8"

        print(f"Loading Whisper model ({model_size}) on {actual_device}...")
        model = WhisperModel(
            model_size,
            device=actual_device,
            compute_type=compute_type
        )

        print(f"Transcribing audio...")
        segments_generator, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=False,  # Segment-level is sufficient
            vad_filter=True,        # Voice activity detection
        )

        # Convert generator to list of TranscriptSegment
        segments = []
        for segment in segments_generator:
            segments.append(TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip()
            ))

        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"Transcribed {len(segments)} segments")

        return segments

    finally:
        # Cleanup Whisper model to release GPU memory
        if 'model' in locals():
            del model
            # Clear CUDA cache if using GPU
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Cleanup temporary audio file
        if cleanup_audio and os.path.exists(audio_path):
            os.remove(audio_path)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def get_full_transcript(segments: List[TranscriptSegment]) -> str:
    """
    Combine segments into full transcript with timestamps.

    Args:
        segments: List of TranscriptSegment objects

    Returns:
        Formatted transcript string
    """
    lines = []
    for seg in segments:
        timestamp = format_timestamp(seg.start)
        lines.append(f"[{timestamp}] {seg.text}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with a video file
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        segments = transcribe(video, model_size="base")
        print("\n" + "="*50)
        print("TRANSCRIPT:")
        print("="*50)
        print(get_full_transcript(segments))
