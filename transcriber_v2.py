"""
Transcriber Module v2.0 - Enhanced with chunking and GPU support
Extracts audio and generates timestamped transcript with support for large files
"""

import subprocess
import tempfile
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass

from faster_whisper import WhisperModel


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


@dataclass
class ChunkInfo:
    """Information about a processing chunk"""
    index: int
    start_time: float
    end_time: float
    audio_path: str
    segments: List[TranscriptSegment] = None


class ChunkedTranscriber:
    """
    Enhanced transcriber with chunking support for large files.
    Supports GPU acceleration and progress callbacks.
    """

    def __init__(
        self,
        chunk_size_minutes: int = 10,
        overlap_seconds: int = 30,
        use_gpu: bool = True
    ):
        """
        Initialize chunked transcriber.

        Args:
            chunk_size_minutes: Size of each audio chunk in minutes
            overlap_seconds: Overlap between chunks to prevent word cuts
            use_gpu: Whether to use GPU acceleration if available
        """
        self.chunk_size_seconds = chunk_size_minutes * 60
        self.overlap_seconds = overlap_seconds
        self.use_gpu = use_gpu and self._check_gpu()
        self.model = None
        self._ffmpeg_path = self._find_ffmpeg()

    def _check_gpu(self) -> bool:
        """Check if CUDA GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _find_ffmpeg(self) -> Optional[str]:
        """Find FFmpeg executable"""
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None

    @property
    def gpu_available(self) -> bool:
        """Whether GPU is available and enabled"""
        return self.use_gpu

    def _load_model(self, model_size: str):
        """Load Whisper model with appropriate settings"""
        if self.model is not None:
            return

        # Determine compute settings based on GPU availability
        if self.use_gpu:
            device = "cuda"
            compute_type = "float16"  # Best for GPU
        else:
            device = "cpu"
            compute_type = "int8"  # Good balance for CPU

        print(f"Loading Whisper model ({model_size}) on {device}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file using FFprobe"""
        cmd = [
            self._ffmpeg_path.replace("ffmpeg", "ffprobe"),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
        except Exception:
            return 0

    def _extract_audio_chunk(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        output_path: str
    ) -> str:
        """Extract a chunk of audio from video"""
        cmd = [
            self._ffmpeg_path,
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            output_path
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")

        return output_path

    def _extract_full_audio(self, video_path: str, output_path: str) -> str:
        """Extract full audio from video"""
        cmd = [
            self._ffmpeg_path,
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            output_path
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")

        return output_path

    def _merge_overlapping_segments(
        self,
        all_segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """Merge segments from overlapping chunks, removing duplicates"""
        if not all_segments:
            return []

        # Sort by start time
        sorted_segments = sorted(all_segments, key=lambda s: s.start)

        merged = []
        for segment in sorted_segments:
            if not merged:
                merged.append(segment)
                continue

            last = merged[-1]

            # Check for overlap (within 1 second)
            if segment.start <= last.end + 1:
                # Check for duplicate text (fuzzy match)
                if self._is_duplicate_text(last.text, segment.text):
                    # Extend the last segment if this one goes further
                    if segment.end > last.end:
                        merged[-1] = TranscriptSegment(
                            start=last.start,
                            end=segment.end,
                            text=last.text
                        )
                else:
                    # Different text, adjust start time and add
                    adjusted_start = max(segment.start, last.end)
                    if adjusted_start < segment.end:
                        merged.append(TranscriptSegment(
                            start=adjusted_start,
                            end=segment.end,
                            text=segment.text
                        ))
            else:
                merged.append(segment)

        return merged

    def _is_duplicate_text(self, text1: str, text2: str) -> bool:
        """Check if two text segments are likely duplicates"""
        # Simple overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        # Check Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union > 0.7 if union > 0 else False

    def transcribe(
        self,
        video_path: str,
        model_size: str = "base",
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        enable_chunking: bool = True
    ) -> List[TranscriptSegment]:
        """
        Transcribe video/audio file with optional chunking.

        Args:
            video_path: Path to video or audio file
            model_size: Whisper model size
            language: Language code (auto-detected if None)
            progress_callback: Callback(current_chunk, total_chunks, status)
            enable_chunking: Whether to use chunked processing

        Returns:
            List of TranscriptSegment objects with timestamps
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load model
        self._load_model(model_size)

        # Create temp directory for audio chunks
        temp_dir = tempfile.mkdtemp(prefix="video2doc_audio_")

        try:
            # Extract full audio first
            if progress_callback:
                progress_callback(0, 1, "Extracting audio...")

            full_audio_path = os.path.join(temp_dir, "full_audio.wav")
            self._extract_full_audio(str(video_path), full_audio_path)

            # Get audio duration
            duration = self._get_audio_duration(full_audio_path)

            if not enable_chunking or duration < self.chunk_size_seconds * 1.5:
                # Process without chunking for short videos
                return self._transcribe_single(
                    full_audio_path,
                    language,
                    progress_callback
                )

            # Process with chunking
            return self._transcribe_chunked(
                full_audio_path,
                duration,
                language,
                progress_callback,
                temp_dir
            )

        finally:
            # Cleanup
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    def _transcribe_single(
        self,
        audio_path: str,
        language: Optional[str],
        progress_callback: Optional[Callable]
    ) -> List[TranscriptSegment]:
        """Transcribe audio without chunking"""
        if progress_callback:
            progress_callback(0, 1, "Transcribing audio...")

        segments_generator, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=False,
            vad_filter=True
        )

        segments = []
        for segment in segments_generator:
            segments.append(TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip()
            ))

        if progress_callback:
            progress_callback(1, 1, f"Transcribed {len(segments)} segments")

        print(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
        print(f"Transcribed {len(segments)} segments")

        return segments

    def _transcribe_chunked(
        self,
        audio_path: str,
        duration: float,
        language: Optional[str],
        progress_callback: Optional[Callable],
        temp_dir: str
    ) -> List[TranscriptSegment]:
        """Transcribe audio with chunking"""
        # Calculate chunks
        chunks = []
        current_time = 0
        chunk_index = 0

        while current_time < duration:
            chunk_end = min(current_time + self.chunk_size_seconds, duration)
            chunk_duration = chunk_end - current_time

            # Add overlap for non-first chunks
            if current_time > 0:
                chunk_start = max(0, current_time - self.overlap_seconds)
            else:
                chunk_start = current_time

            chunks.append(ChunkInfo(
                index=chunk_index,
                start_time=chunk_start,
                end_time=chunk_end,
                audio_path=os.path.join(temp_dir, f"chunk_{chunk_index}.wav")
            ))

            current_time = chunk_end
            chunk_index += 1

        total_chunks = len(chunks)
        print(f"Processing {total_chunks} audio chunks...")

        all_segments = []
        detected_language = language

        for chunk in chunks:
            if progress_callback:
                progress_callback(
                    chunk.index + 1,
                    total_chunks,
                    f"Transcribing chunk {chunk.index + 1}/{total_chunks}..."
                )

            # Extract chunk audio
            chunk_duration = chunk.end_time - chunk.start_time
            self._extract_audio_chunk(
                audio_path,
                chunk.start_time,
                chunk_duration,
                chunk.audio_path
            )

            # Transcribe chunk
            segments_generator, info = self.model.transcribe(
                chunk.audio_path,
                language=detected_language,
                beam_size=5,
                word_timestamps=False,
                vad_filter=True
            )

            # Use detected language for subsequent chunks
            if detected_language is None:
                detected_language = info.language
                print(f"Detected language: {detected_language}")

            # Adjust timestamps to global timeline
            for segment in segments_generator:
                # Calculate offset for this chunk
                if chunk.index > 0:
                    # Account for overlap
                    offset = chunk.start_time
                else:
                    offset = 0

                global_start = offset + segment.start
                global_end = offset + segment.end

                all_segments.append(TranscriptSegment(
                    start=global_start,
                    end=global_end,
                    text=segment.text.strip()
                ))

            # Cleanup chunk file
            try:
                os.remove(chunk.audio_path)
            except Exception:
                pass

        # Merge overlapping segments
        merged_segments = self._merge_overlapping_segments(all_segments)

        if progress_callback:
            progress_callback(total_chunks, total_chunks, f"Transcribed {len(merged_segments)} segments")

        print(f"Transcribed {len(merged_segments)} segments (from {len(all_segments)} raw)")

        return merged_segments


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
    """Combine segments into full transcript with timestamps"""
    lines = []
    for seg in segments:
        timestamp = format_timestamp(seg.start)
        lines.append(f"[{timestamp}] {seg.text}")
    return "\n".join(lines)


# Backward-compatible function
def transcribe(
    video_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    enable_chunking: bool = False,
    chunk_size_minutes: int = 10,
    progress_callback: Optional[Callable] = None
) -> List[TranscriptSegment]:
    """
    Transcribe video/audio file to text with timestamps.
    Backward-compatible wrapper around ChunkedTranscriber.

    Args:
        video_path: Path to video or audio file
        model_size: Whisper model size
        language: Language code (auto-detected if None)
        device: Compute device (auto, cpu, cuda)
        enable_chunking: Enable chunked processing for large files
        chunk_size_minutes: Size of chunks in minutes
        progress_callback: Optional progress callback

    Returns:
        List of TranscriptSegment objects
    """
    use_gpu = device != "cpu"

    transcriber = ChunkedTranscriber(
        chunk_size_minutes=chunk_size_minutes,
        overlap_seconds=30,
        use_gpu=use_gpu
    )

    return transcriber.transcribe(
        video_path=video_path,
        model_size=model_size,
        language=language,
        progress_callback=progress_callback,
        enable_chunking=enable_chunking
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video = sys.argv[1]

        # Test with chunking
        transcriber = ChunkedTranscriber(
            chunk_size_minutes=5,
            overlap_seconds=30,
            use_gpu=True
        )

        print(f"GPU Available: {transcriber.gpu_available}")

        def progress(current, total, status):
            print(f"[{current}/{total}] {status}")

        segments = transcriber.transcribe(
            video,
            model_size="base",
            progress_callback=progress,
            enable_chunking=True
        )

        print("\n" + "=" * 50)
        print("TRANSCRIPT:")
        print("=" * 50)
        print(get_full_transcript(segments))
