#!/usr/bin/env python3
"""
FrameNotes - AI-Powered Video to Documentation Generator

Transforms video content into professional documentation with:
- Automatic transcription with timestamps
- AI-powered screenshot placement
- Multiple output formats (DOCX, PPTX, Markdown)

Usage:
    python main.py video.mp4 --output doc.docx
    python main.py video.mp4 --format pptx --output presentation.pptx
    python main.py video.mp4 --format markdown --output readme.md
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from logger import get_logger, init_logging
from utils.input_validator import (
    VIDEO_PATH_VALIDATOR,
    OUTPUT_DIR_VALIDATOR,
    WHISPER_MODEL_VALIDATOR,
    PathValidator,
    StringValidator,
    sanitize_filename,
)

# Module logger
logger = get_logger(__name__)

# Import modules
from transcriber import transcribe, get_full_transcript, TranscriptSegment
from analyzer import analyze_transcript, get_screenshot_points
from screenshotter import capture_screenshots, get_video_duration
from generators import generate_docx, generate_pptx, generate_markdown
from generators.markdown_gen import generate_markdown_with_transcript


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FrameNotes - Transform videos into professional documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py tutorial.mp4
  python main.py lecture.mp4 --format docx --output manual.docx
  python main.py demo.mp4 --format pptx --output slides.pptx
  python main.py video.mp4 --format markdown --include-transcript
  python main.py video.mp4 --model medium --language en
        """
    )

    parser.add_argument(
        "video",
        help="Path to the input video file"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated based on format)"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["docx", "pptx", "markdown", "md", "all"],
        default="docx",
        help="Output format (default: docx)"
    )

    parser.add_argument(
        "-m", "--model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="base",
        help="Whisper model size (default: base)"
    )

    parser.add_argument(
        "-l", "--language",
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detected if not specified."
    )

    parser.add_argument(
        "--include-transcript",
        action="store_true",
        help="Include full transcript in output (markdown only)"
    )

    parser.add_argument(
        "--temp-dir",
        help="Directory for temporary files (default: system temp)"
    )

    parser.add_argument(
        "--keep-screenshots",
        action="store_true",
        help="Keep screenshot files after generation"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def validate_inputs(args) -> None:
    """Validate command line inputs using secure validators."""
    # Validate video path using secure PathValidator
    # This prevents path traversal attacks, null bytes, and validates extension
    video_result = VIDEO_PATH_VALIDATOR.validate(args.video)
    if not video_result.is_valid:
        logger.error(f"Video validation failed: {video_result.error_message}")
        sys.exit(1)

    # Validate whisper model using ChoiceValidator
    model_result = WHISPER_MODEL_VALIDATOR.validate(args.model)
    if not model_result.is_valid:
        logger.error(f"Model validation failed: {model_result.error_message}")
        sys.exit(1)

    # Validate output path if specified
    if args.output:
        # Validate output directory is accessible (create validator without must_exist)
        output_result = OUTPUT_DIR_VALIDATOR.validate(str(Path(args.output).parent))
        if not output_result.is_valid:
            logger.error(f"Output path validation failed: {output_result.error_message}")
            sys.exit(1)

    # Validate temp directory if specified
    if args.temp_dir:
        temp_validator = PathValidator(
            must_exist=True,
            must_be_dir=True,
            field_name="Temp directory"
        )
        temp_result = temp_validator.validate(args.temp_dir)
        if not temp_result.is_valid:
            logger.error(f"Temp directory validation failed: {temp_result.error_message}")
            sys.exit(1)

    # Validate language code if specified (basic string validation)
    if args.language:
        lang_validator = StringValidator(
            min_length=2,
            max_length=5,
            pattern=r'^[a-z]{2,3}(-[A-Z]{2})?$',
            field_name="Language code"
        )
        lang_result = lang_validator.validate(args.language)
        if not lang_result.is_valid:
            logger.error(f"Language validation failed: {lang_result.error_message}")
            sys.exit(1)

    # Check API key (environment variable only for security)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)


def get_output_path(args, format_type: str) -> str:
    """Generate output path based on format"""
    if args.output and format_type == args.format:
        return args.output

    video_path = Path(args.video)
    # Sanitize filename to prevent path traversal attacks
    base_name = sanitize_filename(video_path.stem)

    extension_map = {
        "docx": ".docx",
        "pptx": ".pptx",
        "markdown": ".md",
        "md": ".md"
    }

    extension = extension_map.get(format_type, ".docx")

    # Fix: Handle None args.output properly
    if args.output:
        output_dir = Path(args.output).parent
    else:
        output_dir = video_path.parent

    return str(output_dir / f"{base_name}_documentation{extension}")


def main():
    """Main entry point"""
    args = parse_args()

    # Initialize logging based on verbose flag
    init_logging(verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("FrameNotes - AI-Powered Documentation Generator")
    logger.info("=" * 60)

    # Validate inputs
    validate_inputs(args)

    video_path = Path(args.video)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Create temp directory for screenshots
    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="framenotes_")
    screenshots_dir = Path(temp_dir) / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using temp directory: {temp_dir}")

    try:
        # ===== Step 1: Get Video Info =====
        logger.info(f"[1/5] Analyzing video: {video_path.name}")
        duration = get_video_duration(str(video_path))
        logger.info(f"      Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

        # ===== Step 2: Transcribe =====
        logger.info(f"[2/5] Transcribing audio (model: {args.model})...")
        segments = transcribe(
            str(video_path),
            model_size=args.model,
            language=args.language
        )
        logger.info(f"      Transcribed {len(segments)} segments")

        # ===== Step 3: Analyze with AI =====
        logger.info("[3/5] Analyzing transcript with Claude AI...")
        analysis = analyze_transcript(segments, api_key=api_key)
        # Security: Clear API key from memory immediately after use
        del api_key
        logger.info(f"      Title: {analysis.get('title', 'Untitled')}")
        logger.info(f"      Sections: {len(analysis.get('sections', []))}")

        # ===== Step 4: Capture Screenshots =====
        logger.info("[4/5] Capturing screenshots...")
        screenshot_points = get_screenshot_points(analysis)
        timestamps = [p.timestamp for p in screenshot_points]

        if timestamps:
            screenshot_list = capture_screenshots(
                str(video_path),
                timestamps,
                str(screenshots_dir),
                prefix="frame",
                format="png"
            )

            # Build timestamp -> path mapping
            screenshots_map = {}
            for ss in screenshot_list:
                screenshots_map[ss.timestamp] = ss.filepath

            logger.info(f"      Captured {len(screenshots_map)} screenshots")
        else:
            screenshots_map = {}
            logger.info("      No screenshot points identified")

        # ===== Step 5: Generate Documents =====
        logger.info("[5/5] Generating documentation...")

        title = analysis.get("title", video_path.stem)
        summary = analysis.get("summary", "")
        sections = analysis.get("sections", [])

        formats_to_generate = []
        if args.format == "all":
            formats_to_generate = ["docx", "pptx", "markdown"]
        elif args.format == "md":
            formats_to_generate = ["markdown"]
        else:
            formats_to_generate = [args.format]

        generated_files = []

        for fmt in formats_to_generate:
            output_path = get_output_path(args, fmt)
            logger.info(f"      Generating {fmt.upper()}: {output_path}")

            if fmt == "docx":
                generate_docx(
                    title=title,
                    summary=summary,
                    sections=sections,
                    screenshots=screenshots_map,
                    output_path=output_path
                )

            elif fmt == "pptx":
                generate_pptx(
                    title=title,
                    summary=summary,
                    sections=sections,
                    screenshots=screenshots_map,
                    output_path=output_path
                )

            elif fmt == "markdown":
                if args.include_transcript:
                    generate_markdown_with_transcript(
                        title=title,
                        summary=summary,
                        sections=sections,
                        screenshots=screenshots_map,
                        transcript_segments=segments,
                        output_path=output_path,
                        copy_images=True
                    )
                else:
                    generate_markdown(
                        title=title,
                        summary=summary,
                        sections=sections,
                        screenshots=screenshots_map,
                        output_path=output_path,
                        copy_images=True
                    )

            generated_files.append(output_path)

        logger.info("=" * 60)
        logger.info("COMPLETE!")
        logger.info("=" * 60)
        logger.info("Generated files:")
        for f in generated_files:
            logger.info(f"  - {f}")

        if args.keep_screenshots:
            logger.info(f"Screenshots saved in: {screenshots_dir}")

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Cleanup temp files (always runs, even on error)
        if not args.keep_screenshots and not args.temp_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary files in {temp_dir}")
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
