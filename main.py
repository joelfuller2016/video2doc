#!/usr/bin/env python3
"""
Video2Doc - AI-Powered Video to Documentation Generator

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

# Import modules
from transcriber import transcribe, get_full_transcript, TranscriptSegment
from analyzer import analyze_transcript, get_screenshot_points
from screenshotter import capture_screenshots, get_video_duration
from generators import generate_docx, generate_pptx, generate_markdown
from generators.markdown_gen import generate_markdown_with_transcript


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Video2Doc - Transform videos into professional documentation",
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
    """Validate command line inputs"""
    # Check video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Check supported video formats
    supported_formats = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    if video_path.suffix.lower() not in supported_formats:
        print(f"Warning: Unsupported video format '{video_path.suffix}'. Attempting anyway...")

    # Check API key (environment variable only for security)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Anthropic API key required.")
        print("Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)


def get_output_path(args, format_type: str) -> str:
    """Generate output path based on format"""
    if args.output and format_type == args.format:
        return args.output

    video_path = Path(args.video)
    base_name = video_path.stem

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

    print("=" * 60)
    print("Video2Doc - AI-Powered Documentation Generator")
    print("=" * 60)
    print()

    # Validate inputs
    validate_inputs(args)

    video_path = Path(args.video)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Create temp directory for screenshots
    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="video2doc_")
    screenshots_dir = Path(temp_dir) / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ===== Step 1: Get Video Info =====
        print(f"[1/5] Analyzing video: {video_path.name}")
        duration = get_video_duration(str(video_path))
        print(f"      Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print()

        # ===== Step 2: Transcribe =====
        print(f"[2/5] Transcribing audio (model: {args.model})...")
        segments = transcribe(
            str(video_path),
            model_size=args.model,
            language=args.language
        )
        print(f"      Transcribed {len(segments)} segments")
        print()

        # ===== Step 3: Analyze with AI =====
        print("[3/5] Analyzing transcript with Claude AI...")
        analysis = analyze_transcript(segments, api_key=api_key)
        print(f"      Title: {analysis.get('title', 'Untitled')}")
        print(f"      Sections: {len(analysis.get('sections', []))}")
        print()

        # ===== Step 4: Capture Screenshots =====
        print("[4/5] Capturing screenshots...")
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

            print(f"      Captured {len(screenshots_map)} screenshots")
        else:
            screenshots_map = {}
            print("      No screenshot points identified")
        print()

        # ===== Step 5: Generate Documents =====
        print("[5/5] Generating documentation...")

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
            print(f"      Generating {fmt.upper()}: {output_path}")

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

        print()
        print("=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        print()
        print("Generated files:")
        for f in generated_files:
            print(f"  - {f}")

        if args.keep_screenshots:
            print(f"\nScreenshots saved in: {screenshots_dir}")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup temp files (always runs, even on error)
        if not args.keep_screenshots and not args.temp_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if args.verbose:
                    print(f"Cleaned up temporary files in {temp_dir}")
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
