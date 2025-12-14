"""
Markdown Generator - Creates Markdown documents from video analysis
Produces portable, version-controllable documentation with image references
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import shutil


def generate_markdown(
    title: str,
    summary: str,
    sections: List[Dict],
    screenshots: Dict[float, str],
    output_path: str,
    copy_images: bool = True,
    images_folder: str = "images"
) -> str:
    """
    Generate a Markdown document from video analysis.

    Args:
        title: Document title
        summary: Brief summary/introduction
        sections: List of section dictionaries with title, content, screenshot_timestamp
        screenshots: Dict mapping timestamps to image file paths
        output_path: Path for output .md file
        copy_images: Whether to copy images to a local folder
        images_folder: Name of the images subfolder

    Returns:
        Path to generated document
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images directory if copying images
    if copy_images and screenshots:
        images_dir = output_dir / images_folder
        images_dir.mkdir(parents=True, exist_ok=True)

    lines = []

    # Title
    lines.append(f"# {title}")
    lines.append("")

    # Metadata
    lines.append(f"> Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    lines.append(f"> Sections: {len(sections)} | Screenshots: {len(screenshots)}")
    lines.append("")

    # Summary
    if summary:
        lines.append("## Overview")
        lines.append("")
        lines.append(summary)
        lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    for i, section in enumerate(sections, 1):
        section_title = section.get("title", f"Section {i}")
        # Create anchor-friendly slug
        slug = section_title.lower().replace(" ", "-").replace(".", "")
        lines.append(f"{i}. [{section_title}](#{slug})")
    lines.append("")

    # Horizontal rule
    lines.append("---")
    lines.append("")

    # Sections
    for i, section in enumerate(sections, 1):
        section_title = section.get("title", f"Section {i}")
        section_content = section.get("content", "")
        screenshot_ts = section.get("screenshot_timestamp")
        screenshot_reason = section.get("screenshot_reason", "")

        # Section heading
        lines.append(f"## {i}. {section_title}")
        lines.append("")

        # Section content
        if section_content:
            lines.append(section_content)
            lines.append("")

        # Add screenshot if available
        if screenshot_ts is not None:
            # Find matching screenshot
            screenshot_path = None

            if screenshot_ts in screenshots:
                screenshot_path = screenshots[screenshot_ts]
            else:
                # Find closest timestamp
                closest_ts = min(screenshots.keys(), key=lambda x: abs(x - screenshot_ts), default=None)
                if closest_ts is not None and abs(closest_ts - screenshot_ts) < 5:
                    screenshot_path = screenshots[closest_ts]

            if screenshot_path and os.path.exists(screenshot_path):
                src_path = Path(screenshot_path)

                if copy_images:
                    # Copy image to local folder
                    dest_filename = f"figure_{i:02d}_{src_path.name}"
                    dest_path = images_dir / dest_filename
                    shutil.copy2(src_path, dest_path)
                    image_ref = f"{images_folder}/{dest_filename}"
                else:
                    # Use absolute or relative path
                    image_ref = str(screenshot_path)

                # Add image with alt text and caption
                alt_text = screenshot_reason if screenshot_reason else f"Screenshot at {screenshot_ts:.1f}s"
                lines.append(f"![{alt_text}]({image_ref})")
                lines.append("")

                # Add caption as italics
                if screenshot_reason:
                    lines.append(f"*Figure {i}: {screenshot_reason}*")
                    lines.append("")

        # Add spacing
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("## Document Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Generator | Video2Doc |")
    lines.append(f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| Sections | {len(sections)} |")
    lines.append(f"| Screenshots | {len(screenshots)} |")
    lines.append("")

    # Write file
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")

    return str(output_path)


def generate_markdown_with_transcript(
    title: str,
    summary: str,
    sections: List[Dict],
    screenshots: Dict[float, str],
    transcript_segments: List,
    output_path: str,
    copy_images: bool = True,
    images_folder: str = "images"
) -> str:
    """
    Generate Markdown with full transcript appendix.

    Args:
        title: Document title
        summary: Brief summary
        sections: List of section dictionaries
        screenshots: Dict mapping timestamps to image paths
        transcript_segments: List of TranscriptSegment objects
        output_path: Path for output file
        copy_images: Whether to copy images locally
        images_folder: Images subfolder name

    Returns:
        Path to generated document
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images directory if copying images
    if copy_images and screenshots:
        images_dir = output_dir / images_folder
        images_dir.mkdir(parents=True, exist_ok=True)

    lines = []

    # Title
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    lines.append("")

    # Summary
    if summary:
        lines.append("## Overview")
        lines.append("")
        lines.append(summary)
        lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    for i, section in enumerate(sections, 1):
        section_title = section.get("title", f"Section {i}")
        slug = section_title.lower().replace(" ", "-").replace(".", "")
        lines.append(f"{i}. [{section_title}](#{slug})")
    lines.append(f"{len(sections) + 1}. [Full Transcript](#full-transcript)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sections (same as above)
    for i, section in enumerate(sections, 1):
        section_title = section.get("title", f"Section {i}")
        section_content = section.get("content", "")
        screenshot_ts = section.get("screenshot_timestamp")
        screenshot_reason = section.get("screenshot_reason", "")

        lines.append(f"## {i}. {section_title}")
        lines.append("")

        if section_content:
            lines.append(section_content)
            lines.append("")

        if screenshot_ts is not None:
            screenshot_path = None
            if screenshot_ts in screenshots:
                screenshot_path = screenshots[screenshot_ts]
            else:
                closest_ts = min(screenshots.keys(), key=lambda x: abs(x - screenshot_ts), default=None)
                if closest_ts is not None and abs(closest_ts - screenshot_ts) < 5:
                    screenshot_path = screenshots[closest_ts]

            if screenshot_path and os.path.exists(screenshot_path):
                src_path = Path(screenshot_path)
                if copy_images:
                    dest_filename = f"figure_{i:02d}_{src_path.name}"
                    dest_path = images_dir / dest_filename
                    shutil.copy2(src_path, dest_path)
                    image_ref = f"{images_folder}/{dest_filename}"
                else:
                    image_ref = str(screenshot_path)

                alt_text = screenshot_reason if screenshot_reason else f"Screenshot"
                lines.append(f"![{alt_text}]({image_ref})")
                lines.append("")
                if screenshot_reason:
                    lines.append(f"*Figure {i}: {screenshot_reason}*")
                    lines.append("")

        lines.append("")

    # Full Transcript Appendix
    lines.append("---")
    lines.append("")
    lines.append("## Full Transcript")
    lines.append("")
    lines.append("The complete transcript with timestamps:")
    lines.append("")
    lines.append("```")

    for seg in transcript_segments:
        # Format timestamp
        seconds = seg.start
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            timestamp = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            timestamp = f"{minutes:02d}:{secs:02d}"

        lines.append(f"[{timestamp}] {seg.text}")

    lines.append("```")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by Video2Doc on {datetime.now().strftime('%Y-%m-%d')}*")

    # Write file
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")

    return str(output_path)


if __name__ == "__main__":
    # Test with mock data
    test_sections = [
        {
            "title": "Introduction",
            "content": "This document provides a guide to the topic.",
            "screenshot_timestamp": 5.0,
            "screenshot_reason": "Opening screen"
        },
        {
            "title": "Main Content",
            "content": "The core concepts are explained here.",
            "screenshot_timestamp": 30.0,
            "screenshot_reason": "Main interface"
        }
    ]

    output = generate_markdown(
        title="Test Documentation",
        summary="A test markdown document.",
        sections=test_sections,
        screenshots={},
        output_path="./test_output.md",
        copy_images=False
    )

    print(f"Generated: {output}")
    print("\nContent:")
    print(Path(output).read_text())
