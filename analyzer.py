"""
Analyzer Module - Uses Claude AI to analyze transcript and determine screenshot locations
Identifies optimal moments for visual documentation based on content analysis
"""

import os
import json
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError

from transcriber import TranscriptSegment, format_timestamp
from logger import get_logger, api_logger

# Module logger
logger = get_logger(__name__)


@dataclass
class ScreenshotPoint:
    """Represents a point where a screenshot should be captured"""
    timestamp: float      # Timestamp in seconds
    reason: str          # Why this moment was selected
    section_title: str   # Title for this section
    description: str     # Description of what's shown


@dataclass
class DocumentSection:
    """Represents a section of the output document"""
    title: str
    content: str
    screenshot_timestamp: Optional[float]
    screenshot_reason: Optional[str]


ANALYSIS_PROMPT = """You are an expert technical writer analyzing a video transcript to create professional documentation.

Your task is to analyze this transcript and identify:
1. The main sections/topics covered
2. Optimal moments for screenshots that would enhance understanding
3. Section titles and descriptions

TRANSCRIPT:
{transcript}

VIDEO DURATION: {duration} seconds

Analyze this transcript and return a JSON response with the following structure:
{{
    "title": "Overall document title based on content",
    "summary": "Brief 1-2 sentence summary of the video content",
    "sections": [
        {{
            "title": "Section title",
            "content": "Content/description for this section (rewrite in clear, professional language)",
            "screenshot_timestamp": 45.5,
            "screenshot_reason": "Why this moment needs a visual (e.g., 'Shows the main interface')"
        }}
    ]
}}

RULES FOR SCREENSHOT SELECTION:
1. Select timestamps where something VISUAL is being demonstrated or explained
2. Look for phrases like: "as you can see", "look at", "here we have", "this shows", "click on", "select"
3. Identify step transitions: "first", "next", "then", "now", "finally"
4. Capture key UI elements, configurations, results, or demonstrations
5. Aim for 1 screenshot per major section (typically 3-8 total for a typical tutorial)
6. Choose timestamps in the MIDDLE of important explanations, not at the very start/end
7. If the transcript mentions specific screens, dialogs, or visual elements, capture those moments

Return ONLY valid JSON, no other text."""


def _extract_json_from_markdown(text: str) -> str:
    """
    Safely extract JSON from markdown code blocks using regex.

    Args:
        text: Response text that may contain markdown code blocks

    Returns:
        Extracted JSON string or original text if no code block found
    """
    # Try to extract from ```json ... ``` block first
    json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_block_pattern, text)
    if match:
        return match.group(1).strip()

    # Try to extract from generic ``` ... ``` block
    generic_block_pattern = r'```\s*([\s\S]*?)\s*```'
    match = re.search(generic_block_pattern, text)
    if match:
        return match.group(1).strip()

    # Return original text if no code block found
    return text.strip()


def _truncate_for_error(text: str, max_length: int = 500) -> str:
    """Truncate text for safe inclusion in error messages."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [truncated, {len(text)} total chars]"


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from AI response with proper error handling.

    Args:
        response_text: Raw response text from Claude API

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If JSON parsing fails
    """
    try:
        # Safely extract JSON from markdown blocks
        json_text = _extract_json_from_markdown(response_text)
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        # Truncate response in error message to avoid information disclosure
        truncated = _truncate_for_error(response_text)
        raise ValueError(
            f"Failed to parse AI response as JSON: {e.msg} at position {e.pos}. "
            f"Response preview: {truncated}"
        )


def _validate_analysis_structure(analysis: Dict[str, Any]) -> None:
    """
    Validate the structure and types of the analysis response.

    Args:
        analysis: Parsed JSON dictionary

    Raises:
        ValueError: If structure validation fails
    """
    # Check required top-level keys
    required_keys = ["title", "summary", "sections"]
    for key in required_keys:
        if key not in analysis:
            raise ValueError(f"Missing required key in AI response: '{key}'")

    # Validate title is a string
    if not isinstance(analysis["title"], str):
        raise ValueError(f"'title' must be a string, got {type(analysis['title']).__name__}")

    # Validate summary is a string
    if not isinstance(analysis["summary"], str):
        raise ValueError(f"'summary' must be a string, got {type(analysis['summary']).__name__}")

    # Validate sections is a list
    if not isinstance(analysis["sections"], list):
        raise ValueError(f"'sections' must be a list, got {type(analysis['sections']).__name__}")

    # Validate each section structure
    for i, section in enumerate(analysis["sections"]):
        if not isinstance(section, dict):
            raise ValueError(f"Section {i} must be a dictionary, got {type(section).__name__}")

        # Validate required section keys
        if "title" not in section:
            raise ValueError(f"Section {i} missing required key 'title'")
        if "content" not in section:
            raise ValueError(f"Section {i} missing required key 'content'")

        # Validate section title and content are strings
        if not isinstance(section["title"], str):
            raise ValueError(f"Section {i} 'title' must be a string")
        if not isinstance(section["content"], str):
            raise ValueError(f"Section {i} 'content' must be a string")

        # Validate screenshot_timestamp if present
        timestamp = section.get("screenshot_timestamp")
        if timestamp is not None:
            if not isinstance(timestamp, (int, float)):
                raise ValueError(
                    f"Section {i} 'screenshot_timestamp' must be a number, "
                    f"got {type(timestamp).__name__}"
                )
            if timestamp < 0:
                raise ValueError(f"Section {i} 'screenshot_timestamp' cannot be negative")


def analyze_transcript(
    segments: List[TranscriptSegment],
    api_key: Optional[str] = None,
    max_retries: int = 3
) -> Dict:
    """
    Analyze transcript using Claude AI to determine document structure and screenshot points.

    Args:
        segments: List of TranscriptSegment from transcription
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        max_retries: Maximum number of API retry attempts

    Returns:
        Dictionary containing document structure and screenshot points

    Raises:
        ValueError: If segments list is empty or API key is missing
        RuntimeError: If API call fails after all retries
    """
    # Validate segments
    if not segments:
        raise ValueError(
            "Cannot analyze empty transcript. "
            "No speech was detected in the video. "
            "Please ensure the video has audible speech."
        )

    # Get API key
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
            "or pass api_key parameter."
        )

    # Build transcript text with timestamps
    transcript_lines = []
    for seg in segments:
        timestamp = format_timestamp(seg.start)
        transcript_lines.append(f"[{timestamp}] ({seg.start:.1f}s) {seg.text}")

    transcript_text = "\n".join(transcript_lines)

    # Calculate duration
    duration = segments[-1].end if segments else 0

    # Build prompt
    prompt = ANALYSIS_PROMPT.format(
        transcript=transcript_text,
        duration=f"{duration:.1f}"
    )

    # Call Claude API with retry logic
    client = Anthropic(api_key=api_key)
    # Security: Clear API key from local scope immediately after client creation
    del api_key

    logger.info("Analyzing transcript with Claude AI...")

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            break  # Success, exit retry loop
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2, 4, 8 seconds
                logger.warning(f"Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"API rate limit exceeded after {max_retries} attempts: {e}")
        except APIConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                logger.warning(f"Connection error. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"API connection failed after {max_retries} attempts: {e}")
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"API error. Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                raise RuntimeError(f"API error after {max_retries} attempts: {e}")

    # Extract and validate response text
    if not hasattr(response, 'content') or not response.content:
        raise RuntimeError(
            "Claude API returned empty response. This may indicate rate limiting, "
            "network issues, or an API error. Please try again."
        )

    first_block = response.content[0]
    if not hasattr(first_block, 'text'):
        raise RuntimeError(
            f"Unexpected response content type: {type(first_block).__name__}. "
            "Expected text response from Claude API."
        )

    response_text = first_block.text

    # Parse JSON response with safer extraction
    analysis = _parse_json_response(response_text)

    # Validate structure and types
    _validate_analysis_structure(analysis)

    logger.info(f"Analysis complete: {len(analysis['sections'])} sections identified")

    return analysis


def get_screenshot_points(analysis: Dict) -> List[ScreenshotPoint]:
    """
    Extract screenshot points from analysis.

    Args:
        analysis: Dictionary from analyze_transcript

    Returns:
        List of ScreenshotPoint objects
    """
    points = []

    for section in analysis.get("sections", []):
        timestamp = section.get("screenshot_timestamp")
        if timestamp is not None:
            points.append(ScreenshotPoint(
                timestamp=float(timestamp),
                reason=section.get("screenshot_reason", "Key moment"),
                section_title=section.get("title", "Section"),
                description=section.get("content", "")
            ))

    # Sort by timestamp
    points.sort(key=lambda p: p.timestamp)

    return points


def get_document_sections(analysis: Dict) -> List[DocumentSection]:
    """
    Convert analysis to DocumentSection objects.

    Args:
        analysis: Dictionary from analyze_transcript

    Returns:
        List of DocumentSection objects
    """
    sections = []

    for section_data in analysis.get("sections", []):
        sections.append(DocumentSection(
            title=section_data.get("title", "Section"),
            content=section_data.get("content", ""),
            screenshot_timestamp=section_data.get("screenshot_timestamp"),
            screenshot_reason=section_data.get("screenshot_reason")
        ))

    return sections


if __name__ == "__main__":
    from logger import init_logging
    init_logging(verbose=True)

    # Test with mock segments
    test_segments = [
        TranscriptSegment(0.0, 5.0, "Hello and welcome to this tutorial on Python basics."),
        TranscriptSegment(5.0, 15.0, "First, let's look at how to install Python. Go to python.org and click download."),
        TranscriptSegment(15.0, 25.0, "As you can see here, we have the installer running. Click next to continue."),
        TranscriptSegment(25.0, 35.0, "Now let's write our first program. Open your text editor."),
        TranscriptSegment(35.0, 45.0, "Type print hello world and save the file as hello.py."),
        TranscriptSegment(45.0, 55.0, "Finally, run the program in your terminal. You should see the output."),
    ]

    try:
        analysis = analyze_transcript(test_segments)
        logger.info("Analysis Result:")
        logger.info(json.dumps(analysis, indent=2))

        points = get_screenshot_points(analysis)
        logger.info(f"Screenshot Points ({len(points)}):")
        for p in points:
            logger.info(f"  - {format_timestamp(p.timestamp)}: {p.reason}")
    except ValueError as e:
        logger.error(f"Error: {e}")
