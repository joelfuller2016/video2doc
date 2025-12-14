"""
Analyzer Module - Uses Claude AI to analyze transcript and determine screenshot locations
Identifies optimal moments for visual documentation based on content analysis
"""

import os
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError

from transcriber import TranscriptSegment, format_timestamp


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

    print("Analyzing transcript with Claude AI...")

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
                print(f"Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"API rate limit exceeded after {max_retries} attempts: {e}")
        except APIConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                print(f"Connection error. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"API connection failed after {max_retries} attempts: {e}")
        except APIError as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"API error. Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                raise RuntimeError(f"API error after {max_retries} attempts: {e}")

    # Extract response text
    response_text = response.content[0].text

    # Parse JSON response
    try:
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        analysis = json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse AI response as JSON: {e}\nResponse: {response_text}")

    # Validate structure
    required_keys = ["title", "summary", "sections"]
    for key in required_keys:
        if key not in analysis:
            raise ValueError(f"Missing required key in AI response: {key}")

    print(f"Analysis complete: {len(analysis['sections'])} sections identified")

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
        print("\nAnalysis Result:")
        print(json.dumps(analysis, indent=2))

        points = get_screenshot_points(analysis)
        print(f"\nScreenshot Points ({len(points)}):")
        for p in points:
            print(f"  - {format_timestamp(p.timestamp)}: {p.reason}")
    except ValueError as e:
        print(f"Error: {e}")
