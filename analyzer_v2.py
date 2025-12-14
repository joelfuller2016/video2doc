"""
Analyzer Module v2.0 - Hierarchical AI Analysis for Large Videos
Supports both single-pass and hierarchical (chapter-based) analysis strategies
"""

import os
import json
import time
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError

from transcriber_v2 import TranscriptSegment, format_timestamp


class AnalysisStrategy(Enum):
    """Analysis strategy based on video length"""
    SINGLE_PASS = "single-pass"        # Short videos (<30 min)
    HIERARCHICAL = "hierarchical"      # Medium videos (30-90 min)
    HIERARCHICAL_PARALLEL = "hierarchical-parallel"  # Large videos (>90 min)


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


@dataclass
class Chapter:
    """Represents a chapter of content for hierarchical analysis"""
    index: int
    start_time: float
    end_time: float
    segments: List[TranscriptSegment]
    analysis: Optional[Dict] = None


@dataclass
class ChapterAnalysis:
    """Analysis result for a single chapter"""
    chapter_index: int
    title: str
    summary: str
    key_points: List[str]
    sections: List[Dict]
    screenshot_points: List[Dict]


# Prompt for analyzing individual chapters
CHAPTER_ANALYSIS_PROMPT = """You are an expert technical writer analyzing a CHAPTER of a longer video transcript.

This is Chapter {chapter_num} of {total_chapters}, covering approximately {start_time} to {end_time}.
{context_note}

CHAPTER TRANSCRIPT:
{transcript}

Analyze this chapter and return a JSON response:
{{
    "chapter_title": "A descriptive title for this chapter's content",
    "chapter_summary": "2-3 sentence summary of what this chapter covers",
    "key_points": ["Key point 1", "Key point 2", ...],
    "sections": [
        {{
            "title": "Section title",
            "content": "Clear, professional description of this section",
            "screenshot_timestamp": 125.5,
            "screenshot_reason": "Why this moment needs a screenshot"
        }}
    ]
}}

SCREENSHOT SELECTION RULES:
1. Select timestamps where something VISUAL is being demonstrated
2. Look for: "as you can see", "look at", "here we have", "click on"
3. Capture key UI elements, configurations, results
4. Aim for 1-3 screenshots per chapter
5. Use timestamps relative to ORIGINAL video (provided in transcript)

Return ONLY valid JSON."""


# Prompt for synthesizing chapter analyses into unified document
SYNTHESIS_PROMPT = """You are synthesizing multiple chapter analyses into a unified document structure.

VIDEO INFORMATION:
- Total Duration: {duration}
- Total Chapters: {total_chapters}

CHAPTER ANALYSES:
{chapter_summaries}

Create a unified document structure that:
1. Provides an overarching title and summary
2. Organizes chapters into a coherent flow
3. Identifies any recurring themes
4. Removes redundant content between chapters

Return JSON:
{{
    "title": "Overall document title",
    "summary": "Executive summary of the entire video (3-5 sentences)",
    "table_of_contents": [
        {{"chapter": 1, "title": "Chapter title", "sections": ["Section 1", "Section 2"]}}
    ],
    "key_themes": ["Theme 1", "Theme 2"],
    "total_sections": 12,
    "recommended_reading_order": "sequential or can-skip-around"
}}

Return ONLY valid JSON."""


# Single-pass prompt (for short videos)
SINGLE_PASS_PROMPT = """You are an expert technical writer analyzing a video transcript to create professional documentation.

TRANSCRIPT:
{transcript}

VIDEO DURATION: {duration} seconds

Analyze and return JSON:
{{
    "title": "Document title based on content",
    "summary": "Brief 1-2 sentence summary",
    "sections": [
        {{
            "title": "Section title",
            "content": "Professional description of this section",
            "screenshot_timestamp": 45.5,
            "screenshot_reason": "Why this moment needs a visual"
        }}
    ]
}}

SCREENSHOT RULES:
1. Select timestamps where something VISUAL is demonstrated
2. Look for: "as you can see", "look at", "click on", "here we have"
3. Aim for 1 screenshot per major section (3-8 total)
4. Choose timestamps in MIDDLE of explanations

Return ONLY valid JSON."""


class HierarchicalAnalyzer:
    """
    Enhanced analyzer with hierarchical analysis for large videos.
    Supports single-pass, hierarchical, and parallel strategies.
    """

    # Thresholds for strategy selection (in minutes)
    SINGLE_PASS_MAX_MINUTES = 30
    HIERARCHICAL_MAX_MINUTES = 90

    # Chapter configuration
    DEFAULT_CHAPTER_MINUTES = 12  # ~12 minute chapters
    MIN_CHAPTER_MINUTES = 8
    MAX_CHAPTER_MINUTES = 18

    def __init__(
        self,
        api_key: Optional[str] = None,
        chapter_size_minutes: int = 12,
        max_retries: int = 3
    ):
        """
        Initialize hierarchical analyzer.

        Args:
            api_key: Anthropic API key (uses env var if not provided)
            chapter_size_minutes: Target chapter size for splitting
            max_retries: Max API retry attempts
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.chapter_size_minutes = chapter_size_minutes
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self) -> Anthropic:
        """Lazy-load Anthropic client"""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
                )
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def analyze(
        self,
        segments: List[TranscriptSegment],
        strategy: str = "auto",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict:
        """
        Analyze transcript using appropriate strategy.

        Args:
            segments: List of TranscriptSegment from transcription
            strategy: 'auto', 'single-pass', 'hierarchical', or 'hierarchical-parallel'
            progress_callback: Callback(current_step, total_steps, status)

        Returns:
            Dictionary containing document structure and screenshot points
        """
        if not segments:
            raise ValueError("Cannot analyze empty transcript.")

        # Calculate duration
        duration_seconds = segments[-1].end if segments else 0
        duration_minutes = duration_seconds / 60

        # Determine strategy
        if strategy == "auto":
            if duration_minutes < self.SINGLE_PASS_MAX_MINUTES:
                strategy = "single-pass"
            elif duration_minutes < self.HIERARCHICAL_MAX_MINUTES:
                strategy = "hierarchical"
            else:
                strategy = "hierarchical-parallel"

        print(f"Analysis strategy: {strategy} (duration: {duration_minutes:.1f} min)")

        if strategy == "single-pass":
            return self._analyze_single_pass(segments, progress_callback)
        else:
            return self._analyze_hierarchical(
                segments,
                duration_seconds,
                progress_callback,
                parallel=(strategy == "hierarchical-parallel")
            )

    def _analyze_single_pass(
        self,
        segments: List[TranscriptSegment],
        progress_callback: Optional[Callable]
    ) -> Dict:
        """Analyze short video in single pass"""
        if progress_callback:
            progress_callback(0, 1, "Analyzing transcript...")

        # Build transcript text
        transcript_lines = []
        for seg in segments:
            timestamp = format_timestamp(seg.start)
            transcript_lines.append(f"[{timestamp}] ({seg.start:.1f}s) {seg.text}")
        transcript_text = "\n".join(transcript_lines)

        duration = segments[-1].end if segments else 0

        prompt = SINGLE_PASS_PROMPT.format(
            transcript=transcript_text,
            duration=f"{duration:.1f}"
        )

        response = self._call_api(prompt)
        analysis = self._parse_json_response(response)

        if progress_callback:
            progress_callback(1, 1, f"Analysis complete: {len(analysis.get('sections', []))} sections")

        return analysis

    def _analyze_hierarchical(
        self,
        segments: List[TranscriptSegment],
        duration_seconds: float,
        progress_callback: Optional[Callable],
        parallel: bool = False
    ) -> Dict:
        """Analyze long video using hierarchical chapter-based approach"""

        # Split into chapters
        chapters = self._split_into_chapters(segments, duration_seconds)
        total_chapters = len(chapters)

        print(f"Split into {total_chapters} chapters for analysis")

        # Analyze each chapter
        chapter_analyses = []

        for i, chapter in enumerate(chapters):
            if progress_callback:
                progress_callback(
                    i + 1,
                    total_chapters + 1,  # +1 for synthesis step
                    f"Analyzing chapter {i + 1} of {total_chapters}..."
                )

            analysis = self._analyze_chapter(chapter, total_chapters)
            chapter_analyses.append(analysis)
            chapter.analysis = analysis

        # Synthesize into unified document
        if progress_callback:
            progress_callback(
                total_chapters + 1,
                total_chapters + 1,
                "Synthesizing document structure..."
            )

        final_analysis = self._synthesize_chapters(
            chapter_analyses,
            chapters,
            duration_seconds
        )

        return final_analysis

    def _split_into_chapters(
        self,
        segments: List[TranscriptSegment],
        duration_seconds: float
    ) -> List[Chapter]:
        """Split transcript into chapters based on time"""
        chapters = []
        chapter_duration = self.chapter_size_minutes * 60  # Convert to seconds

        current_chapter_segments = []
        current_chapter_start = 0
        chapter_index = 0

        for segment in segments:
            current_chapter_segments.append(segment)

            # Check if we've exceeded chapter duration
            chapter_elapsed = segment.end - current_chapter_start

            if chapter_elapsed >= chapter_duration:
                # Create chapter
                chapters.append(Chapter(
                    index=chapter_index,
                    start_time=current_chapter_start,
                    end_time=segment.end,
                    segments=current_chapter_segments.copy()
                ))

                # Start new chapter
                chapter_index += 1
                current_chapter_start = segment.end
                current_chapter_segments = []

        # Add final chapter if there are remaining segments
        if current_chapter_segments:
            chapters.append(Chapter(
                index=chapter_index,
                start_time=current_chapter_start,
                end_time=segments[-1].end,
                segments=current_chapter_segments
            ))

        return chapters

    def _analyze_chapter(
        self,
        chapter: Chapter,
        total_chapters: int
    ) -> Dict:
        """Analyze a single chapter"""
        # Build transcript text for this chapter
        transcript_lines = []
        for seg in chapter.segments:
            timestamp = format_timestamp(seg.start)
            transcript_lines.append(f"[{timestamp}] ({seg.start:.1f}s) {seg.text}")
        transcript_text = "\n".join(transcript_lines)

        # Context note for middle chapters
        if chapter.index == 0:
            context_note = "This is the BEGINNING of the video."
        elif chapter.index == total_chapters - 1:
            context_note = "This is the FINAL chapter of the video."
        else:
            context_note = f"This chapter continues from previous content."

        prompt = CHAPTER_ANALYSIS_PROMPT.format(
            chapter_num=chapter.index + 1,
            total_chapters=total_chapters,
            start_time=format_timestamp(chapter.start_time),
            end_time=format_timestamp(chapter.end_time),
            context_note=context_note,
            transcript=transcript_text
        )

        response = self._call_api(prompt)
        analysis = self._parse_json_response(response)

        # Add chapter metadata
        analysis["chapter_index"] = chapter.index
        analysis["start_time"] = chapter.start_time
        analysis["end_time"] = chapter.end_time

        return analysis

    def _synthesize_chapters(
        self,
        chapter_analyses: List[Dict],
        chapters: List[Chapter],
        duration_seconds: float
    ) -> Dict:
        """Synthesize chapter analyses into unified document"""
        # Build chapter summaries for synthesis prompt
        chapter_summaries = []
        all_sections = []

        for i, analysis in enumerate(chapter_analyses):
            chapter_summary = f"""
Chapter {i + 1}: {analysis.get('chapter_title', f'Chapter {i + 1}')}
Time: {format_timestamp(chapters[i].start_time)} - {format_timestamp(chapters[i].end_time)}
Summary: {analysis.get('chapter_summary', 'No summary')}
Key Points: {', '.join(analysis.get('key_points', []))}
Sections: {len(analysis.get('sections', []))}
"""
            chapter_summaries.append(chapter_summary)

            # Collect all sections with chapter prefix
            for section in analysis.get("sections", []):
                section["chapter_index"] = i
                all_sections.append(section)

        prompt = SYNTHESIS_PROMPT.format(
            duration=format_timestamp(duration_seconds),
            total_chapters=len(chapter_analyses),
            chapter_summaries="\n---\n".join(chapter_summaries)
        )

        response = self._call_api(prompt)
        synthesis = self._parse_json_response(response)

        # Build final analysis structure
        final_analysis = {
            "title": synthesis.get("title", "Video Documentation"),
            "summary": synthesis.get("summary", ""),
            "table_of_contents": synthesis.get("table_of_contents", []),
            "key_themes": synthesis.get("key_themes", []),
            "chapters": chapter_analyses,
            "sections": all_sections,
            "metadata": {
                "duration_seconds": duration_seconds,
                "total_chapters": len(chapter_analyses),
                "total_sections": len(all_sections),
                "analysis_strategy": "hierarchical"
            }
        }

        return final_analysis

    def _call_api(self, prompt: str) -> str:
        """Call Claude API with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API rate limit exceeded: {e}")

            except APIConnectionError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Connection error. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API connection failed: {e}")

            except APIError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"API error. Retrying...")
                    time.sleep(2)
                else:
                    raise RuntimeError(f"API error: {e}")

        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from API response"""
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse AI response as JSON: {e}")


def get_screenshot_points(analysis: Dict) -> List[ScreenshotPoint]:
    """Extract screenshot points from analysis (supports both formats)"""
    points = []

    # Handle hierarchical analysis (sections at top level)
    sections = analysis.get("sections", [])

    # Also check chapters for hierarchical
    for chapter in analysis.get("chapters", []):
        sections.extend(chapter.get("sections", []))

    for section in sections:
        timestamp = section.get("screenshot_timestamp")
        if timestamp is not None:
            points.append(ScreenshotPoint(
                timestamp=float(timestamp),
                reason=section.get("screenshot_reason", "Key moment"),
                section_title=section.get("title", "Section"),
                description=section.get("content", "")
            ))

    # Sort by timestamp and remove duplicates
    points.sort(key=lambda p: p.timestamp)

    # Remove near-duplicates (within 5 seconds)
    filtered_points = []
    for point in points:
        if not filtered_points or point.timestamp - filtered_points[-1].timestamp > 5:
            filtered_points.append(point)

    return filtered_points


def get_document_sections(analysis: Dict) -> List[DocumentSection]:
    """Convert analysis to DocumentSection objects"""
    sections = []

    # Handle both single-pass and hierarchical formats
    raw_sections = analysis.get("sections", [])

    for section_data in raw_sections:
        sections.append(DocumentSection(
            title=section_data.get("title", "Section"),
            content=section_data.get("content", ""),
            screenshot_timestamp=section_data.get("screenshot_timestamp"),
            screenshot_reason=section_data.get("screenshot_reason")
        ))

    return sections


# Backward-compatible function
def analyze_transcript(
    segments: List[TranscriptSegment],
    api_key: Optional[str] = None,
    max_retries: int = 3,
    strategy: str = "auto",
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Analyze transcript using Claude AI.
    Backward-compatible wrapper around HierarchicalAnalyzer.

    Args:
        segments: List of TranscriptSegment from transcription
        api_key: Anthropic API key
        max_retries: Maximum retry attempts
        strategy: 'auto', 'single-pass', or 'hierarchical'
        progress_callback: Optional progress callback

    Returns:
        Dictionary containing document structure
    """
    analyzer = HierarchicalAnalyzer(
        api_key=api_key,
        max_retries=max_retries
    )

    return analyzer.analyze(
        segments=segments,
        strategy=strategy,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    from transcriber_v2 import TranscriptSegment

    # Test with mock segments (simulating a longer video)
    test_segments = []
    for i in range(60):  # 60 segments over 30 minutes
        start = i * 30
        end = start + 30
        text = f"This is segment {i + 1} covering topic {(i // 10) + 1}. "
        if i % 10 == 0:
            text += "As you can see on the screen, this is an important visual element."
        test_segments.append(TranscriptSegment(start, end, text))

    def progress(current, total, status):
        print(f"[{current}/{total}] {status}")

    try:
        analyzer = HierarchicalAnalyzer()
        analysis = analyzer.analyze(
            test_segments,
            strategy="auto",
            progress_callback=progress
        )

        print("\nAnalysis Result:")
        print(f"Title: {analysis.get('title')}")
        print(f"Summary: {analysis.get('summary')}")
        print(f"Chapters: {len(analysis.get('chapters', []))}")
        print(f"Sections: {len(analysis.get('sections', []))}")

        points = get_screenshot_points(analysis)
        print(f"\nScreenshot Points ({len(points)}):")
        for p in points:
            print(f"  - {format_timestamp(p.timestamp)}: {p.section_title}")

    except Exception as e:
        print(f"Error: {e}")
