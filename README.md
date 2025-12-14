# FrameNotes

AI-powered tool that transforms video content into professional documentation with automatic transcription, intelligent screenshot placement, and multiple output formats.

## Features

- **Automatic Transcription**: Uses faster-whisper for accurate speech-to-text with timestamps
- **AI-Powered Analysis**: Claude AI determines optimal screenshot locations based on content
- **Smart Screenshots**: Extracts frames at key moments using FFmpeg
- **Multiple Output Formats**: Generate DOCX, PPTX, or Markdown documents
- **Modular Architecture**: Easy to extend and customize
- **Graphical User Interface**: User-friendly GUI for non-technical users

## Requirements

### System Requirements

- **Python**: 3.9 or higher
- **FFmpeg**: Must be installed and available in PATH
- **Anthropic API Key**: Required for AI analysis

### Installing FFmpeg

**Windows (via Chocolatey):**
```bash
choco install ffmpeg
```

**Windows (via Winget):**
```bash
winget install FFmpeg.FFmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

## Installation

1. Clone or download the project:
```bash
cd framenotes
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your Anthropic API key:
```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "your-api-key-here"

# Windows CMD
set ANTHROPIC_API_KEY=your-api-key-here

# macOS/Linux
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Graphical User Interface (GUI)

Launch the GUI for a visual, point-and-click experience:
```bash
python gui.py
```

The GUI allows you to:
- Browse and select input video files
- Choose output directory
- Select output formats (DOCX, PPTX, Markdown)
- Configure Whisper model and language settings
- View real-time progress with elapsed time
- Cancel processing at any time

### Command Line (CLI)

Generate a Word document from a video:
```bash
python main.py video.mp4
```

### Specify Output Format

```bash
# Word document
python main.py video.mp4 --format docx --output manual.docx

# PowerPoint presentation
python main.py video.mp4 --format pptx --output slides.pptx

# Markdown
python main.py video.mp4 --format markdown --output readme.md

# Generate all formats
python main.py video.mp4 --format all
```

### Additional Options

```bash
# Use a larger Whisper model for better accuracy
python main.py video.mp4 --model medium

# Specify language (auto-detected by default)
python main.py video.mp4 --language en

# Include full transcript in Markdown output
python main.py video.mp4 --format markdown --include-transcript

# Keep screenshot files after generation
python main.py video.mp4 --keep-screenshots

# Verbose output for debugging
python main.py video.mp4 --verbose
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `video` | Path to input video file (required) |
| `-o, --output` | Output file path |
| `-f, --format` | Output format: docx, pptx, markdown, md, all |
| `-m, --model` | Whisper model: tiny, base, small, medium, large-v3 |
| `-l, --language` | Language code (e.g., en, es, fr) |
| `--include-transcript` | Include full transcript (markdown only) |
| `--temp-dir` | Custom temp directory for screenshots |
| `--keep-screenshots` | Keep screenshot files after generation |
| `-v, --verbose` | Enable verbose output |

## Supported Video Formats

- MP4, MKV, AVI, MOV, WMV, FLV, WebM, M4V

## Project Structure

```
framenotes/
├── main.py              # CLI entry point
├── gui.py               # Graphical user interface (v1 - Tkinter)
├── gui_v2.py            # Graphical user interface (v2 - CustomTkinter)
├── transcriber.py       # Audio extraction and transcription (v1)
├── transcriber_v2.py    # Audio with chunking support (v2)
├── analyzer.py          # AI-powered content analysis (v1)
├── analyzer_v2.py       # Hierarchical analysis (v2)
├── screenshotter.py     # Video frame extraction
├── config.py            # Configuration and settings
├── generators/
│   ├── __init__.py
│   ├── docx_gen.py      # Word document generator
│   ├── pptx_gen.py      # PowerPoint generator
│   └── markdown_gen.py  # Markdown generator
├── requirements.txt
└── README.md
```

## Module Overview

### gui.py
Tkinter-based graphical user interface that provides:
- File browser dialogs for video selection
- Visual configuration of all output options
- Real-time progress bar and status updates
- Background processing with cancelation support

### transcriber.py
Handles audio extraction from video files using FFmpeg and speech-to-text transcription using faster-whisper. Produces timestamped transcript segments.

### analyzer.py
Uses Claude AI to analyze the transcript and determine:
- Document title and summary
- Section breakdowns with content
- Optimal timestamps for screenshots
- Reasons for each screenshot

### screenshotter.py
Extracts high-quality frames from video at specified timestamps using FFmpeg. Handles various video formats and resolutions.

### generators/
Contains format-specific document generators:
- **docx_gen.py**: Professional Word documents with embedded images
- **pptx_gen.py**: Presentation slides with proper formatting
- **markdown_gen.py**: Portable Markdown with optional transcript

## How It Works

1. **Video Analysis**: Extract video duration and metadata
2. **Transcription**: Convert speech to text with timestamps
3. **AI Analysis**: Claude analyzes transcript to identify key sections and screenshot points
4. **Screenshot Capture**: Extract frames at optimal timestamps
5. **Document Generation**: Create formatted output in selected format(s)

## Whisper Model Selection

| Model | Speed | Accuracy | VRAM | Best For |
|-------|-------|----------|------|----------|
| tiny | Fastest | Lower | ~1GB | Quick drafts |
| base | Fast | Good | ~1GB | General use (default) |
| small | Medium | Better | ~2GB | Better accuracy |
| medium | Slower | High | ~5GB | Professional use |
| large-v3 | Slowest | Highest | ~10GB | Maximum accuracy |

## Examples

### Tutorial Documentation
```bash
python main.py tutorial.mp4 --format docx --output "Tutorial Guide.docx"
```

### Lecture Notes with Transcript
```bash
python main.py lecture.mp4 --format markdown --include-transcript --output notes.md
```

### Training Presentation
```bash
python main.py training.mp4 --format pptx --model medium --output "Training Slides.pptx"
```

### Complete Documentation Package
```bash
python main.py demo.mp4 --format all --keep-screenshots
```

## Troubleshooting

### FFmpeg not found
Ensure FFmpeg is installed and in your system PATH. Test with:
```bash
ffmpeg -version
```

### CUDA/GPU errors
faster-whisper will automatically fall back to CPU if CUDA is unavailable.

### API key errors
Verify your Anthropic API key is set correctly:
```bash
echo $env:ANTHROPIC_API_KEY  # PowerShell
echo $ANTHROPIC_API_KEY      # Bash
```

### Out of memory
Use a smaller Whisper model (tiny or base) for long videos.

## License

MIT License - Feel free to use and modify for your needs.
