#!/usr/bin/env python3
"""
FrameNotes GUI - Graphical User Interface for FrameNotes
A Tkinter-based interface for converting videos to documentation.
"""

import os
import sys
import threading
import queue
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


@dataclass(frozen=True)
class ProcessingSnapshot:
    """
    Thread-safe snapshot of all processing settings.
    Captured in main thread before starting background processing.
    This prevents race conditions from accessing tkinter variables in background thread.
    """
    video_path: str
    output_dir: str
    whisper_model: str
    language: Optional[str]
    format_docx: bool
    format_pptx: bool
    format_md: bool
    include_transcript: bool
    keep_screenshots: bool

from logger import get_logger, init_logging
from utils.input_validator import (
    VIDEO_PATH_VALIDATOR,
    OUTPUT_DIR_VALIDATOR,
    WHISPER_MODEL_VALIDATOR,
    PathValidator,
    ValidationResult,
)

# Module logger
logger = get_logger(__name__)

# Import FrameNotes modules
from transcriber import transcribe, get_full_transcript
from analyzer import analyze_transcript, get_screenshot_points
from screenshotter import capture_screenshots, get_video_duration
from generators import generate_docx, generate_pptx, generate_markdown
from generators.markdown_gen import generate_markdown_with_transcript


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.
    Removes dangerous characters and path components.
    """
    import re
    # Remove path separators and parent directory references
    filename = filename.replace('/', '_').replace('\\', '_').replace('..', '_')
    # Remove null bytes and other control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
    # Keep only safe characters: alphanumeric, space, dash, underscore, dot
    filename = re.sub(r'[^\w\s\-.]', '_', filename)
    # Collapse multiple underscores
    filename = re.sub(r'_+', '_', filename)
    # Strip leading/trailing whitespace and underscores
    filename = filename.strip(' _')
    # Ensure non-empty result
    return filename if filename else 'video'


class FrameNotesApp:
    """Main application class for FrameNotes GUI"""

    def __init__(self, root):
        logger.info("Initializing FrameNotes GUI application")
        self.root = root
        self.root.title("FrameNotes - AI Documentation Generator")
        self.root.geometry("550x580")
        self.root.minsize(500, 550)

        # Processing state
        self.processing = False
        self.cancel_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.start_time = None
        self._processing_thread = None  # Track thread for graceful shutdown

        # Register graceful shutdown handler for window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Variables
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.format_docx = tk.BooleanVar(value=True)
        self.format_pptx = tk.BooleanVar(value=True)
        self.format_md = tk.BooleanVar(value=True)
        self.model_var = tk.StringVar(value="base")
        self.language_var = tk.StringVar(value="auto")
        self.include_transcript = tk.BooleanVar(value=False)
        self.keep_screenshots = tk.BooleanVar(value=False)

        self._create_widgets()
        self._check_api_key()
        logger.debug("GUI initialization complete")

    def _check_api_key(self):
        """Check if API key is set"""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY environment variable not set")
            messagebox.showwarning(
                "API Key Missing",
                "ANTHROPIC_API_KEY environment variable not set.\n\n"
                "Please set it before generating documentation:\n"
                "Windows: $env:ANTHROPIC_API_KEY = 'your-key'\n"
                "Mac/Linux: export ANTHROPIC_API_KEY='your-key'"
            )
        else:
            logger.debug("API key found in environment")

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Configure style
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))

        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Header ===
        header = ttk.Label(
            main_frame,
            text="FrameNotes - AI Documentation Generator",
            style="Header.TLabel"
        )
        header.pack(pady=(0, 15))

        # === Input Section ===
        input_frame = ttk.LabelFrame(main_frame, text="Input Video", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        input_row = ttk.Frame(input_frame)
        input_row.pack(fill=tk.X)

        self.video_entry = ttk.Entry(input_row, textvariable=self.video_path, width=50)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_btn = ttk.Button(input_row, text="Browse...", command=self._browse_video)
        browse_btn.pack(side=tk.RIGHT)

        self.video_info_label = ttk.Label(input_frame, text="No video selected", style="Status.TLabel")
        self.video_info_label.pack(anchor=tk.W, pady=(5, 0))

        # === Output Section ===
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        # Output directory
        dir_row = ttk.Frame(output_frame)
        dir_row.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(dir_row, text="Output Folder:").pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(dir_row, textvariable=self.output_dir, width=40)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        dir_btn = ttk.Button(dir_row, text="Browse...", command=self._browse_output)
        dir_btn.pack(side=tk.RIGHT)

        # Format checkboxes
        format_row = ttk.Frame(output_frame)
        format_row.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(format_row, text="Output Formats:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(format_row, text="Word (.docx)", variable=self.format_docx).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(format_row, text="PowerPoint (.pptx)", variable=self.format_pptx).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(format_row, text="Markdown (.md)", variable=self.format_md).pack(side=tk.LEFT, padx=5)

        # === Options Section ===
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        # Model and language dropdowns
        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(options_row1, text="Whisper Model:").pack(side=tk.LEFT)
        model_combo = ttk.Combobox(
            options_row1,
            textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large-v3"],
            state="readonly",
            width=12
        )
        model_combo.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(options_row1, text="Language:").pack(side=tk.LEFT)
        lang_combo = ttk.Combobox(
            options_row1,
            textvariable=self.language_var,
            values=["auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
            state="readonly",
            width=8
        )
        lang_combo.pack(side=tk.LEFT, padx=5)

        # Additional options
        options_row2 = ttk.Frame(options_frame)
        options_row2.pack(fill=tk.X)

        ttk.Checkbutton(
            options_row2,
            text="Include full transcript (Markdown only)",
            variable=self.include_transcript
        ).pack(side=tk.LEFT, padx=(0, 20))

        ttk.Checkbutton(
            options_row2,
            text="Keep screenshot files",
            variable=self.keep_screenshots
        ).pack(side=tk.LEFT)

        # === Progress Section ===
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.status_label = ttk.Label(
            progress_frame,
            text="Ready to generate documentation",
            style="Status.TLabel"
        )
        self.status_label.pack(anchor=tk.W)

        self.time_label = ttk.Label(
            progress_frame,
            text="Elapsed: --:--",
            style="Status.TLabel"
        )
        self.time_label.pack(anchor=tk.W)

        # === Control Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.generate_btn = ttk.Button(
            button_frame,
            text="Generate Documentation",
            command=self._start_generation
        )
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel_generation,
            state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self._clear_form
        )
        self.clear_btn.pack(side=tk.LEFT)

        # Version label
        version_label = ttk.Label(
            main_frame,
            text="FrameNotes v1.0 - Powered by Claude AI",
            style="Status.TLabel"
        )
        version_label.pack(side=tk.BOTTOM, pady=(10, 0))

    def _browse_video(self):
        """Open file dialog to select video"""
        filetypes = [
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        if filepath:
            logger.info(f"Video selected: {filepath}")
            self.video_path.set(filepath)
            self._update_video_info(filepath)

            # Auto-set output directory to video's directory
            if not self.output_dir.get():
                self.output_dir.set(str(Path(filepath).parent))

    def _update_video_info(self, filepath):
        """Update video info label - runs FFmpeg probe in background thread to avoid UI freeze"""
        try:
            path = Path(filepath)
            size_mb = path.stat().st_size / (1024 * 1024)

            # Show loading state immediately
            self.video_info_label.config(text=f"{path.name} ({size_mb:.1f} MB, loading...)")

            # Run FFmpeg probe in background thread to avoid blocking main thread
            def get_duration_async():
                try:
                    duration = get_video_duration(filepath)
                    # Schedule UI update on main thread
                    self.root.after(0, lambda: self._update_video_duration(path.name, size_mb, duration))
                except Exception as e:
                    logger.error(f"Error getting video duration: {e}")
                    self.root.after(0, lambda: self.video_info_label.config(
                        text=f"{path.name} ({size_mb:.1f} MB)"
                    ))

            thread = threading.Thread(target=get_duration_async, daemon=True)
            thread.start()
        except Exception as e:
            logger.error(f"Error reading video info: {e}")
            self.video_info_label.config(text=f"Error reading video: {e}")

    def _update_video_duration(self, name, size_mb, duration):
        """Update video info label with duration (called from main thread)"""
        logger.debug(f"Video info: {name}, {size_mb:.1f} MB, {duration:.0f}s")
        self.video_info_label.config(text=f"{name} ({size_mb:.1f} MB, {duration:.0f}s)")

    def _browse_output(self):
        """Open dialog to select output directory"""
        directory = filedialog.askdirectory(title="Select Output Folder")
        if directory:
            self.output_dir.set(directory)

    def _clear_form(self):
        """Reset all form fields"""
        self.video_path.set("")
        self.output_dir.set("")
        self.video_info_label.config(text="No video selected")
        self.progress_bar['value'] = 0
        self.status_label.config(text="Ready to generate documentation")
        self.time_label.config(text="Elapsed: --:--")

    def _validate_inputs(self):
        """Validate form inputs before processing using secure validators."""
        # Validate video path using secure PathValidator
        video_path = self.video_path.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return False

        video_result = VIDEO_PATH_VALIDATOR.validate(video_path)
        if not video_result.is_valid:
            logger.warning(f"Video path validation failed: {video_result.error_message}")
            messagebox.showerror("Error", video_result.error_message)
            return False

        # Validate output directory
        output_dir = self.output_dir.get()
        if not output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return False

        output_result = OUTPUT_DIR_VALIDATOR.validate(output_dir)
        if not output_result.is_valid:
            logger.warning(f"Output directory validation failed: {output_result.error_message}")
            messagebox.showerror("Error", output_result.error_message)
            return False

        # Validate whisper model selection
        model_result = WHISPER_MODEL_VALIDATOR.validate(self.model_var.get())
        if not model_result.is_valid:
            logger.warning(f"Model validation failed: {model_result.error_message}")
            messagebox.showerror("Error", model_result.error_message)
            return False

        # Validate at least one output format selected
        if not any([self.format_docx.get(), self.format_pptx.get(), self.format_md.get()]):
            messagebox.showerror("Error", "Please select at least one output format.")
            return False

        # Validate API key is set
        if not os.environ.get("ANTHROPIC_API_KEY"):
            messagebox.showerror(
                "Error",
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Please set it and restart the application."
            )
            return False

        return True

    def _start_generation(self):
        """Start the documentation generation process"""
        # Guard against concurrent processing - prevent duplicate generation attempts
        if self.processing:
            logger.warning("Generation already in progress, ignoring duplicate request")
            return

        if not self._validate_inputs():
            return

        logger.info("Starting documentation generation")
        logger.info(f"  Video: {self.video_path.get()}")
        logger.info(f"  Output: {self.output_dir.get()}")
        logger.info(f"  Formats: DOCX={self.format_docx.get()}, PPTX={self.format_pptx.get()}, MD={self.format_md.get()}")
        logger.info(f"  Model: {self.model_var.get()}, Language: {self.language_var.get()}")

        self.processing = True
        self.cancel_event.clear()
        self.start_time = time.time()

        # Update UI state
        self.generate_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0

        # THREAD SAFETY: Snapshot all settings in main thread before background processing
        # This prevents race conditions from accessing tkinter variables in background thread
        language_val = self.language_var.get()
        snapshot = ProcessingSnapshot(
            video_path=self.video_path.get(),
            output_dir=self.output_dir.get(),
            whisper_model=self.model_var.get(),
            language=None if language_val == "auto" else language_val,
            format_docx=self.format_docx.get(),
            format_pptx=self.format_pptx.get(),
            format_md=self.format_md.get(),
            include_transcript=self.include_transcript.get(),
            keep_screenshots=self.keep_screenshots.get()
        )

        # Start processing thread with snapshot
        # Store thread reference for graceful shutdown
        self._processing_thread = threading.Thread(target=self._process_video, args=(snapshot,), daemon=True)
        self._processing_thread.start()

        # Start progress polling
        self._poll_progress()

    def _cancel_generation(self):
        """Cancel the ongoing generation"""
        logger.warning("User requested cancellation")
        self.cancel_event.set()
        self.status_label.config(text="Cancelling...")

    def _on_window_close(self):
        """Handle graceful shutdown when window is closed.

        Ensures background threads complete or timeout gracefully to prevent:
        - Orphaned temp files
        - Corrupted output files from mid-write interruption
        - Resource leaks
        """
        if self.processing and self._processing_thread and self._processing_thread.is_alive():
            logger.info("Window close requested during processing - initiating graceful shutdown")

            # Signal thread to stop
            self.cancel_event.set()

            # Wait for thread to finish with timeout
            shutdown_timeout = 5.0  # seconds
            self._processing_thread.join(timeout=shutdown_timeout)

            if self._processing_thread.is_alive():
                logger.warning(f"Processing thread did not stop within {shutdown_timeout}s timeout")
            else:
                logger.info("Processing thread stopped gracefully")
        else:
            logger.debug("No active processing - closing immediately")

        # Destroy the window
        self.root.destroy()

    def _poll_progress(self):
        """Poll progress queue and update UI"""
        try:
            while True:
                msg = self.progress_queue.get_nowait()

                if msg['type'] == 'progress':
                    self.progress_bar['value'] = msg['value']
                    self.status_label.config(text=msg['status'])
                elif msg['type'] == 'complete':
                    self._on_complete(msg['files'])
                    return
                elif msg['type'] == 'error':
                    self._on_error(msg['message'])
                    return
                elif msg['type'] == 'cancelled':
                    self._on_cancelled()
                    return
        except queue.Empty:
            pass

        # Update elapsed time
        if self.processing and self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.time_label.config(text=f"Elapsed: {mins:02d}:{secs:02d}")

        if self.processing:
            self.root.after(100, self._poll_progress)

    def _process_video(self, snapshot: ProcessingSnapshot):
        """Process video in background thread.

        Args:
            snapshot: Thread-safe snapshot of all settings captured in main thread.
                     Never access tkinter variables (self.xxx.get()) from this method.
        """
        video_path = snapshot.video_path
        output_dir = Path(snapshot.output_dir)
        video_name = sanitize_filename(Path(video_path).stem)

        logger.debug(f"Processing video in background thread: {video_name}")
        temp_dir = tempfile.mkdtemp(prefix="framenotes_gui_")
        screenshots_dir = Path(temp_dir) / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using temp directory: {temp_dir}")

        try:
            # Step 1: Transcribe (0-30%)
            self.progress_queue.put({
                'type': 'progress',
                'value': 5,
                'status': 'Extracting audio and transcribing...'
            })

            if self.cancel_event.is_set():
                self.progress_queue.put({'type': 'cancelled'})
                return

            # Use settings from snapshot (thread-safe)
            segments = transcribe(video_path, model_size=snapshot.whisper_model, language=snapshot.language)

            self.progress_queue.put({
                'type': 'progress',
                'value': 30,
                'status': f'Transcribed {len(segments)} segments'
            })

            # Step 2: Analyze with Claude (30-50%)
            if self.cancel_event.is_set():
                self.progress_queue.put({'type': 'cancelled'})
                return

            self.progress_queue.put({
                'type': 'progress',
                'value': 35,
                'status': 'Analyzing transcript with Claude AI...'
            })

            analysis = analyze_transcript(segments)

            self.progress_queue.put({
                'type': 'progress',
                'value': 50,
                'status': f'Analysis complete: {len(analysis.get("sections", []))} sections'
            })

            # Step 3: Capture screenshots (50-70%)
            if self.cancel_event.is_set():
                self.progress_queue.put({'type': 'cancelled'})
                return

            self.progress_queue.put({
                'type': 'progress',
                'value': 55,
                'status': 'Capturing screenshots...'
            })

            screenshot_points = get_screenshot_points(analysis)
            timestamps = [p.timestamp for p in screenshot_points]

            screenshots_map = {}
            if timestamps:
                screenshot_list = capture_screenshots(
                    video_path, timestamps, str(screenshots_dir),
                    prefix="frame", format="png"
                )
                for ss in screenshot_list:
                    screenshots_map[ss.timestamp] = ss.filepath

            self.progress_queue.put({
                'type': 'progress',
                'value': 70,
                'status': f'Captured {len(screenshots_map)} screenshots'
            })

            # Step 4: Generate documents (70-95%)
            if self.cancel_event.is_set():
                self.progress_queue.put({'type': 'cancelled'})
                return

            title = analysis.get("title", video_name)
            summary = analysis.get("summary", "")
            sections = analysis.get("sections", [])
            generated_files = []

            # Use snapshot values for thread safety
            if snapshot.format_docx:
                self.progress_queue.put({
                    'type': 'progress',
                    'value': 75,
                    'status': 'Generating Word document...'
                })
                docx_path = str(output_dir / f"{video_name}_documentation.docx")
                generate_docx(title, summary, sections, screenshots_map, docx_path)
                generated_files.append(docx_path)

            if snapshot.format_pptx:
                self.progress_queue.put({
                    'type': 'progress',
                    'value': 82,
                    'status': 'Generating PowerPoint presentation...'
                })
                pptx_path = str(output_dir / f"{video_name}_documentation.pptx")
                generate_pptx(title, summary, sections, screenshots_map, pptx_path)
                generated_files.append(pptx_path)

            if snapshot.format_md:
                self.progress_queue.put({
                    'type': 'progress',
                    'value': 90,
                    'status': 'Generating Markdown document...'
                })
                md_path = str(output_dir / f"{video_name}_documentation.md")
                if snapshot.include_transcript:
                    generate_markdown_with_transcript(
                        title, summary, sections, screenshots_map,
                        segments, md_path, copy_images=True
                    )
                else:
                    generate_markdown(
                        title, summary, sections, screenshots_map,
                        md_path, copy_images=True
                    )
                generated_files.append(md_path)

            # Copy screenshots if requested (use snapshot for thread safety)
            if snapshot.keep_screenshots and screenshots_map:
                ss_output_dir = output_dir / f"{video_name}_screenshots"
                ss_output_dir.mkdir(exist_ok=True)
                for ss_path in screenshots_map.values():
                    shutil.copy2(ss_path, ss_output_dir)
                generated_files.append(str(ss_output_dir))

            self.progress_queue.put({
                'type': 'progress',
                'value': 100,
                'status': 'Complete!'
            })

            self.progress_queue.put({
                'type': 'complete',
                'files': generated_files
            })

        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            self.progress_queue.put({
                'type': 'error',
                'message': str(e)
            })
        finally:
            # Cleanup temp directory (use snapshot for thread safety)
            if not snapshot.keep_screenshots:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp directory: {cleanup_error}")

    def _on_complete(self, files):
        """Handle successful completion"""
        self.processing = False
        self.generate_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.NORMAL)

        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)

        logger.info(f"Documentation generation completed in {mins}m {secs}s")
        for f in files:
            logger.info(f"  Generated: {f}")

        file_list = "\n".join([f"  - {f}" for f in files])
        messagebox.showinfo(
            "Success",
            f"Documentation generated successfully!\n\n"
            f"Time: {mins}m {secs}s\n\n"
            f"Generated files:\n{file_list}"
        )

    def _on_error(self, message):
        """Handle error during processing"""
        self.processing = False
        self.generate_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Error: {message[:50]}...")

        logger.error(f"Generation failed: {message}")
        messagebox.showerror("Error", f"Generation failed:\n\n{message}")

    def _on_cancelled(self):
        """Handle cancellation"""
        self.processing = False
        self.generate_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Generation cancelled")
        self.progress_bar['value'] = 0
        logger.info("Generation cancelled by user")


def main():
    """Main entry point for GUI"""
    # Initialize logging for GUI mode
    init_logging(verbose=False)
    logger.info("Starting FrameNotes GUI application")

    root = tk.Tk()

    # Set icon if available
    try:
        # On Windows, try to use a default icon
        if sys.platform == 'win32':
            root.iconbitmap(default='')
    except Exception:
        pass

    app = FrameNotesApp(root)
    logger.debug("Entering main event loop")
    root.mainloop()
    logger.info("FrameNotes GUI application closed")


if __name__ == "__main__":
    main()
