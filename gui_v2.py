#!/usr/bin/env python3
"""
Video2Doc GUI v2.0 - Modern CustomTkinter Interface
Features: Dark mode, auto-detection, advanced options, chunked processing support
"""

import os
import sys
import threading
import queue
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Import Video2Doc modules (v2 with chunking and hierarchical analysis)
from transcriber_v2 import ChunkedTranscriber, transcribe, get_full_transcript
from analyzer_v2 import HierarchicalAnalyzer, analyze_transcript, get_screenshot_points
from screenshotter import capture_screenshots, get_video_duration
from generators import generate_docx, generate_pptx, generate_markdown
from generators.markdown_gen import generate_markdown_with_transcript
from video_analyzer import VideoAnalyzer, VideoInfo, ProcessingConfig, ProcessingTier
from config import (
    APP_TITLE, APP_VERSION, WHISPER_MODELS, LANGUAGE_CODES,
    QUALITY_PRESETS, SUPPORTED_VIDEO_FORMATS, ProcessingSettings
)


# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class Video2DocApp(ctk.CTk):
    """Main application class for Video2Doc GUI v2.0"""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(APP_TITLE)
        self.geometry("650x750")
        self.minsize(600, 700)

        # Processing state
        self.processing = False
        self.cancel_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.start_time = None

        # Video analyzer
        self.analyzer = VideoAnalyzer()
        self.current_video_info: Optional[VideoInfo] = None
        self.current_config: Optional[ProcessingConfig] = None

        # Settings
        self.settings = ProcessingSettings()

        # Variables
        self.video_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.format_docx = ctk.BooleanVar(value=True)
        self.format_pptx = ctk.BooleanVar(value=True)
        self.format_md = ctk.BooleanVar(value=True)
        self.model_var = ctk.StringVar(value="Auto")
        self.language_var = ctk.StringVar(value="Auto-detect")
        self.include_transcript = ctk.BooleanVar(value=False)
        self.keep_screenshots = ctk.BooleanVar(value=False)

        # Advanced options variables
        self.auto_detect_var = ctk.BooleanVar(value=True)
        self.enable_chunking_var = ctk.BooleanVar(value=True)
        self.use_gpu_var = ctk.BooleanVar(value=True)
        self.enable_checkpoints_var = ctk.BooleanVar(value=True)
        self.quality_preset_var = ctk.StringVar(value="Balanced")
        self.chunk_size_var = ctk.IntVar(value=10)

        self._create_widgets()
        self._check_api_key()
        self._update_gpu_status()

    def _check_api_key(self):
        """Check if API key is set"""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            messagebox.showwarning(
                "API Key Missing",
                "ANTHROPIC_API_KEY environment variable not set.\n\n"
                "Please set it before generating documentation:\n"
                "Windows: $env:ANTHROPIC_API_KEY = 'your-key'\n"
                "Mac/Linux: export ANTHROPIC_API_KEY='your-key'"
            )

    def _update_gpu_status(self):
        """Update GPU availability status"""
        if hasattr(self, 'gpu_status_label'):
            status = "Available" if self.analyzer.gpu_available else "Not Available"
            color = "#4ade80" if self.analyzer.gpu_available else "#f87171"
            self.gpu_status_label.configure(text=f"GPU: {status}", text_color=color)

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main scrollable frame
        self.main_frame = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # === Header ===
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        header_frame.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="Video2Doc",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="AI-Powered Documentation Generator",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle_label.grid(row=1, column=0, sticky="w")

        # GPU status
        self.gpu_status_label = ctk.CTkLabel(
            header_frame,
            text="GPU: Checking...",
            font=ctk.CTkFont(size=12)
        )
        self.gpu_status_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=10)

        # === Input Section ===
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        input_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            input_frame,
            text="Input Video",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))

        ctk.CTkLabel(input_frame, text="File:").grid(row=1, column=0, sticky="w", padx=(15, 5), pady=5)

        self.video_entry = ctk.CTkEntry(
            input_frame,
            textvariable=self.video_path,
            placeholder_text="Select a video file...",
            width=400
        )
        self.video_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        browse_btn = ctk.CTkButton(
            input_frame,
            text="Browse",
            width=80,
            command=self._browse_video
        )
        browse_btn.grid(row=1, column=2, padx=(5, 15), pady=5)

        # Video info display
        self.video_info_frame = ctk.CTkFrame(input_frame, fg_color=("gray90", "gray20"))
        self.video_info_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=15, pady=(5, 15))
        self.video_info_frame.grid_columnconfigure(0, weight=1)

        self.video_info_label = ctk.CTkLabel(
            self.video_info_frame,
            text="No video selected",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.video_info_label.grid(row=0, column=0, sticky="w", padx=10, pady=8)

        self.tier_label = ctk.CTkLabel(
            self.video_info_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.tier_label.grid(row=0, column=1, sticky="e", padx=10, pady=8)

        # === Output Section ===
        output_frame = ctk.CTkFrame(self.main_frame)
        output_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        output_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            output_frame,
            text="Output Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))

        # Output directory
        ctk.CTkLabel(output_frame, text="Folder:").grid(row=1, column=0, sticky="w", padx=(15, 5), pady=5)

        self.output_entry = ctk.CTkEntry(
            output_frame,
            textvariable=self.output_dir,
            placeholder_text="Select output folder...",
            width=400
        )
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        dir_btn = ctk.CTkButton(
            output_frame,
            text="Browse",
            width=80,
            command=self._browse_output
        )
        dir_btn.grid(row=1, column=2, padx=(5, 15), pady=5)

        # Format checkboxes
        format_frame = ctk.CTkFrame(output_frame, fg_color="transparent")
        format_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=15, pady=(5, 15))

        ctk.CTkLabel(format_frame, text="Formats:").pack(side="left", padx=(0, 10))
        ctk.CTkCheckBox(format_frame, text="Word (.docx)", variable=self.format_docx).pack(side="left", padx=10)
        ctk.CTkCheckBox(format_frame, text="PowerPoint (.pptx)", variable=self.format_pptx).pack(side="left", padx=10)
        ctk.CTkCheckBox(format_frame, text="Markdown (.md)", variable=self.format_md).pack(side="left", padx=10)

        # === Options Section ===
        options_frame = ctk.CTkFrame(self.main_frame)
        options_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        options_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            options_frame,
            text="Processing Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))

        # Model selection
        ctk.CTkLabel(options_frame, text="Whisper Model:").grid(row=1, column=0, sticky="w", padx=(15, 5), pady=5)
        model_values = ["Auto"] + list(WHISPER_MODELS.keys())
        self.model_combo = ctk.CTkComboBox(
            options_frame,
            values=model_values,
            variable=self.model_var,
            width=150,
            command=self._on_model_change
        )
        self.model_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Language selection
        ctk.CTkLabel(options_frame, text="Language:").grid(row=2, column=0, sticky="w", padx=(15, 5), pady=5)
        lang_values = list(LANGUAGE_CODES.values())
        self.lang_combo = ctk.CTkComboBox(
            options_frame,
            values=lang_values,
            variable=self.language_var,
            width=150
        )
        self.lang_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Additional checkboxes
        options_extra = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_extra.grid(row=3, column=0, columnspan=2, sticky="ew", padx=15, pady=(10, 15))

        ctk.CTkCheckBox(
            options_extra,
            text="Include full transcript (Markdown)",
            variable=self.include_transcript
        ).pack(side="left", padx=(0, 20))

        ctk.CTkCheckBox(
            options_extra,
            text="Keep screenshots",
            variable=self.keep_screenshots
        ).pack(side="left")

        # === Advanced Options (Collapsible) ===
        self.advanced_frame = ctk.CTkFrame(self.main_frame)
        self.advanced_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.advanced_frame.grid_columnconfigure(0, weight=1)

        # Advanced header with toggle
        advanced_header = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        advanced_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        advanced_header.grid_columnconfigure(1, weight=1)

        self.advanced_toggle = ctk.CTkButton(
            advanced_header,
            text="▼ Advanced Options",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="transparent",
            text_color=("gray20", "gray80"),
            hover_color=("gray80", "gray30"),
            anchor="w",
            command=self._toggle_advanced
        )
        self.advanced_toggle.grid(row=0, column=0, sticky="w")

        # Advanced options content (initially visible)
        self.advanced_content = ctk.CTkFrame(self.advanced_frame, fg_color=("gray90", "gray20"))
        self.advanced_content.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.advanced_content.grid_columnconfigure((0, 1), weight=1)
        self.advanced_visible = True

        # Auto-detect toggle
        ctk.CTkSwitch(
            self.advanced_content,
            text="Auto-detect optimal settings",
            variable=self.auto_detect_var,
            command=self._on_auto_detect_change
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))

        # Quality preset
        ctk.CTkLabel(self.advanced_content, text="Quality Preset:").grid(row=1, column=0, sticky="w", padx=(15, 5), pady=5)
        ctk.CTkComboBox(
            self.advanced_content,
            values=["Speed", "Balanced", "Quality"],
            variable=self.quality_preset_var,
            width=150,
            command=self._on_preset_change
        ).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Chunking options
        self.chunking_switch = ctk.CTkSwitch(
            self.advanced_content,
            text="Enable chunked processing",
            variable=self.enable_chunking_var
        )
        self.chunking_switch.grid(row=2, column=0, sticky="w", padx=15, pady=5)

        chunk_frame = ctk.CTkFrame(self.advanced_content, fg_color="transparent")
        chunk_frame.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ctk.CTkLabel(chunk_frame, text="Chunk size:").pack(side="left")
        self.chunk_slider = ctk.CTkSlider(
            chunk_frame,
            from_=5,
            to=20,
            number_of_steps=15,
            variable=self.chunk_size_var,
            width=100
        )
        self.chunk_slider.pack(side="left", padx=5)
        self.chunk_label = ctk.CTkLabel(chunk_frame, text="10 min")
        self.chunk_label.pack(side="left")
        self.chunk_size_var.trace_add("write", self._update_chunk_label)

        # GPU toggle
        self.gpu_switch = ctk.CTkSwitch(
            self.advanced_content,
            text="Use GPU acceleration",
            variable=self.use_gpu_var
        )
        self.gpu_switch.grid(row=3, column=0, sticky="w", padx=15, pady=5)

        # Checkpoint toggle
        ctk.CTkSwitch(
            self.advanced_content,
            text="Enable checkpoints (crash recovery)",
            variable=self.enable_checkpoints_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=15, pady=(5, 15))

        # === Progress Section ===
        progress_frame = ctk.CTkFrame(self.main_frame)
        progress_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        progress_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            progress_frame,
            text="Progress",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=500)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 5))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(
            progress_frame,
            text="Ready to generate documentation",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=2, column=0, sticky="w", padx=15, pady=2)

        self.time_label = ctk.CTkLabel(
            progress_frame,
            text="Elapsed: --:--",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.time_label.grid(row=3, column=0, sticky="w", padx=15, pady=(2, 15))

        # === Control Buttons ===
        button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        button_frame.grid(row=6, column=0, sticky="ew", pady=(0, 10))

        self.generate_btn = ctk.CTkButton(
            button_frame,
            text="Generate Documentation",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self._start_generation
        )
        self.generate_btn.pack(side="left", padx=(0, 10))

        self.cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            height=40,
            fg_color="gray40",
            hover_color="gray30",
            command=self._cancel_generation,
            state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=(0, 10))

        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear",
            height=40,
            fg_color="transparent",
            border_width=2,
            text_color=("gray20", "gray80"),
            command=self._clear_form
        )
        self.clear_btn.pack(side="left")

        # Theme toggle
        self.theme_btn = ctk.CTkButton(
            button_frame,
            text="☀",
            width=40,
            height=40,
            command=self._toggle_theme
        )
        self.theme_btn.pack(side="right")

        # Version label
        version_label = ctk.CTkLabel(
            self.main_frame,
            text=f"Video2Doc v{APP_VERSION} - Powered by Claude AI",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        version_label.grid(row=7, column=0, pady=(10, 0))

    def _toggle_advanced(self):
        """Toggle advanced options visibility"""
        if self.advanced_visible:
            self.advanced_content.grid_remove()
            self.advanced_toggle.configure(text="▶ Advanced Options")
            self.advanced_visible = False
        else:
            self.advanced_content.grid()
            self.advanced_toggle.configure(text="▼ Advanced Options")
            self.advanced_visible = True

    def _toggle_theme(self):
        """Toggle between dark and light mode"""
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.theme_btn.configure(text="☀" if new_mode == "Dark" else "🌙")

    def _update_chunk_label(self, *args):
        """Update chunk size label"""
        self.chunk_label.configure(text=f"{self.chunk_size_var.get()} min")

    def _on_model_change(self, value):
        """Handle model selection change"""
        if value != "Auto":
            self.auto_detect_var.set(False)

    def _on_auto_detect_change(self):
        """Handle auto-detect toggle"""
        if self.auto_detect_var.get():
            self.model_var.set("Auto")
            if self.current_video_info:
                self._update_auto_settings()

    def _on_preset_change(self, value):
        """Handle quality preset change"""
        preset_key = value.lower()
        if preset_key in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[preset_key]
            if not self.auto_detect_var.get():
                self.model_var.set(preset['whisper_model'])
            self.chunk_size_var.set(preset['chunk_size_minutes'])

    def _browse_video(self):
        """Open file dialog to select video"""
        extensions = " ".join([f"*{ext}" for ext in SUPPORTED_VIDEO_FORMATS])
        filetypes = [
            ("Video files", extensions),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        if filepath:
            self.video_path.set(filepath)
            self._analyze_video(filepath)

            # Auto-set output directory
            if not self.output_dir.get():
                self.output_dir.set(str(Path(filepath).parent))

    def _analyze_video(self, filepath: str):
        """Analyze video and update UI with recommendations"""
        try:
            self.video_info_label.configure(text="Analyzing video...", text_color="gray")
            self.tier_label.configure(text="")
            self.update()

            video_info, config = self.analyzer.analyze(filepath)
            self.current_video_info = video_info
            self.current_config = config

            # Update video info display
            size_str = self.analyzer.format_file_size(video_info.file_size_bytes)
            info_text = f"{Path(filepath).name} • {size_str} • {video_info.duration_formatted}"
            self.video_info_label.configure(text=info_text, text_color=("gray20", "gray80"))

            # Update tier badge
            tier_display = self.analyzer.get_tier_display(config.tier)
            tier_colors = {
                ProcessingTier.SMALL: ("#22c55e", "Small"),
                ProcessingTier.MEDIUM: ("#f59e0b", "Medium"),
                ProcessingTier.LARGE: ("#ef4444", "Large")
            }
            color, _ = tier_colors.get(config.tier, ("#6b7280", "Unknown"))
            self.tier_label.configure(text=f"⬤ {tier_display}", text_color=color)

            # Apply auto-detected settings if enabled
            if self.auto_detect_var.get():
                self._update_auto_settings()

        except Exception as e:
            self.video_info_label.configure(text=f"Error: {str(e)[:50]}...", text_color="#ef4444")
            self.tier_label.configure(text="")
            self.current_video_info = None
            self.current_config = None

    def _update_auto_settings(self):
        """Update settings based on auto-detection"""
        if not self.current_config:
            return

        config = self.current_config

        # Update model (display as Auto but use recommended internally)
        self.model_var.set("Auto")

        # Update chunking settings
        self.enable_chunking_var.set(config.chunking_enabled)
        if config.chunking_enabled:
            self.chunk_size_var.set(config.chunk_size_minutes)

        # Update GPU setting
        self.use_gpu_var.set(config.use_gpu)

        # Update checkpoint setting
        self.enable_checkpoints_var.set(config.checkpointing_enabled)

    def _browse_output(self):
        """Open dialog to select output directory"""
        directory = filedialog.askdirectory(title="Select Output Folder")
        if directory:
            self.output_dir.set(directory)

    def _clear_form(self):
        """Reset all form fields"""
        self.video_path.set("")
        self.output_dir.set("")
        self.video_info_label.configure(text="No video selected", text_color="gray")
        self.tier_label.configure(text="")
        self.progress_bar.set(0)
        self.status_label.configure(text="Ready to generate documentation")
        self.time_label.configure(text="Elapsed: --:--")
        self.current_video_info = None
        self.current_config = None

    def _validate_inputs(self):
        """Validate form inputs before processing"""
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file.")
            return False

        if not Path(self.video_path.get()).exists():
            messagebox.showerror("Error", "Selected video file does not exist.")
            return False

        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return False

        if not any([self.format_docx.get(), self.format_pptx.get(), self.format_md.get()]):
            messagebox.showerror("Error", "Please select at least one output format.")
            return False

        if not os.environ.get("ANTHROPIC_API_KEY"):
            messagebox.showerror(
                "Error",
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Please set it and restart the application."
            )
            return False

        return True

    def _get_effective_model(self) -> str:
        """Get the effective whisper model to use"""
        if self.model_var.get() == "Auto" and self.current_config:
            return self.current_config.whisper_model
        elif self.model_var.get() == "Auto":
            return "base"
        return self.model_var.get()

    def _get_effective_language(self) -> Optional[str]:
        """Get the effective language setting"""
        lang = self.language_var.get()
        if lang == "Auto-detect":
            return None
        # Find the language code from display name
        for code, name in LANGUAGE_CODES.items():
            if name == lang:
                return code if code else None
        return None

    def _start_generation(self):
        """Start the documentation generation process"""
        if not self._validate_inputs():
            return

        self.processing = True
        self.cancel_event.clear()
        self.start_time = time.time()

        # Update UI state
        self.generate_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.clear_btn.configure(state="disabled")
        self.progress_bar.set(0)

        # Start processing thread
        thread = threading.Thread(target=self._process_video, daemon=True)
        thread.start()

        # Start progress polling
        self._poll_progress()

    def _cancel_generation(self):
        """Cancel the ongoing generation"""
        self.cancel_event.set()
        self.status_label.configure(text="Cancelling...")

    def _poll_progress(self):
        """Poll progress queue and update UI"""
        try:
            while True:
                msg = self.progress_queue.get_nowait()

                if msg['type'] == 'progress':
                    self.progress_bar.set(msg['value'] / 100)
                    self.status_label.configure(text=msg['status'])
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
            self.time_label.configure(text=f"Elapsed: {mins:02d}:{secs:02d}")

        if self.processing:
            self.after(100, self._poll_progress)

    def _process_video(self):
        """Process video in background thread with v2 chunking and hierarchical analysis"""
        video_path = self.video_path.get()
        output_dir = Path(self.output_dir.get())
        video_name = Path(video_path).stem

        temp_dir = tempfile.mkdtemp(prefix="video2doc_gui_")
        screenshots_dir = Path(temp_dir) / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get effective settings
            model = self._get_effective_model()
            language = self._get_effective_language()
            enable_chunking = self.enable_chunking_var.get()
            use_gpu = self.use_gpu_var.get()
            chunk_size = self.chunk_size_var.get()

            # Determine Claude strategy based on config
            claude_strategy = "auto"
            if self.current_config:
                claude_strategy = self.current_config.claude_strategy

            # Step 1: Transcribe with v2 ChunkedTranscriber (0-30%)
            self.progress_queue.put({
                'type': 'progress',
                'value': 5,
                'status': f'Transcribing with {model} model (GPU: {"Yes" if use_gpu else "No"})...'
            })

            if self.cancel_event.is_set():
                self.progress_queue.put({'type': 'cancelled'})
                return

            # Create progress callback for transcription
            def transcription_progress(current, total, status):
                # Map transcription progress to 5-30% range
                progress = 5 + int((current / max(total, 1)) * 25)
                self.progress_queue.put({
                    'type': 'progress',
                    'value': progress,
                    'status': status
                })

            # Use ChunkedTranscriber for better large file handling
            transcriber = ChunkedTranscriber(
                chunk_size_minutes=chunk_size,
                overlap_seconds=30,
                use_gpu=use_gpu
            )

            segments = transcriber.transcribe(
                video_path=video_path,
                model_size=model,
                language=language,
                progress_callback=transcription_progress,
                enable_chunking=enable_chunking
            )

            self.progress_queue.put({
                'type': 'progress',
                'value': 30,
                'status': f'Transcribed {len(segments)} segments'
            })

            # Step 2: Analyze with Claude using HierarchicalAnalyzer (30-50%)
            if self.cancel_event.is_set():
                self.progress_queue.put({'type': 'cancelled'})
                return

            self.progress_queue.put({
                'type': 'progress',
                'value': 35,
                'status': f'Analyzing transcript ({claude_strategy} strategy)...'
            })

            # Create progress callback for analysis
            def analysis_progress(current, total, status):
                # Map analysis progress to 35-50% range
                progress = 35 + int((current / max(total, 1)) * 15)
                self.progress_queue.put({
                    'type': 'progress',
                    'value': progress,
                    'status': status
                })

            # Use HierarchicalAnalyzer for better long video handling
            ai_analyzer = HierarchicalAnalyzer(chapter_size_minutes=12)
            analysis = ai_analyzer.analyze(
                segments=segments,
                strategy=claude_strategy,
                progress_callback=analysis_progress
            )

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

            if self.format_docx.get():
                self.progress_queue.put({
                    'type': 'progress',
                    'value': 75,
                    'status': 'Generating Word document...'
                })
                docx_path = str(output_dir / f"{video_name}_documentation.docx")
                generate_docx(title, summary, sections, screenshots_map, docx_path)
                generated_files.append(docx_path)

            if self.format_pptx.get():
                self.progress_queue.put({
                    'type': 'progress',
                    'value': 82,
                    'status': 'Generating PowerPoint presentation...'
                })
                pptx_path = str(output_dir / f"{video_name}_documentation.pptx")
                generate_pptx(title, summary, sections, screenshots_map, pptx_path)
                generated_files.append(pptx_path)

            if self.format_md.get():
                self.progress_queue.put({
                    'type': 'progress',
                    'value': 90,
                    'status': 'Generating Markdown document...'
                })
                md_path = str(output_dir / f"{video_name}_documentation.md")
                if self.include_transcript.get():
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

            # Copy screenshots if requested
            if self.keep_screenshots.get() and screenshots_map:
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
            self.progress_queue.put({
                'type': 'error',
                'message': str(e)
            })
        finally:
            # Cleanup temp directory
            if not self.keep_screenshots.get():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    def _on_complete(self, files):
        """Handle successful completion"""
        self.processing = False
        self.generate_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.clear_btn.configure(state="normal")

        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)

        file_list = "\n".join([f"  • {Path(f).name}" for f in files])
        messagebox.showinfo(
            "Success",
            f"Documentation generated successfully!\n\n"
            f"Time: {mins}m {secs}s\n\n"
            f"Generated files:\n{file_list}"
        )

    def _on_error(self, message):
        """Handle error during processing"""
        self.processing = False
        self.generate_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.clear_btn.configure(state="normal")
        self.status_label.configure(text=f"Error: {message[:50]}...")

        messagebox.showerror("Error", f"Generation failed:\n\n{message}")

    def _on_cancelled(self):
        """Handle cancellation"""
        self.processing = False
        self.generate_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.clear_btn.configure(state="normal")
        self.status_label.configure(text="Generation cancelled")
        self.progress_bar.set(0)


def main():
    """Main entry point for GUI"""
    app = Video2DocApp()
    app.mainloop()


if __name__ == "__main__":
    main()
