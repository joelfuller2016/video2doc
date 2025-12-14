#!/usr/bin/env python3
"""
FrameNotes GUI v2.0 - Modern CustomTkinter Interface
Features: Dark mode, auto-detection, advanced options, chunked processing support
Settings: Tabbed settings window with tooltips and help documentation
"""

import os
import sys
import threading
import queue
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Import FrameNotes modules (v2 with chunking and hierarchical analysis)
from transcriber_v2 import ChunkedTranscriber, transcribe, get_full_transcript
from analyzer_v2 import HierarchicalAnalyzer, analyze_transcript, get_screenshot_points
from screenshotter import capture_screenshots, get_video_duration
from generators import generate_docx, generate_pptx, generate_markdown
from generators.markdown_gen import generate_markdown_with_transcript
from video_analyzer import VideoAnalyzer, VideoInfo, ProcessingConfig, ProcessingTier
from config import (
    APP_TITLE, APP_VERSION, WHISPER_MODELS, LANGUAGE_CODES,
    QUALITY_PRESETS, SUPPORTED_VIDEO_FORMATS, ProcessingSettings,
    CLAUDE_MODELS, SETTING_TOOLTIPS, get_settings_manager, SettingsManager
)


# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


@dataclass
class ProcessingSnapshot:
    """
    Thread-safe snapshot of all processing settings.
    Captured in main thread before starting background processing.
    """
    video_path: str
    output_dir: str
    whisper_model: str
    language: Optional[str]
    enable_chunking: bool
    use_gpu: bool
    chunk_size: int
    claude_strategy: str
    claude_model: str
    super_detailed: bool
    api_key: str
    format_docx: bool
    format_pptx: bool
    format_md: bool
    include_transcript: bool
    keep_screenshots: bool


class CTkToolTip:
    """
    Simple tooltip implementation for CustomTkinter widgets.
    Shows tooltip text when hovering over a widget.
    """

    def __init__(self, widget, text: str, delay: int = 500):
        """
        Create a tooltip for a widget.

        Args:
            widget: The widget to attach the tooltip to
            text: The tooltip text to display
            delay: Delay in ms before showing tooltip
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.schedule_id = None

        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, event=None):
        """Schedule tooltip display"""
        self.schedule_id = self.widget.after(self.delay, self._show_tooltip)

    def _on_leave(self, event=None):
        """Hide tooltip and cancel any pending display"""
        if self.schedule_id:
            self.widget.after_cancel(self.schedule_id)
            self.schedule_id = None
        self._hide_tooltip()

    def _show_tooltip(self):
        """Display the tooltip"""
        if self.tooltip_window:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip_window = ctk.CTkToplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Make tooltip stay on top
        self.tooltip_window.attributes("-topmost", True)

        label = ctk.CTkLabel(
            self.tooltip_window,
            text=self.text,
            corner_radius=6,
            fg_color=("gray85", "gray25"),
            text_color=("gray10", "gray90"),
            padx=10,
            pady=5,
            wraplength=300
        )
        label.pack()

    def _hide_tooltip(self):
        """Hide the tooltip"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class SettingsWindow(ctk.CTkToplevel):
    """
    Settings window with tabbed interface.
    Tabs: General, API, Processing, Output, Help
    """

    def __init__(self, parent, settings_manager: SettingsManager, on_save: Callable = None):
        super().__init__(parent)

        self.settings_manager = settings_manager
        self.on_save_callback = on_save

        # Window configuration
        self.title("Settings")
        self.geometry("700x600")
        self.minsize(600, 500)
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 700) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 600) // 2
        self.geometry(f"+{x}+{y}")

        # Create UI
        self._create_widgets()
        self._load_settings()

    def _create_widgets(self):
        """Create settings UI with tabs"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Tabview
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        # Create tabs
        self.tab_general = self.tabview.add("General")
        self.tab_api = self.tabview.add("API")
        self.tab_processing = self.tabview.add("Processing")
        self.tab_output = self.tabview.add("Output")
        self.tab_help = self.tabview.add("Help")

        # Configure tab grids
        for tab in [self.tab_general, self.tab_api, self.tab_processing, self.tab_output]:
            tab.grid_columnconfigure(0, weight=1)

        # Build each tab
        self._build_general_tab()
        self._build_api_tab()
        self._build_processing_tab()
        self._build_output_tab()
        self._build_help_tab()

        # Buttons at bottom
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.grid(row=1, column=0, sticky="ew", pady=(20, 0))

        ctk.CTkButton(
            button_frame,
            text="Save",
            width=100,
            command=self._save_settings
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            width=100,
            fg_color="gray40",
            hover_color="gray30",
            command=self.destroy
        ).pack(side="right")

        ctk.CTkButton(
            button_frame,
            text="Reset to Defaults",
            width=120,
            fg_color="transparent",
            border_width=2,
            text_color=("gray20", "gray80"),
            command=self._reset_defaults
        ).pack(side="left")

    def _build_general_tab(self):
        """Build General settings tab"""
        tab = self.tab_general

        # Theme section
        section = ctk.CTkFrame(tab)
        section.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        section.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            section,
            text="Appearance",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))

        # Theme dropdown
        ctk.CTkLabel(section, text="Theme:").grid(row=1, column=0, sticky="w", padx=(15, 10), pady=5)
        self.theme_var = ctk.StringVar(value="Dark")
        theme_combo = ctk.CTkComboBox(
            section,
            values=["Dark", "Light", "System"],
            variable=self.theme_var,
            width=150
        )
        theme_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        CTkToolTip(theme_combo, SETTING_TOOLTIPS["theme"])

        # Language section
        section2 = ctk.CTkFrame(tab)
        section2.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        section2.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            section2,
            text="Language",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))

        ctk.CTkLabel(section2, text="Transcription Language:").grid(row=1, column=0, sticky="w", padx=(15, 10), pady=5)
        self.language_var = ctk.StringVar(value="Auto-detect")
        lang_combo = ctk.CTkComboBox(
            section2,
            values=list(LANGUAGE_CODES.values()),
            variable=self.language_var,
            width=150
        )
        lang_combo.grid(row=1, column=1, sticky="w", padx=5, pady=(5, 15))
        CTkToolTip(lang_combo, SETTING_TOOLTIPS["language"])

    def _build_api_tab(self):
        """Build API settings tab"""
        tab = self.tab_api

        section = ctk.CTkFrame(tab)
        section.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        section.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            section,
            text="Claude API Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))

        # API Key
        ctk.CTkLabel(section, text="API Key:").grid(row=1, column=0, sticky="w", padx=(15, 10), pady=5)
        self.api_key_var = ctk.StringVar()
        self.api_key_entry = ctk.CTkEntry(
            section,
            textvariable=self.api_key_var,
            width=350,
            show="•",
            placeholder_text="Enter your Anthropic API key..."
        )
        self.api_key_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        CTkToolTip(self.api_key_entry, SETTING_TOOLTIPS["api_key"])

        self.show_key_var = ctk.BooleanVar(value=False)
        show_btn = ctk.CTkCheckBox(
            section,
            text="Show",
            variable=self.show_key_var,
            width=60,
            command=self._toggle_api_key_visibility
        )
        show_btn.grid(row=1, column=2, padx=(5, 15), pady=5)

        # API Key info
        info_label = ctk.CTkLabel(
            section,
            text="Get your API key at console.anthropic.com",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        info_label.grid(row=2, column=1, sticky="w", padx=5, pady=(0, 10))

        # Claude Model selection
        ctk.CTkLabel(section, text="Claude Model:").grid(row=3, column=0, sticky="w", padx=(15, 10), pady=5)
        self.claude_model_var = ctk.StringVar(value="claude-sonnet-4-20250514")
        model_values = list(CLAUDE_MODELS.keys())
        model_combo = ctk.CTkComboBox(
            section,
            values=model_values,
            variable=self.claude_model_var,
            width=250
        )
        model_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        CTkToolTip(model_combo, SETTING_TOOLTIPS["claude_model"])

        # Model description
        self.model_desc_label = ctk.CTkLabel(
            section,
            text=CLAUDE_MODELS["claude-sonnet-4-20250514"]["description"],
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.model_desc_label.grid(row=4, column=1, sticky="w", padx=5, pady=(0, 15))

        # Update description on model change
        def update_model_desc(choice):
            if choice in CLAUDE_MODELS:
                self.model_desc_label.configure(text=CLAUDE_MODELS[choice]["description"])
        model_combo.configure(command=update_model_desc)

        # Test connection button
        test_btn = ctk.CTkButton(
            section,
            text="Test Connection",
            width=120,
            command=self._test_api_connection
        )
        test_btn.grid(row=5, column=1, sticky="w", padx=5, pady=(5, 15))

    def _build_processing_tab(self):
        """Build Processing settings tab"""
        tab = self.tab_processing

        # Whisper section
        section = ctk.CTkFrame(tab)
        section.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        section.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            section,
            text="Transcription Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))

        # Whisper model
        ctk.CTkLabel(section, text="Whisper Model:").grid(row=1, column=0, sticky="w", padx=(15, 10), pady=5)
        self.whisper_model_var = ctk.StringVar(value="base")
        whisper_combo = ctk.CTkComboBox(
            section,
            values=list(WHISPER_MODELS.keys()),
            variable=self.whisper_model_var,
            width=150
        )
        whisper_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        CTkToolTip(whisper_combo, SETTING_TOOLTIPS["whisper_model"])

        # Quality preset
        ctk.CTkLabel(section, text="Quality Preset:").grid(row=2, column=0, sticky="w", padx=(15, 10), pady=5)
        self.quality_preset_var = ctk.StringVar(value="balanced")
        preset_combo = ctk.CTkComboBox(
            section,
            values=["speed", "balanced", "quality"],
            variable=self.quality_preset_var,
            width=150,
            command=self._on_preset_change
        )
        preset_combo.grid(row=2, column=1, sticky="w", padx=5, pady=(5, 15))
        CTkToolTip(preset_combo, SETTING_TOOLTIPS["quality_preset"])

        # GPU and Chunking section
        section2 = ctk.CTkFrame(tab)
        section2.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        section2.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            section2,
            text="Performance Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))

        # GPU toggle
        self.use_gpu_var = ctk.BooleanVar(value=True)
        gpu_switch = ctk.CTkSwitch(
            section2,
            text="Use GPU acceleration",
            variable=self.use_gpu_var
        )
        gpu_switch.grid(row=1, column=0, columnspan=2, sticky="w", padx=15, pady=5)
        CTkToolTip(gpu_switch, SETTING_TOOLTIPS["use_gpu"])

        # Chunking toggle
        self.enable_chunking_var = ctk.BooleanVar(value=True)
        chunk_switch = ctk.CTkSwitch(
            section2,
            text="Enable chunked processing",
            variable=self.enable_chunking_var
        )
        chunk_switch.grid(row=2, column=0, columnspan=2, sticky="w", padx=15, pady=5)
        CTkToolTip(chunk_switch, SETTING_TOOLTIPS["enable_chunking"])

        # Chunk size slider
        chunk_frame = ctk.CTkFrame(section2, fg_color="transparent")
        chunk_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=15, pady=(5, 15))

        ctk.CTkLabel(chunk_frame, text="Chunk size:").pack(side="left")
        self.chunk_size_var = ctk.IntVar(value=10)
        chunk_slider = ctk.CTkSlider(
            chunk_frame,
            from_=5,
            to=20,
            number_of_steps=15,
            variable=self.chunk_size_var,
            width=150
        )
        chunk_slider.pack(side="left", padx=10)
        self.chunk_label = ctk.CTkLabel(chunk_frame, text="10 min")
        self.chunk_label.pack(side="left")
        CTkToolTip(chunk_slider, SETTING_TOOLTIPS["chunk_size_minutes"])

        # Update label when slider changes
        def update_chunk_label(*args):
            self.chunk_label.configure(text=f"{self.chunk_size_var.get()} min")
        self.chunk_size_var.trace_add("write", update_chunk_label)

        # Claude strategy
        section3 = ctk.CTkFrame(tab)
        section3.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        section3.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            section3,
            text="Analysis Strategy",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 10))

        ctk.CTkLabel(section3, text="Strategy:").grid(row=1, column=0, sticky="w", padx=(15, 10), pady=5)
        self.claude_strategy_var = ctk.StringVar(value="auto")
        strategy_combo = ctk.CTkComboBox(
            section3,
            values=["auto", "single-pass", "hierarchical"],
            variable=self.claude_strategy_var,
            width=150
        )
        strategy_combo.grid(row=1, column=1, sticky="w", padx=5, pady=(5, 15))
        CTkToolTip(strategy_combo, SETTING_TOOLTIPS["claude_strategy"])

    def _build_output_tab(self):
        """Build Output settings tab"""
        tab = self.tab_output

        # Output formats section
        section = ctk.CTkFrame(tab)
        section.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            section,
            text="Output Formats",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        formats_frame = ctk.CTkFrame(section, fg_color="transparent")
        formats_frame.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 15))

        self.format_docx_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(formats_frame, text="Word (.docx)", variable=self.format_docx_var).pack(side="left", padx=(0, 20))

        self.format_pptx_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(formats_frame, text="PowerPoint (.pptx)", variable=self.format_pptx_var).pack(side="left", padx=(0, 20))

        self.format_md_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(formats_frame, text="Markdown (.md)", variable=self.format_md_var).pack(side="left")

        CTkToolTip(formats_frame, SETTING_TOOLTIPS["output_formats"])

        # Additional output options
        section2 = ctk.CTkFrame(tab)
        section2.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        section2.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            section2,
            text="Additional Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Include transcript
        self.include_transcript_var = ctk.BooleanVar(value=False)
        transcript_cb = ctk.CTkCheckBox(
            section2,
            text="Include full transcript in Markdown output",
            variable=self.include_transcript_var
        )
        transcript_cb.grid(row=1, column=0, sticky="w", padx=15, pady=5)
        CTkToolTip(transcript_cb, SETTING_TOOLTIPS["include_transcript"])

        # Keep screenshots
        self.keep_screenshots_var = ctk.BooleanVar(value=False)
        screenshots_cb = ctk.CTkCheckBox(
            section2,
            text="Save screenshots to separate folder",
            variable=self.keep_screenshots_var
        )
        screenshots_cb.grid(row=2, column=0, sticky="w", padx=15, pady=5)
        CTkToolTip(screenshots_cb, SETTING_TOOLTIPS["keep_screenshots"])

        # Super detailed output
        self.super_detailed_var = ctk.BooleanVar(value=False)
        detailed_cb = ctk.CTkCheckBox(
            section2,
            text="Super Detailed Output",
            variable=self.super_detailed_var
        )
        detailed_cb.grid(row=3, column=0, sticky="w", padx=15, pady=5)
        CTkToolTip(detailed_cb, SETTING_TOOLTIPS["super_detailed_output"])

        # Description for super detailed
        detailed_desc = ctk.CTkLabel(
            section2,
            text="When enabled, generates comprehensive documentation with:\n"
                 "• Detailed chapter breakdowns with exact timestamps\n"
                 "• Complete technical terminology explanations\n"
                 "• Step-by-step procedure descriptions\n"
                 "• All mentioned tools, settings, and parameters\n"
                 "• Extended summaries for each section",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            justify="left"
        )
        detailed_desc.grid(row=4, column=0, sticky="w", padx=35, pady=(0, 15))

    def _build_help_tab(self):
        """Build Help tab with full documentation"""
        tab = self.tab_help
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Scrollable text box
        help_text = ctk.CTkTextbox(tab, wrap="word", font=ctk.CTkFont(size=13))
        help_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Insert documentation
        documentation = """
FRAMENOTES - AI-Powered Video Documentation
================================================

FrameNotes automatically converts video tutorials and presentations into
professional documentation using AI transcription and analysis.


QUICK START
-----------
1. Select a video file using the Browse button
2. Choose an output folder for the generated documents
3. Select your desired output formats (Word, PowerPoint, Markdown)
4. Click "Generate Documentation" and wait for processing

The app will automatically:
• Extract and transcribe audio from the video
• Analyze content using Claude AI
• Capture relevant screenshots
• Generate formatted documentation


SETTINGS OVERVIEW
-----------------

General Tab:
• Theme: Choose between Dark, Light, or System theme
• Language: Set the transcription language (Auto-detect recommended)

API Tab:
• API Key: Your Anthropic API key for Claude AI
  Get one at: https://console.anthropic.com
• Claude Model: Select which Claude model to use for analysis

Processing Tab:
• Whisper Model: Transcription accuracy vs speed tradeoff
  - tiny: Fastest, lower accuracy
  - base: Good balance (recommended)
  - small: Better accuracy
  - medium: High accuracy
  - large-v3: Best accuracy, slowest

• Quality Preset: Quick settings adjustment
  - Speed: Fast processing, lower quality
  - Balanced: Recommended for most videos
  - Quality: Best output, slower processing

• GPU Acceleration: Use your GPU for faster transcription
• Chunked Processing: Process long videos in segments

Output Tab:
• Output Formats: Choose Word, PowerPoint, and/or Markdown
• Include Transcript: Add full transcript to Markdown output
• Keep Screenshots: Save extracted screenshots separately
• Super Detailed Output: Generate comprehensive documentation


VIDEO SIZE TIERS
----------------
The app automatically detects video complexity:

🟢 Small (< 500MB, < 30 min)
   Simple processing, fast results

🟡 Medium (500MB - 2GB, 30-90 min)
   Chunked processing recommended

🔴 Large (> 2GB, > 90 min)
   Full chunking with checkpoints


KEYBOARD SHORTCUTS
------------------
• Ctrl+O: Open video file
• Ctrl+S: Open settings
• Escape: Cancel current operation


TROUBLESHOOTING
---------------
"API Key Missing" error:
→ Enter your API key in Settings > API tab

Slow transcription:
→ Enable GPU acceleration in Settings > Processing
→ Use a smaller Whisper model

Out of memory:
→ Enable chunked processing
→ Reduce chunk size to 5 minutes
→ Use a smaller Whisper model

Poor transcription quality:
→ Use a larger Whisper model
→ Set the correct language instead of auto-detect


SUPPORT
-------
Report issues: https://github.com/your-repo/framenotes/issues
Documentation: https://github.com/your-repo/framenotes/wiki

Version: """ + APP_VERSION + """
Powered by Whisper AI and Claude AI
"""

        help_text.insert("1.0", documentation)
        help_text.configure(state="disabled")  # Make read-only

    def _toggle_api_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key_var.get():
            self.api_key_entry.configure(show="")
        else:
            self.api_key_entry.configure(show="•")

    def _test_api_connection(self):
        """Test the API connection"""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key first.")
            return

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Make a minimal API call
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            messagebox.showinfo("Success", "API connection successful!")
        except Exception as e:
            messagebox.showerror("Error", f"API connection failed:\n{str(e)}")

    def _on_preset_change(self, value):
        """Update settings based on quality preset"""
        if value in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[value]
            self.whisper_model_var.set(preset["whisper_model"])
            self.chunk_size_var.set(preset["chunk_size_minutes"])
            self.claude_strategy_var.set(preset["claude_strategy"])

    def _load_settings(self):
        """Load current settings into UI"""
        app = self.settings_manager.app_settings
        proc = self.settings_manager.processing_settings

        # General
        self.theme_var.set(app.theme.capitalize())
        lang_name = LANGUAGE_CODES.get(proc.language, "Auto-detect")
        self.language_var.set(lang_name)

        # API
        self.api_key_var.set(app.api_key)
        self.claude_model_var.set(proc.claude_model)

        # Processing
        self.whisper_model_var.set(proc.whisper_model)
        self.quality_preset_var.set(proc.quality_preset)
        self.use_gpu_var.set(proc.use_gpu)
        self.enable_chunking_var.set(proc.enable_chunking)
        self.chunk_size_var.set(proc.chunk_size_minutes)
        self.claude_strategy_var.set(proc.claude_strategy)

        # Output
        self.format_docx_var.set("docx" in proc.output_formats)
        self.format_pptx_var.set("pptx" in proc.output_formats)
        self.format_md_var.set("md" in proc.output_formats)
        self.include_transcript_var.set(proc.include_transcript)
        self.keep_screenshots_var.set(proc.keep_screenshots)
        self.super_detailed_var.set(proc.super_detailed_output)

    def _save_settings(self):
        """Save settings from UI"""
        app = self.settings_manager.app_settings
        proc = self.settings_manager.processing_settings

        # General
        app.theme = self.theme_var.get().lower()

        # Find language code from display name
        lang_name = self.language_var.get()
        for code, name in LANGUAGE_CODES.items():
            if name == lang_name:
                proc.language = code
                break

        # API
        app.api_key = self.api_key_var.get().strip()
        proc.claude_model = self.claude_model_var.get()

        # Processing
        proc.whisper_model = self.whisper_model_var.get()
        proc.quality_preset = self.quality_preset_var.get()
        proc.use_gpu = self.use_gpu_var.get()
        proc.enable_chunking = self.enable_chunking_var.get()
        proc.chunk_size_minutes = self.chunk_size_var.get()
        proc.claude_strategy = self.claude_strategy_var.get()

        # Output
        formats = []
        if self.format_docx_var.get():
            formats.append("docx")
        if self.format_pptx_var.get():
            formats.append("pptx")
        if self.format_md_var.get():
            formats.append("md")
        proc.output_formats = formats

        proc.include_transcript = self.include_transcript_var.get()
        proc.keep_screenshots = self.keep_screenshots_var.get()
        proc.super_detailed_output = self.super_detailed_var.get()

        # Save to file
        self.settings_manager.save()

        # Apply theme immediately
        ctk.set_appearance_mode(app.theme)

        # Notify parent
        if self.on_save_callback:
            self.on_save_callback()

        messagebox.showinfo("Settings", "Settings saved successfully!")
        self.destroy()

    def _reset_defaults(self):
        """Reset settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Reset all settings to defaults?\n(API key will be preserved)"):
            self.settings_manager.reset_to_defaults()
            self._load_settings()


class FrameNotesApp(ctk.CTk):
    """Main application class for FrameNotes GUI v2.0"""

    def __init__(self):
        super().__init__()

        # Settings manager (load settings from file)
        self.settings_manager = get_settings_manager()

        # Window configuration
        self.title(APP_TITLE)
        self.geometry("650x750")
        self.minsize(600, 700)

        # Apply saved theme
        saved_theme = self.settings_manager.app_settings.theme
        ctk.set_appearance_mode(saved_theme)

        # Processing state
        self.processing = False
        self.cancel_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.start_time = None

        # Video analyzer
        self.analyzer = VideoAnalyzer()
        self.current_video_info: Optional[VideoInfo] = None
        self.current_config: Optional[ProcessingConfig] = None

        # Variables (loaded from settings)
        proc = self.settings_manager.processing_settings
        self.video_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.format_docx = ctk.BooleanVar(value="docx" in proc.output_formats)
        self.format_pptx = ctk.BooleanVar(value="pptx" in proc.output_formats)
        self.format_md = ctk.BooleanVar(value="md" in proc.output_formats)
        self.model_var = ctk.StringVar(value="Auto")
        self.language_var = ctk.StringVar(value=LANGUAGE_CODES.get(proc.language, "Auto-detect"))
        self.include_transcript = ctk.BooleanVar(value=proc.include_transcript)
        self.keep_screenshots = ctk.BooleanVar(value=proc.keep_screenshots)

        # Advanced options variables (loaded from settings)
        self.auto_detect_var = ctk.BooleanVar(value=proc.auto_detect)
        self.enable_chunking_var = ctk.BooleanVar(value=proc.enable_chunking)
        self.use_gpu_var = ctk.BooleanVar(value=proc.use_gpu)
        self.enable_checkpoints_var = ctk.BooleanVar(value=proc.enable_checkpoints)
        self.quality_preset_var = ctk.StringVar(value=proc.quality_preset.capitalize())
        self.chunk_size_var = ctk.IntVar(value=proc.chunk_size_minutes)

        self._create_widgets()
        self._setup_keyboard_shortcuts()
        self._check_api_key()
        self._update_gpu_status()

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.bind("<Control-o>", lambda e: self._browse_video())
        self.bind("<Control-s>", lambda e: self._open_settings())
        self.bind("<Escape>", lambda e: self._cancel_generation() if self.processing else None)

    def _check_api_key(self):
        """Check if API key is set"""
        if not self.settings_manager.get_api_key():
            messagebox.showwarning(
                "API Key Missing",
                "Anthropic API key not configured.\n\n"
                "Please set it in Settings > API tab,\n"
                "or set the ANTHROPIC_API_KEY environment variable.\n\n"
                "Press Ctrl+S to open Settings."
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
            text="FrameNotes",
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

        # GPU status and Settings button frame
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.grid(row=0, column=1, rowspan=2, sticky="e")

        self.gpu_status_label = ctk.CTkLabel(
            status_frame,
            text="GPU: Checking...",
            font=ctk.CTkFont(size=12)
        )
        self.gpu_status_label.pack(side="left", padx=(0, 10))

        # Settings button
        settings_btn = ctk.CTkButton(
            status_frame,
            text="⚙",
            width=35,
            height=35,
            font=ctk.CTkFont(size=16),
            command=self._open_settings
        )
        settings_btn.pack(side="left")
        CTkToolTip(settings_btn, "Settings (Ctrl+S)")

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
            text=f"FrameNotes v{APP_VERSION} - Powered by Claude AI",
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

    def _open_settings(self):
        """Open settings window"""
        def on_settings_saved():
            # Reload settings into UI variables
            proc = self.settings_manager.processing_settings
            self.format_docx.set("docx" in proc.output_formats)
            self.format_pptx.set("pptx" in proc.output_formats)
            self.format_md.set("md" in proc.output_formats)
            self.language_var.set(LANGUAGE_CODES.get(proc.language, "Auto-detect"))
            self.include_transcript.set(proc.include_transcript)
            self.keep_screenshots.set(proc.keep_screenshots)
            self.enable_chunking_var.set(proc.enable_chunking)
            self.use_gpu_var.set(proc.use_gpu)
            self.chunk_size_var.set(proc.chunk_size_minutes)
            self.quality_preset_var.set(proc.quality_preset.capitalize())
            self.enable_checkpoints_var.set(proc.enable_checkpoints)
            self.auto_detect_var.set(proc.auto_detect)  # Fixed: was missing
            # Update model dropdown based on auto_detect
            if proc.auto_detect:
                self.model_var.set("Auto")
            # Update theme button icon
            current = ctk.get_appearance_mode()
            self.theme_btn.configure(text="☀" if current == "Dark" else "🌙")

        SettingsWindow(self, self.settings_manager, on_save=on_settings_saved)

    def _toggle_theme(self):
        """Toggle between dark and light mode and save preference"""
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.theme_btn.configure(text="☀" if new_mode == "Dark" else "🌙")
        # Save theme preference
        self.settings_manager.app_settings.theme = new_mode.lower()
        self.settings_manager.save()

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

        if not self.settings_manager.get_api_key():
            messagebox.showerror(
                "Error",
                "Anthropic API key not configured.\n\n"
                "Please set it in Settings > API tab,\n"
                "or set the ANTHROPIC_API_KEY environment variable."
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

        # THREAD SAFETY: Snapshot all settings in main thread before background processing
        proc_settings = self.settings_manager.processing_settings
        claude_strategy = proc_settings.claude_strategy
        if claude_strategy == "auto" and self.current_config:
            claude_strategy = self.current_config.claude_strategy

        snapshot = ProcessingSnapshot(
            video_path=self.video_path.get(),
            output_dir=self.output_dir.get(),
            whisper_model=self._get_effective_model(),
            language=self._get_effective_language(),
            enable_chunking=self.enable_chunking_var.get(),
            use_gpu=self.use_gpu_var.get(),
            chunk_size=self.chunk_size_var.get(),
            claude_strategy=claude_strategy,
            claude_model=proc_settings.claude_model,
            super_detailed=proc_settings.super_detailed_output,
            api_key=self.settings_manager.get_api_key(),
            format_docx=self.format_docx.get(),
            format_pptx=self.format_pptx.get(),
            format_md=self.format_md.get(),
            include_transcript=self.include_transcript.get(),
            keep_screenshots=self.keep_screenshots.get()
        )

        # Start processing thread with snapshot
        thread = threading.Thread(target=self._process_video, args=(snapshot,), daemon=True)
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

    def _process_video(self, snapshot: ProcessingSnapshot):
        """Process video in background thread with v2 chunking and hierarchical analysis.

        Args:
            snapshot: Thread-safe snapshot of all settings captured in main thread.
        """
        video_path = snapshot.video_path
        output_dir = Path(snapshot.output_dir)
        video_name = Path(video_path).stem

        temp_dir = tempfile.mkdtemp(prefix="framenotes_gui_")
        screenshots_dir = Path(temp_dir) / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use settings from snapshot (thread-safe)
            model = snapshot.whisper_model
            language = snapshot.language
            enable_chunking = snapshot.enable_chunking
            use_gpu = snapshot.use_gpu
            chunk_size = snapshot.chunk_size
            claude_strategy = snapshot.claude_strategy
            claude_model = snapshot.claude_model
            super_detailed = snapshot.super_detailed
            api_key = snapshot.api_key

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
            ai_analyzer = HierarchicalAnalyzer(
                api_key=api_key,
                model=claude_model,
                chapter_size_minutes=12,
                super_detailed=super_detailed
            )
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

            # Copy screenshots if requested
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
            self.progress_queue.put({
                'type': 'error',
                'message': str(e)
            })
        finally:
            # Cleanup temp directory (use snapshot for thread safety)
            if not snapshot.keep_screenshots:
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
    app = FrameNotesApp()
    app.mainloop()


if __name__ == "__main__":
    main()
