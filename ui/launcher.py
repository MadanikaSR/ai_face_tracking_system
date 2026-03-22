import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class LauncherUI:
    """
    Launcher UI — supports Video File and RTSP Stream only.
    Webcam option has been removed.
    """
    def __init__(self, current_config, reset_callback):
        self.root = tk.Tk()
        self.root.title("AI Face Tracking System — Launcher")
        self.root.geometry("540x420")
        self.root.resizable(False, False)
        self.result_config = None
        self.current_config = current_config
        self.reset_callback = reset_callback
        self._setup_ui()

    def _setup_ui(self):
        # ── Style ──────────────────────────────────────────────────────────────
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel",   font=("Segoe UI", 9))

        main_frame = ttk.Frame(self.root, padding="24")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="🎯  AI Face Tracking System", style="Title.TLabel").pack(pady=(0, 2))
        ttk.Label(main_frame, text="Select input source and press START", style="Sub.TLabel", foreground="gray").pack(pady=(0, 16))

        # ── Input Type ─────────────────────────────────────────────────────────
        # Default to current config; fall back to "video" if webcam was previously selected
        current_type = self.current_config.get("input_type", "video")
        if current_type not in ("video", "rtsp"):
            current_type = "video"
        self.input_type = tk.StringVar(value=current_type)

        types_frame = ttk.LabelFrame(main_frame, text="  Input Source  ", padding="12")
        types_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Radiobutton(types_frame, text="📂  Video File", variable=self.input_type,
                        value="video", command=self._update_fields).pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(types_frame, text="📡  RTSP Stream", variable=self.input_type,
                        value="rtsp",  command=self._update_fields).pack(anchor=tk.W, pady=3)

        # ── Settings ───────────────────────────────────────────────────────────
        settings_frame = ttk.LabelFrame(main_frame, text="  Source Settings  ", padding="12")
        settings_frame.pack(fill=tk.X, pady=(0, 12))
        settings_frame.columnconfigure(1, weight=1)

        # Video path row
        self.lbl_video = ttk.Label(settings_frame, text="Video Path:")
        self.lbl_video.grid(row=0, column=0, sticky=tk.W, pady=4)
        self.video_path = ttk.Entry(settings_frame)
        self.video_path.insert(0, self.current_config.get("video_path", ""))
        self.video_path.grid(row=0, column=1, sticky=tk.EW, padx=6)
        self.btn_browse = ttk.Button(settings_frame, text="Browse", command=self._browse_video)
        self.btn_browse.grid(row=0, column=2)

        # RTSP URL row
        self.lbl_rtsp = ttk.Label(settings_frame, text="RTSP URL:")
        self.lbl_rtsp.grid(row=1, column=0, sticky=tk.W, pady=4)
        self.rtsp_url = ttk.Entry(settings_frame)
        self.rtsp_url.insert(0, self.current_config.get("rtsp_url", ""))
        self.rtsp_url.grid(row=1, column=1, sticky=tk.EW, padx=6, columnspan=2)

        # RTSP placeholder hint
        self.lbl_rtsp_hint = ttk.Label(
            settings_frame,
            text="e.g. rtsp://username:password@ip:port/stream",
            foreground="gray", font=("Segoe UI", 8)
        )
        self.lbl_rtsp_hint.grid(row=2, column=1, sticky=tk.W, padx=6, columnspan=2)

        self._update_fields()  # Set correct enabled/disabled state

        # ── Reset flag ─────────────────────────────────────────────────────────
        self.reset_on_start = tk.BooleanVar(value=self.current_config.get("reset_on_start", False))
        ttk.Checkbutton(main_frame, text="⚠️  Reset all data on system start",
                        variable=self.reset_on_start).pack(pady=(0, 16))

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack()

        ttk.Button(btn_frame, text="▶  START SYSTEM",  command=self._on_start,     width=20).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="🗑  RESET DATA NOW", command=self._on_reset_now, width=20).pack(side=tk.LEFT, padx=8)

    def _update_fields(self):
        """Enable/disable fields depending on selected input type."""
        is_video = self.input_type.get() == "video"
        state_video = tk.NORMAL if is_video  else tk.DISABLED
        state_rtsp  = tk.NORMAL if not is_video else tk.DISABLED

        self.video_path.configure(state=state_video)
        self.btn_browse.configure(state=state_video)
        self.rtsp_url.configure(state=state_rtsp)

    def _browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.ts"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.delete(0, tk.END)
            self.video_path.insert(0, filename)

    def _on_reset_now(self):
        if messagebox.askyesno("Confirm Reset",
                               "Delete ALL registered faces and event logs?\nThis cannot be undone."):
            self.reset_callback()
            messagebox.showinfo("Done", "System data has been cleared.")

    def _on_start(self):
        input_type = self.input_type.get()

        if input_type == "video":
            path = self.video_path.get().strip()
            if not path:
                messagebox.showerror("Error", "Please select a video file.")
                return

        elif input_type == "rtsp":
            url = self.rtsp_url.get().strip()
            if not url:
                messagebox.showerror("Error", "Please enter an RTSP URL.\ne.g. rtsp://user:pass@ip:port/stream")
                return

        self.result_config = {
            "input_type":         input_type,
            "webcam_index":       0,   # kept for schema compatibility, unused
            "video_path":         self.video_path.get().strip(),
            "rtsp_url":           self.rtsp_url.get().strip(),
            "reset_on_start":     self.reset_on_start.get(),
            "detection_frame_skip":    self.current_config.get("detection_frame_skip", 2),
            "recognition_threshold":   self.current_config.get("recognition_threshold", 0.6),
            "exit_timeout_seconds":    self.current_config.get("exit_timeout_seconds", 30),
        }
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result_config
