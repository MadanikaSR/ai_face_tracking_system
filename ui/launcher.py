import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class LauncherUI:
    def __init__(self, current_config, reset_callback):
        self.root = tk.Tk()
        self.root.title("AI Face Tracking System - Launcher")
        self.root.geometry("500x450")
        self.result_config = None
        self.current_config = current_config
        self.reset_callback = reset_callback
        
        self._setup_ui()

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Select Input Source", font=("Arial", 14, "bold")).pack(pady=10)

        # Input Type Selection
        self.input_type = tk.StringVar(value=self.current_config.get("input_type", "webcam"))
        
        types_frame = ttk.LabelFrame(main_frame, text="Input Type", padding="10")
        types_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(types_frame, text="Webcam", variable=self.input_type, value="webcam").pack(anchor=tk.W)
        ttk.Radiobutton(types_frame, text="Video File", variable=self.input_type, value="video").pack(anchor=tk.W)
        ttk.Radiobutton(types_frame, text="RTSP Stream", variable=self.input_type, value="rtsp").pack(anchor=tk.W)

        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)

        # Webcam Index
        ttk.Label(settings_frame, text="Webcam Index:").grid(row=0, column=0, sticky=tk.W)
        self.webcam_idx = ttk.Entry(settings_frame)
        self.webcam_idx.insert(0, str(self.current_config.get("webcam_index", 0)))
        self.webcam_idx.grid(row=0, column=1, sticky=tk.EW, padx=5)

        # Video Path
        ttk.Label(settings_frame, text="Video Path:").grid(row=1, column=0, sticky=tk.W)
        self.video_path = ttk.Entry(settings_frame)
        self.video_path.insert(0, self.current_config.get("video_path", ""))
        self.video_path.grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Button(settings_frame, text="Browse", command=self._browse_video).grid(row=1, column=2)

        # RTSP URL
        ttk.Label(settings_frame, text="RTSP URL:").grid(row=2, column=0, sticky=tk.W)
        self.rtsp_url = ttk.Entry(settings_frame)
        self.rtsp_url.insert(0, self.current_config.get("rtsp_url", ""))
        self.rtsp_url.grid(row=2, column=1, sticky=tk.EW, padx=5, columnspan=2)

        # Reset Flag
        self.reset_on_start = tk.BooleanVar(value=self.current_config.get("reset_on_start", False))
        ttk.Checkbutton(main_frame, text="Reset data on system start", variable=self.reset_on_start).pack(pady=5)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="START SYSTEM", command=self._on_start).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="RESET DATA NOW", command=self._on_reset_now).pack(side=tk.LEFT, padx=10)

    def _browse_video(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if filename:
            self.video_path.delete(0, tk.END)
            self.video_path.insert(0, filename)

    def _on_reset_now(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to delete all registered faces and logs?"):
            self.reset_callback()
            messagebox.showinfo("Success", "System data has been cleared.")

    def _on_start(self):
        try:
            self.result_config = {
                "input_type": self.input_type.get(),
                "webcam_index": int(self.webcam_idx.get()),
                "video_path": self.video_path.get(),
                "rtsp_url": self.rtsp_url.get(),
                "reset_on_start": self.reset_on_start.get(),
                "detection_frame_skip": self.current_config.get("detection_frame_skip", 2),
                "recognition_threshold": self.current_config.get("recognition_threshold", 0.6),
                "exit_timeout_seconds": self.current_config.get("exit_timeout_seconds", 30)
            }
            self.root.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid Webcam Index. Must be an integer.")

    def run(self):
        self.root.mainloop()
        return self.result_config
