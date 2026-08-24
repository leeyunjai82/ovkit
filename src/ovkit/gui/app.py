"""The window. Everything slow lives in :mod:`ovkit.gui.controller`.

    ovkit gui

Pick what you want on the left, point it at a webcam or a picture, read the
answer at the bottom. Tk only draws and forwards clicks, so a model download
never freezes the window.
"""

from __future__ import annotations

import base64
from typing import Any

from .controller import Controller, choices

_BG = "#12161d"
_PANEL = "#1a2029"
_LINE = "#2b3442"
_FG = "#e6edf3"
_MUTED = "#8b949e"
_BRAND = "#22b8cf"

_POLL_MS = 30  # how often the window checks for a new frame


def _tk() -> Any:
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise SystemExit(
            "The ovkit GUI needs tkinter, which is missing from this Python.\n"
            "  Ubuntu/Debian : sudo apt install python3-tk\n"
            "  Fedora        : sudo dnf install python3-tkinter\n"
            "  macOS (brew)  : brew install python-tk\n"
            "Windows and the python.org installers already include it.\n"
            "No display? Use the web demo instead: python examples/web_app.py"
        ) from exc
    return tk, ttk, filedialog


class App:
    """A small window over :class:`~ovkit.gui.controller.Controller`."""

    def __init__(self, device: str = "AUTO", camera: int = 0) -> None:
        self.tk, self.ttk, self.filedialog = _tk()
        self.camera = camera
        self.controller = Controller(device=device)
        self._photo: Any = None  # a live reference, or Tk drops the image
        self._seen = -1
        self._build(device)

    # -- layout -------------------------------------------------------------

    def _build(self, device: str) -> None:
        tk = self.tk
        self.root = tk.Tk()
        self.root.title("ovkit")
        self.root.configure(bg=_BG)
        self.root.geometry("1040x680")
        self.root.minsize(820, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        body = tk.Frame(self.root, bg=_BG)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        right = tk.Frame(body, bg=_BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_toolbar(right, device)
        self._build_canvas(right)
        self._build_answer(right)

    def _build_sidebar(self, parent: Any) -> None:
        tk = self.tk
        side = tk.Frame(parent, bg=_PANEL, width=210)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="ovkit", bg=_PANEL, fg=_BRAND, font=("Segoe UI", 18, "bold")).pack(
            anchor="w", padx=16, pady=(16, 2)
        )
        tk.Label(side, text="pick what you want", bg=_PANEL, fg=_MUTED, font=("Segoe UI", 9)).pack(
            anchor="w", padx=16, pady=(0, 10)
        )

        self.buttons: dict[str, Any] = {}
        for choice in choices():
            button = tk.Button(
                side,
                text=choice.label,
                anchor="w",
                relief="flat",
                bg=_PANEL,
                fg=_FG,
                activebackground=_LINE,
                activeforeground=_FG,
                font=("Segoe UI", 11),
                padx=14,
                pady=6,
                command=lambda c=choice: self._pick(c),
            )
            button.pack(fill="x")
            self.buttons[choice.name] = button

    def _build_toolbar(self, parent: Any, device: str) -> None:
        tk, ttk = self.tk, self.ttk
        bar = tk.Frame(parent, bg=_BG)
        bar.pack(fill="x", padx=14, pady=12)

        self.webcam_button = tk.Button(
            bar,
            text="▶ Webcam",
            relief="flat",
            bg=_BRAND,
            fg="#04222a",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=6,
            command=self._toggle_webcam,
        )
        self.webcam_button.pack(side="left")
        for text, command in (("Open image", self._open), ("Save", self._save)):
            tk.Button(
                bar,
                text=text,
                relief="flat",
                bg=_PANEL,
                fg=_FG,
                font=("Segoe UI", 10),
                padx=12,
                pady=6,
                command=command,
            ).pack(side="left", padx=(8, 0))

        tk.Label(bar, text="device", bg=_BG, fg=_MUTED).pack(side="left", padx=(18, 4))
        self.device_var = tk.StringVar(value=device)
        picker = ttk.Combobox(
            bar,
            textvariable=self.device_var,
            values=["AUTO", "CPU", "GPU", "NPU"],
            width=6,
            state="readonly",
        )
        picker.pack(side="left")
        picker.bind(
            "<<ComboboxSelected>>", lambda _e: self.controller.set_device(self.device_var.get())
        )

        tk.Label(bar, text="conf", bg=_BG, fg=_MUTED).pack(side="left", padx=(18, 4))
        self.conf = tk.Scale(
            bar,
            from_=0.05,
            to=0.95,
            resolution=0.05,
            orient="horizontal",
            length=120,
            bg=_BG,
            fg=_FG,
            troughcolor=_PANEL,
            highlightthickness=0,
            command=lambda v: self.controller.set_conf(float(v)),
        )
        self.conf.set(0.25)
        self.conf.pack(side="left")

    def _build_canvas(self, parent: Any) -> None:
        tk = self.tk
        wrap = tk.Frame(parent, bg="#0b0e13", highlightbackground=_LINE, highlightthickness=1)
        wrap.pack(fill="both", expand=True, padx=14)
        self.canvas = tk.Label(
            wrap,
            bg="#0b0e13",
            fg=_MUTED,
            text="Pick something on the left, then Webcam or Open image.",
            font=("Segoe UI", 11),
        )
        self.canvas.pack(fill="both", expand=True)

    def _build_answer(self, parent: Any) -> None:
        tk = self.tk
        panel = tk.Frame(parent, bg=_BG)
        panel.pack(fill="x", padx=14, pady=12)
        self.answer = tk.Label(
            panel,
            text="",
            bg=_BG,
            fg=_FG,
            font=("Segoe UI", 14, "bold"),
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self.answer.pack(fill="x")
        self.status = tk.Label(
            panel, text="", bg=_BG, fg=_MUTED, font=("Segoe UI", 9), anchor="w", justify="left"
        )
        self.status.pack(fill="x", pady=(4, 0))

    # -- actions ------------------------------------------------------------

    def _pick(self, choice: Any) -> None:
        for name, button in self.buttons.items():
            button.configure(bg=_LINE if name == choice.name else _PANEL)
        self.status.configure(text=choice.hint)
        self.controller.select(choice.name)

    def _toggle_webcam(self) -> None:
        if self.controller.view().live:
            self.controller.stop()
        else:
            self.controller.start_webcam(self.camera)

    def _open(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Open an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if path:
            self.controller.open_image(path)

    def _save(self) -> None:
        if self.controller.view().frame is None:
            self.status.configure(text="Nothing to save yet.")
            return
        path = self.filedialog.asksaveasfilename(defaultextension=".jpg")
        if path and self.controller.save(path):
            self.status.configure(text=f"Saved {path}")

    def _close(self) -> None:
        self.controller.close()
        self.root.destroy()

    # -- drawing ------------------------------------------------------------

    def _tick(self) -> None:
        view = self.controller.view()
        if view.counter != self._seen:
            self._seen = view.counter
            self._draw(view)
        self.root.after(_POLL_MS, self._tick)

    def _draw(self, view: Any) -> None:
        self.webcam_button.configure(text="■ Stop" if view.live else "▶ Webcam")
        if view.error:
            self.answer.configure(text=view.error, fg="#ff7b72")
        else:
            self.answer.configure(text=view.answer, fg=_FG)
        self.status.configure(text=view.status)
        if view.frame is None:
            return
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 200)
        self._photo = self.tk.PhotoImage(data=encode(view.frame, width, height))
        self.canvas.configure(image=self._photo, text="")

    def run(self) -> int:
        """Show the window and block until it is closed."""
        self.root.after(_POLL_MS, self._tick)
        self.root.mainloop()
        return 0


def encode(frame: Any, max_width: int, max_height: int) -> bytes:
    """Fit a BGR frame into the canvas and return base64 PNG for Tk.

    Tk 8.6 reads PNG straight from ``data=``, which keeps the GUI free of
    Pillow — one less thing to install before seeing a result.
    """
    import cv2
    import numpy as np

    image = np.asarray(frame)
    h, w = image.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (max(int(w * scale), 1), max(int(h * scale), 1)))
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode the frame for display")
    return base64.b64encode(buffer.tobytes())


def main(device: str = "AUTO", camera: int = 0) -> int:
    """Entry point for ``ovkit gui``."""
    return App(device=device, camera=camera).run()
