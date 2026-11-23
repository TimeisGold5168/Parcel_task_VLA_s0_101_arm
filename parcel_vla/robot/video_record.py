"""Full-rate video recording for showcase clips.

One thread per camera writes every frame at the camera rate. Missed
frames repeat the last frame, so the clip plays in real time. After
the episode, ffmpeg converts each clip to H.264 (plays in browsers)
and builds a side-by-side showcase clip with a caption and a timer.
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPLAY = shutil.which("ffplay") or "ffplay"


class Recorder:
    """Writes frames from read_frame() (RGB array) to an MJPG .avi."""

    def __init__(self, name, read_frame, path, fps=30):
        self.name, self.read_frame = name, read_frame
        self.path, self.fps = Path(path), fps
        self.running = False
        self.thread = None
        self.error = None
        self.latest = None

    def start(self, t0):
        self.t0, self.running = t0, True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)

    def _run(self):
        writer, written = None, 0
        try:
            while self.running:
                frame = self.read_frame()
                if frame is None:
                    continue
                self.latest = frame
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if writer is None:
                    h, w = bgr.shape[:2]
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(self.path), cv2.VideoWriter_fourcc(*"MJPG"),
                        self.fps, (w, h))
                due = int((time.perf_counter() - self.t0) * self.fps) + 1
                while written < due:
                    writer.write(bgr)
                    written += 1
        except Exception as exc:  # keep the robot running
            self.error = exc
        finally:
            if writer is not None:
                writer.release()


class ExtraCamera:
    """A camera that the robot does not use (e.g. a side view)."""

    def __init__(self, device, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open showcase camera {device}")
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            ok, bgr = self.cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def release(self):
        self.cap.release()


class Preview:
    """Live side-by-side window (ffplay) of the cameras.

    idle maps each name to a read function used while no recorder runs
    (e.g. before Enter, to aim the cameras). While a recorder with that
    name runs, its latest frame is shown instead, so the camera is not
    read twice. Closing the window stops the preview; the run continues.
    """

    def __init__(self, names, idle, tile=(424, 318), fps=15):
        self.names, self.idle = names, idle
        self.tile, self.fps = tile, fps
        self.sources = {}
        w, h = tile
        self.proc = subprocess.Popen(
            [FFPLAY, "-loglevel", "error", "-nostats", "-fflags", "nobuffer",
             "-window_title", " | ".join(names), "-f", "rawvideo",
             "-pixel_format", "rgb24", "-video_size",
             f"{w * len(names)}x{h}", "-framerate", str(fps), "-i", "-"],
            stdin=subprocess.PIPE)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def show(self, recorders):
        self.sources = {r.name: r for r in recorders if r.name in self.names}

    def _run(self):
        w, h = self.tile
        tiles = [np.zeros((h, w, 3), np.uint8) for _ in self.names]
        while self.running:
            for i, name in enumerate(self.names):
                r = self.sources.get(name)
                if r is not None and r.running:
                    frame = r.latest
                else:
                    frame = self.idle[name]()
                if frame is not None:
                    tiles[i] = cv2.resize(frame, (w, h))
            try:
                self.proc.stdin.write(np.hstack(tiles).tobytes())
            except (BrokenPipeError, ValueError):
                break
            time.sleep(1 / self.fps)

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.terminate()


def to_h264(src, dst):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    str(dst)], check=True)


def showcase(clips, dst, caption):
    """Side-by-side clip (height 480) with a caption and a timer."""
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]
    scaled = "".join(f"[{i}:v]scale=-2:480[v{i}];" for i in range(len(clips)))
    stack = "".join(f"[v{i}]" for i in range(len(clips)))
    stack += (f"hstack=inputs={len(clips)}[s];" if len(clips) > 1
              else "copy[s];")
    text = caption.replace(":", r"\:").replace("'", "")
    draw = ("[s]drawbox=x=0:y=0:w=iw:h=40:color=black@0.6:t=fill,"
            f"drawtext=text='{text}':x=12:y=10:fontsize=22:fontcolor=white,"
            "drawtext=text='%{eif\\:t\\:d}.%{eif\\:mod(t*10\\,10)\\:d} s':"
            "x=w-tw-12:y=10:fontsize=22:fontcolor=white[out]")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", scaled + stack + draw,
                    "-map", "[out]", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "20", str(dst)], check=True)


def finish(recorders, out, prefix, caption):
    """Convert the raw clips and build the showcase clip."""
    clips = []
    for r in recorders:
        if r.error is not None:
            print(f"video {r.name}: recording failed ({r.error})")
        if not r.path.exists():
            continue
        dst = Path(out) / f"{prefix}_{r.name}.mp4"
        to_h264(r.path, dst)
        r.path.unlink()
        clips.append(dst)
    if clips:
        try:
            showcase(clips, Path(out) / f"{prefix}_showcase.mp4", caption)
        except subprocess.CalledProcessError as exc:
            print(f"showcase clip failed: {exc}")
    return clips
