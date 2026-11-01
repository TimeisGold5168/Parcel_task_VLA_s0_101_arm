"""lerobot-record with a large live camera preview (no Rerun).

Takes the same arguments as lerobot-record. A background thread copies
the newest frame of each robot camera (it does not consume frames, so
the recording loop is not slowed) and shows them in one mpv window
(ffplay if mpv is missing):
the first camera large on the left, the others stacked on the right.

  python outputs/portfolio/parcel_task/record_preview.py <record args>

Keys (Right, Left, Esc, MARK_KEY) work only while the preview window
has the focus; LeRobot's letter shortcuts n, r and q are off.

Phase marks: press MARK_KEY (default space) at the end of each phase
while an episode records. The marks of each saved episode go to
<dataset root>/phase_marks.csv (episode, mark, frame, t_s); marks of a
re-recorded episode are dropped. cut_phases.py uses them. Episode
numbers are not updated if episodes are deleted later.

Env: PREVIEW_W (window width, default 1600), PREVIEW_FPS (default 15),
MARK_KEY (default space).
"""

import atexit
import contextlib
import csv
import logging
import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_record as rec
from lerobot.utils.keyboard_input import (apply_recording_control,
                                          create_key_listener)

MPV = shutil.which("mpv")
FFPLAY = shutil.which("ffplay") or "ffplay"


def viewer_cmd(w, h, fps):
    """Window that reads rgb24 frames from stdin and ignores all keys.

    ffplay is only the fallback: its window pauses on Space and seeks on
    the arrow keys, which are the recording keys.
    """
    if MPV:
        return [MPV, "--really-quiet", "--title=recording preview", "--ontop",
                "--no-input-default-bindings", "--input-vo-keyboard=no",
                "--no-osc", "--cache=no", "--untimed",
                "--profile=low-latency", "--demuxer=rawvideo",
                f"--demuxer-rawvideo-w={w}", f"--demuxer-rawvideo-h={h}",
                "--demuxer-rawvideo-mp-format=rgb24",
                f"--demuxer-rawvideo-fps={fps}", "-"]
    return [FFPLAY, "-loglevel", "error", "-window_title",
            "recording preview", "-f", "rawvideo", "-pixel_format",
            "rgb24", "-video_size", f"{w}x{h}", "-framerate", str(fps),
            "-fflags", "nobuffer", "-i", "-"]


class PhaseMarks:
    """Frame indices marked with MARK_KEY in the episode being recorded."""

    def __init__(self, key):
        self.key = key
        self.ds = None
        self.last_frame_t = 0.0
        self.pending = []

    def press(self, name):
        ds = self.ds
        # Frames stop during the reset period; ignore marks there.
        if (name != self.key or ds is None
                or time.time() - self.last_frame_t > 0.3):
            return
        frame = ds.writer.episode_buffer["size"]
        self.pending.append(frame)
        logging.info(f"phase mark {len(self.pending)} at "
                     f"{frame / ds.fps:.1f} s")

    def save(self, ds, episode):
        path = ds.root / "phase_marks.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["episode", "mark", "frame", "t_s"])
            for i, frame in enumerate(self.pending, 1):
                w.writerow([episode, i, frame, round(frame / ds.fps, 2)])
        logging.info(f"episode {episode}: {len(self.pending)} phase marks")
        self.pending = []


MARKS = PhaseMarks(os.environ.get("MARK_KEY", "space"))
_add_frame = LeRobotDataset.add_frame
_save_episode = LeRobotDataset.save_episode
_clear_buffer = LeRobotDataset.clear_episode_buffer


def add_frame(self, frame):
    MARKS.ds = self
    MARKS.last_frame_t = time.time()
    _add_frame(self, frame)


def save_episode(self, *args, **kwargs):
    episode = self.num_episodes
    _save_episode(self, *args, **kwargs)
    MARKS.save(self, episode)


def clear_episode_buffer(self, *args, **kwargs):
    MARKS.pending = []
    _clear_buffer(self, *args, **kwargs)


LeRobotDataset.add_frame = add_frame
LeRobotDataset.save_episode = save_episode
LeRobotDataset.clear_episode_buffer = clear_episode_buffer
XDOTOOL = shutil.which("xdotool")


def preview_focused():
    """True if the preview window has the keyboard focus (X11 only)."""
    if not XDOTOOL:
        return True
    try:
        name = subprocess.run([XDOTOOL, "getactivewindow", "getwindowname"],
                              capture_output=True, text=True,
                              timeout=0.5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return "recording preview" in name


def init_keyboard_listener():
    """Arrows, Esc and the mark key, only while the preview has focus.

    The listener sees keys typed in every window, so LeRobot's letter
    shortcuts (n, r, q) end episodes while the user types elsewhere.
    """
    events = {"exit_early": False, "rerecord_episode": False,
              "stop_recording": False}

    def on_key(name):
        if name not in ("right", "left", "esc", MARKS.key):
            return
        if not preview_focused():
            logging.info(f"key '{name}' ignored: click the preview window")
            return
        if name == MARKS.key:
            MARKS.press(name)
        else:
            apply_recording_control(name, events)

    listener = create_key_listener(
        on_key, controls_help="preview window focused: Right/Left/Esc, "
        f"{MARKS.key}=phase mark")
    return listener, events


rec.init_keyboard_listener = init_keyboard_listener


def even(x):
    return int(x) // 2 * 2


class Preview(threading.Thread):
    """Show the newest frame of every camera in one ffplay window."""

    def __init__(self, cameras, width=1600, fps=15):
        super().__init__(daemon=True)
        self.cams = cameras
        self.fps = fps
        self.big_w = even(width * 2 / 3)
        self.big_h = even(self.big_w * 9 / 16)
        n_small = max(1, len(cameras) - 1)
        self.small_w = even(width - self.big_w)
        self.small_h = even(self.small_w * 3 / 4)
        self.w = self.big_w + self.small_w
        self.h = max(self.big_h, self.small_h * n_small)
        self.proc = None
        self.focused = False

    def _frame(self, cam, w, h):
        with cam.frame_lock:
            f = cam.latest_frame
        if f is None:
            return np.zeros((h, w, 3), np.uint8)
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)

    def _compose(self):
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        names = list(self.cams)
        top = even((self.h - self.big_h) / 2)
        tiles = [(names[0], 0, top, self.big_w, self.big_h)]
        tiles += [(n, self.big_w, i * self.small_h, self.small_w,
                   self.small_h) for i, n in enumerate(names[1:])]
        for name, x, y, w, h in tiles:
            canvas[y:y + h, x:x + w] = self._frame(self.cams[name], w, h)
            cv2.putText(canvas, name, (x + 10, y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(canvas, f"phase marks: {len(MARKS.pending)}",
                    (10, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)
        if self.focused:
            text, colour = "keys ON", (0, 255, 0)
        else:
            text, colour = "keys OFF: click here", (255, 60, 60)
        cv2.putText(canvas, text, (self.big_w * 2 // 5, self.h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
        return canvas

    def _take_focus(self):
        if not XDOTOOL:
            return
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run([XDOTOOL, "search", "--sync", "--name",
                            "recording preview", "windowactivate"],
                           capture_output=True, timeout=10)

    def run(self):
        self.proc = subprocess.Popen(
            viewer_cmd(self.w, self.h, self.fps), stdin=subprocess.PIPE)
        atexit.register(self.close)
        threading.Thread(target=self._take_focus, daemon=True).start()
        period = 1 / self.fps
        n = 0
        while self.proc.poll() is None:
            t0 = time.time()
            if n % max(1, self.fps // 2) == 0:
                self.focused = preview_focused()
            n += 1
            try:
                self.proc.stdin.write(self._compose().tobytes())
            except (BrokenPipeError, ValueError):
                break
            time.sleep(max(0.0, period - (time.time() - t0)))

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


_make_robot = rec.make_robot_from_config


def make_robot_with_preview(cfg):
    robot = _make_robot(cfg)
    if robot.cameras:
        Preview(robot.cameras, int(os.environ.get("PREVIEW_W", 1600)),
                int(os.environ.get("PREVIEW_FPS", 15))).start()
    return robot


rec.make_robot_from_config = make_robot_with_preview

if __name__ == "__main__":
    rec.main()
