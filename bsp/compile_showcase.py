"""Stack the showcase clips of several speeds into one video.

Each row plays one speed. When the arm is back home after the drop,
the row freezes, turns 50% grey and shows the finish time in big text.
The video ends --hold seconds after the slowest row stops.

Stop time (per row): the first time after the farthest reach when all
5 arm joints are within --home-tol deg of the first command (from
ep<n>_cmd.npy), moved onto the clip timeline. --stops overrides it.

Usage:
  python outputs/portfolio/bsp/compile_showcase.py out.mp4 \
    runs/sweep_A/x1 runs/sweep_B/x1.5 runs/sweep_A/x2 ...
  # manual stops in clip seconds (as the clip timer shows):
  ... --stops 1=12.7,1.5=8.5
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from video_record import FFMPEG

GO_HOME_S = 2.5  # go_to(home) after Enter, see run_bsp.py


def clip_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)])
    return float(out)


def auto_stop(run, ep, tol):
    """Stop time on the clip timeline (s) from the command log."""
    cmd = np.load(run / f"ep{ep}_cmd.npy")
    t, arm = cmd[:, 0], cmd[:, 2:7]
    dev = np.abs(arm - arm[0]).max(1)
    far = int(np.argmax(dev))
    back = np.nonzero(dev[far:] < tol)[0]
    t_stop = t[far + back[0]] if len(back) else t[-1]
    # the clip starts slightly before the episode clock
    offset = clip_duration(run / f"ep{ep}_showcase.mp4") - t[-1] - GO_HOME_S
    return t_stop + max(offset, 0.0)


def build(rows, out, width, hold):
    end = max(s for _, _, s in rows) + hold
    h = width * 480 // 1920 // 2 * 2
    inputs, chains = [], []
    for i, (speed, clip, stop) in enumerate(rows):
        inputs += ["-i", str(clip)]
        # freeze on the last frame at or before stop; the label truncates
        # like the clip timer, so both show the same number
        last = np.floor(stop * 30 + 1e-6) / 30
        text = f"{speed:g}x   done in {np.floor(last * 10 + 1e-6) / 10:.1f} s"
        on = f"enable='gte(t,{last + 1 / 60:.3f})'"
        chains.append(
            f"[{i}:v]trim=end={last + 1 / 60:.3f},setpts=PTS-STARTPTS,"
            f"scale={width}:{h},"
            f"tpad=stop_mode=clone:stop_duration={end - stop + 1:.3f},"
            f"trim=end={end:.3f},"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=gray@0.5:t=fill:{on},"
            f"drawtext=text='{text}':x=(w-tw)/2:y=(h-th)/2:"
            f"fontsize={h // 4}:fontcolor=white:borderw=4:"
            f"bordercolor=black:{on}[r{i}]")
    stack = "".join(f"[r{i}]" for i in range(len(rows)))
    graph = ";".join(chains) + f";{stack}vstack=inputs={len(rows)}[out]"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-r", "30", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="speed folders, e.g. .../x1.5")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--stops", default="",
                    help="manual stops in clip seconds, e.g. 1=12.7,2=6.5")
    ap.add_argument("--home-tol", type=float, default=15.0)
    ap.add_argument("--hold", type=float, default=2.0,
                    help="seconds to hold after the last row stops")
    ap.add_argument("--width", type=int, default=1440)
    args = ap.parse_args()
    manual = {float(k): float(v) for k, v in
              (p.split("=") for p in args.stops.split(",") if p)}
    rows = []
    for run in map(Path, args.runs):
        speed = float(run.name.lstrip("x"))
        stop = manual.get(speed, auto_stop(run, args.ep, args.home_tol))
        print(f"{speed:g}x  stop {stop:.2f} s  "
              f"{'manual' if speed in manual else 'auto'}  {run}")
        rows.append((speed, run / f"ep{args.ep}_showcase.mp4", stop))
    rows.sort()
    build(rows, args.out, args.width, args.hold)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
