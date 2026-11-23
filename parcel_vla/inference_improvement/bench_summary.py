"""Markdown tables from bench_results/{plan,control}.jsonl (item 14).

Gain = reference p50 / variant p50 for the same model. The last row of
each label counts (a run again replaces the old row).

  python -m parcel_vla.inference_improvement.bench_summary \
    > runs/bench_results/summary.md

The folder is runs/bench_results, or the first argument (for example
results/inference_improvement).
"""

import json
import sys
from pathlib import Path

from parcel_vla import paths

R = (Path(sys.argv[1]) if len(sys.argv) > 1
     else paths.RUNS / "bench_results")


def rows(name):
    p = R / name
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[r.get("label") or r.get("tag") or str(len(out))] = r
    return out


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    plan = rows("plan.jsonl")
    print("## Plan time (bench_plan.py, 5 fixed dataset B frames, fixed "
          "noise, 40 repeats)\n")
    print("| Label | Variant | p50 (ms) | p95 (ms) | Err (deg) | "
          "Err (norm.) | Device mem (MB) | Gain p50 |")
    print("|---|---|---|---|---|---|---|---|")
    for label, r in plan.items():
        model = "diffusion" if "diffusion" in r["tag"] else "smolvla"
        ref = plan.get(f"ref_{model}")
        gain = (f"{ref['p50'] / r['p50']:.2f}x"
                if ref and not label.startswith("ref") else "-")
        print(f"| {label} | {r['tag']} | {fmt(r['p50'])} | "
              f"{fmt(r['p95'])} | {fmt(r.get('err_deg'), 3)} | "
              f"{r.get('err', 0):.1e} | {r['device_mb']} | {gain} |")

    ctrl = rows("control.jsonl")
    bus = [r for r in ctrl.values() if r.get("bench") == "bus_read"]
    cams = [r for r in ctrl.values() if r.get("bench") == "camera"]
    loops = {k: r for k, r in ctrl.items() if r.get("bench") == "loop"}
    if bus:
        print("\n## Servo bus read, Present_Position, 6 servos (ms)\n")
        print("| Driver | p50 | p95 | max | errors |")
        print("|---|---|---|---|---|")
        for r in bus:
            print(f"| {r['impl']} | {r['p50']} | {r['p95']} | {r['max']} "
                  f"| {r.get('errors', '-')} |")
    if cams:
        print("\n## Cameras: read_latest() time and frame age (ms)\n")
        print("| Camera | read p50 | read max | age p50 | age p95 |")
        print("|---|---|---|---|---|")
        for r in cams:
            print(f"| {r['cam']} | {r['read_ms']['p50']} | "
                  f"{r['read_ms']['max']} | {r['age_ms']['p50']} | "
                  f"{r['age_ms']['p95']} |")
    if loops:
        print("\n## Control tick period while the planner runs "
              "(50 Hz = 20 ms, Diffusion-BSP B at 1.5x, fake arm)\n")
        print("| Label | Control | Plan engine | Video | Cameras | Tick p50 "
              "| Tick p95 | Tick max | Late % | Plan p50 | Plan p95 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for label, r in loops.items():
            t, p = r["tick_ms"], r["plan_ms"]
            print(f"| {label} | {r['control']} | {r['trt']} | {r['video']}"
                  f" | {r['cams']} | {t['p50']} | {t['p95']} | {t['max']} "
                  f"| {r['late_pct']} | {p['p50']} | {p['p95']} |")


if __name__ == "__main__":
    main()
