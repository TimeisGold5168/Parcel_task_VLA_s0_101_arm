"""Make a balanced, shuffled heap schedule for a parcel dataset.

Each block is one heap build: the parcels that go into the heap and the
label state of each. The episode order inside a block comes from the
heap rule (free parcel nearest to the flip zone), not from this file.

  python outputs/portfolio/parcel_task/make_schedule.py \
    --out outputs/portfolio/parcel_task/schedule_A1.csv \
    --flip 16 --noflip 9 --block 7 --seed 1

--mix random (default) gives each heap a random mix; --mix even gives
each heap 1-2 of each kind. The totals are balanced in both cases.

Add --keep N to keep heaps 1..N of the existing file (heaps already
recorded) and deal only the remaining cases into the later heaps.
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
INVENTORY = HERE.parent / "parcel_labels" / "parcel_inventory.csv"


def load_train_parcels():
    """Return {(colour, type): [marker_id, ...]} for train parcels."""
    groups = {}
    with open(INVENTORY) as f:
        for row in csv.DictReader(f):
            if row["role"] != "train":
                continue
            key = (row["colour"], row["type"])
            groups.setdefault(key, []).append(int(row["marker_id"]))
    return groups


def deal(need, block, rng):
    """Deal the parcels of each kind round the heaps, one per heap in turn.

    Heap sizes and the count of each kind per heap differ by at most 1.
    Returns one Counter of kinds per heap.
    """
    total = sum(up + down for up, down in need.values())
    n_heaps = -(-total // block)
    order = list(range(n_heaps))
    kinds = list(need)
    rng.shuffle(order)
    rng.shuffle(kinds)
    heaps = [Counter() for _ in range(n_heaps)]
    i = 0
    for k in kinds:
        for _ in range(sum(need[k])):
            heaps[order[i % n_heaps]][k] += 1
            i += 1
    return heaps


def add_labels(heaps, need, rng):
    """Give each label-down case to the heap that has the fewest.

    The fewest is first within the kind, then over the whole heap.
    """
    down = [Counter() for _ in heaps]
    kinds = list(need)
    rng.shuffle(kinds)
    for k in kinds:
        for _ in range(need[k][1]):
            free = [h for h in range(len(heaps))
                    if heaps[h][k] > down[h][k]]
            h = min(free, key=lambda h: (
                down[h][k] / heaps[h][k],
                sum(down[h].values()) / sum(heaps[h].values()),
                rng.random()))
            down[h][k] += 1
    blocks = []
    for heap, dn in zip(heaps, down):
        cases = [(k, "down") for k in heap for _ in range(dn[k])]
        cases += [(k, "up") for k in heap for _ in range(heap[k] - dn[k])]
        rng.shuffle(cases)
        blocks.append(cases)
    return blocks


def random_blocks(groups, need, block, rng, tries=10000):
    """Shuffle all cases and cut them into heaps of near-equal size.

    Each heap gets a random mix; only the totals are balanced. A heap
    never needs more parcels of a kind than exist.
    """
    cases = [(k, lab) for k, (up, down) in need.items()
             for lab in ["up"] * up + ["down"] * down]
    n_heaps = -(-len(cases) // block)
    sizes = [len(cases) // n_heaps + (i < len(cases) % n_heaps)
             for i in range(n_heaps)]
    for _ in range(tries):
        rng.shuffle(cases)
        blocks, i = [], 0
        for n in sizes:
            blocks.append(cases[i:i + n])
            i += n
        if all(Counter(k for k, _ in b)[k] <= len(groups[k])
               for b in blocks for k in groups):
            return blocks
    raise RuntimeError("no valid split; use a smaller --block")


def make_blocks(groups, need, block, rng, mix="random"):
    """Split the cases into heaps (random or even mix per heap)."""
    if mix == "random":
        return random_blocks(groups, need, block, rng)
    heaps = deal(need, block, rng)
    if any(h[k] > len(groups[k]) for h in heaps for k in groups):
        raise RuntimeError("no valid split; use a smaller --block")
    return add_labels(heaps, need, rng)


def load_kept(path, keep):
    """Rows of heaps 1..keep from an existing schedule file."""
    if keep == 0:
        return []
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["block"]) <= keep]
    for r in rows:
        r["block"], r["marker_id"] = int(r["block"]), int(r["marker_id"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flip", type=int, default=16,
                    help="label-up episodes per parcel kind")
    ap.add_argument("--noflip", type=int, default=9,
                    help="label-down episodes per parcel kind")
    ap.add_argument("--block", type=int, default=7,
                    help="most parcels per heap build (stack area size)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mix", choices=["random", "even"], default="random",
                    help="random: random mix per heap, balanced totals; "
                         "even: 1-2 of each kind per heap")
    ap.add_argument("--keep", type=int, default=0,
                    help="keep heaps 1..N of the existing --out file")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    groups = load_train_parcels()
    rows = load_kept(args.out, args.keep)
    done = Counter(((r["colour"], r["type"]), r["label"]) for r in rows)
    need = {k: (args.flip - done[(k, "up")], args.noflip - done[(k, "down")])
            for k in groups}
    blocks = make_blocks(groups, need, args.block, rng, args.mix)
    use = Counter(r["marker_id"] for r in rows)
    for b, cases in enumerate(blocks, args.keep + 1):
        for (colour, kind), label in cases:
            ids = sorted(groups[(colour, kind)], key=lambda i: use[i])
            taken = {r["marker_id"] for r in rows if r["block"] == b}
            mid = next(i for i in ids if i not in taken)
            use[mid] += 1
            rows.append({"block": b, "marker_id": mid, "colour": colour,
                         "type": kind, "label": label})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_heaps = args.keep + len(blocks)
    print(f"{len(rows)} episodes in {n_heaps} heaps -> {args.out}")
    for b in range(1, n_heaps + 1):
        part = [r for r in rows if r["block"] == b]
        txt = ", ".join(f"{r['marker_id']}{'U' if r['label'] == 'up' else 'D'}"
                        for r in part)
        print(f"  heap {b:2d} ({len(part):2d}): {txt}")
    print("U = label up (flip), D = label down (no flip)")


if __name__ == "__main__":
    main()
