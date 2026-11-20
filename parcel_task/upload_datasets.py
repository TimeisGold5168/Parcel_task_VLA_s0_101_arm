"""Upload the parcel-project datasets to the Hugging Face Hub (private).

Only the datasets behind the working models and the robot results:

  train (default):
    parcel-sort-b                 dataset B (grasp approach), 99 eps:
                                  base of all working parcel models
    parcel-skill-boxstack-cut     20 box-on-box demos, start wait cut
    rollout_dagger_far_edge_pull  16 far-edge DAgger corrections, cut
                                  to reach, pick and place
    rollout_dagger_far_stack_pull far-edge stack corrections (if made)
  --evals (robot test recordings, evidence for the write-up):
    rollout_eval_parcel_smolvla_b_{medium,hard}  SmolVLA B baseline
    rollout_eval_parcel_smolvla_b_dg_medium      DAgger model that did
                                                 not leave home
    rollout_eval_parcel_smolvla_b_fe_hard        far-edge model
    rollout_eval_parcel_smolvla_b_box_medium     box model, old skill
    rollout_eval_parcel_smolvla_b_fb_hard        combined model

Not uploaded: approach A (parcel-sort-a, push approach, replaced by
B), pilots, backups (_old, _backup*), merged training mixes (made
again from the parts with lerobot-edit-dataset merge), the full DAgger
corrections (rollout_dagger_parcel_b*: the models trained on them did
not leave home), intermediate cuts, and the cube (ACT) datasets.

Log in once first:  hf auth login   (a token with write access)

  python outputs/portfolio/parcel_task/upload_datasets.py --dry-run
  python outputs/portfolio/parcel_task/upload_datasets.py --evals
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOW = Path.home() / ".cache/huggingface/lerobot/Low101"
TRAIN = ["parcel-sort-b", "parcel-skill-boxstack-cut",
         "rollout_dagger_far_edge_pull", "rollout_dagger_far_stack_pull"]
EVALS = ["rollout_eval_parcel_smolvla_b_medium",
         "rollout_eval_parcel_smolvla_b_hard",
         "rollout_eval_parcel_smolvla_b_dg_medium",
         "rollout_eval_parcel_smolvla_b_fe_hard",
         "rollout_eval_parcel_smolvla_b_box_medium",
         "rollout_eval_parcel_smolvla_b_fb_hard"]
TAGS = ["LeRobot", "so101", "parcel-sorting", "smolvla"]


def size_gb(path):
    return sum(f.stat().st_size for f in path.rglob("*")
               if f.is_file()) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", action="store_true",
                    help="also upload the robot test recordings")
    ap.add_argument("--owner", default=None,
                    help="Hub user or org (default: the logged-in user)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = TRAIN + (EVALS if args.evals else [])
    todo = [n for n in names if (LOW / n / "meta/info.json").exists()]
    for n in sorted(set(names) - set(todo)):
        print(f"skip (not found): {n}")
    total = 0.0
    for n in todo:
        gb = size_gb(LOW / n)
        total += gb
        print(f"  {n:45s} {gb:5.2f} GB")
    print(f"{len(todo)} datasets, {total:.1f} GB")
    if args.dry_run:
        return

    owner = args.owner or HfApi().whoami()["name"]
    for n in todo:
        repo = f"{owner}/{n}"
        print(f"upload {repo} (private)", flush=True)
        ds = LeRobotDataset(repo, root=LOW / n)
        ds.push_to_hub(private=True, tags=TAGS, license="apache-2.0",
                       upload_large_folder=size_gb(LOW / n) > 1.0)
    print("done")


if __name__ == "__main__":
    main()
