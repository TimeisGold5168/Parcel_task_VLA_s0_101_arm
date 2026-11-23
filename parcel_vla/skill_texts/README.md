# skill_texts (phase 05)

This folder teaches SmolVLA to read the task text. Each episode gets
the text of its skill. Then a key press or the planner selects the
skill on the robot.

| File | Purpose |
|---|---|
| `make_skill3.py` | Copy a dataset and give each episode the text of its skill |
| `queue_skill3.sh` | Train a skill-text model (10k steps from SmolVLA B), then run the language test |
| `lang_test.py` | Language test on held-out frames: does the correct text give the lowest error? |
| `splits/` | The texts and the held-out episodes of each skill-text model |

## The texts

| Model | Box | Bag | Top box |
|---|---|---|---|
| `sk3` | "pick the box and place it in the flip zone" | "pick the bag and place it in the flip zone" | "pick the top box off the stack" |
| `sk3d`, `sk5d` | "grasp rigid box with wide grip and set it at flip zone" | "pinch soft bag by its edge then drop onto table" | "lift upper carton off stacked pile" |

`sk5d` also has "reach far rim, pull parcel closer" (far edge) and
"sweep lone item beyond wooden desk border" (on the table).

## Results (language test)

| Model | Texts | Bag wins | Text shift | Verdict |
|---|---|---|---|---|
| `b_all` (control) | 1 for all | 8-30 % | 0.3-1.4° | ignores the text |
| `sk3` | differ in 1 word | 62 % | 0.40° (noise 1.62°) | ignores box/bag |
| `sk3d` | no shared word | **85 %** | **3.41°** (noise 1.63°) | reads the text |
| `sk5d` | 5, no shared word | 89 % | 3.49° | the 2 new skills (13-16 episodes) are not learned |

A skill needs about 20 good episodes. The full tables are in
[`results/skill_texts/`](../../results/skill_texts).

```bash
python -m parcel_vla.skill_texts.make_skill3 --variant distinct
VARIANT=distinct bash parcel_vla/skill_texts/queue_skill3.sh
python -m parcel_vla.skill_texts.lang_test smolvla_parcel_b_sk3d_10k:010000 \
  --split outputs/train/skill3d_split.json
```

`make_skill3.py` writes the split files to `TRAIN_DIR`.
`run_parcel.sh` reads them there, or uses the copies in `splits/`.
