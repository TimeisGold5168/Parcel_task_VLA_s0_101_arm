"""Print the episodes that start or end away from the home pose."""
import glob
import json
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
cols = ["episode_index", "observation.state"]
d = pd.concat(pd.read_parquet(f, columns=cols)
              for f in glob.glob(f"{root}/data/*/*.parquet"))
names = json.load(open(f"{root}/meta/info.json"))["features"][
    "observation.state"]["names"]
s = {e: np.stack(g["observation.state"].values)
     for e, g in d.groupby("episode_index")}
home = np.median(np.stack([v[0] for v in s.values()]), axis=0)
for e, v in sorted(s.items()):
    for tag, p in (("start", v[0]), ("end", v[-1])):
        off = {names[k]: round(float(p[k] - home[k]), 1)
               for k in range(5) if abs(p[k] - home[k]) > 5}
        if off:
            print(e, tag, off)
