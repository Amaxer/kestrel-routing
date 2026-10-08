"""Train on data/train.csv + data/resolution_log.csv, write model/router.joblib."""
import sys
from pathlib import Path
import pandas as pd
from kestrel import model
from kestrel.text import normalise_team

D = Path("data")
if not (D / "train.csv").exists():
    sys.exit("Put train.csv and resolution_log.csv in ./data (not committed: Ops Policy section 10).")

t = pd.read_csv(D / "train.csv").merge(pd.read_csv(D / "resolution_log.csv"), on="request_id")
t["final"] = t.final_team.map(normalise_team)   # target = where the request actually ended up
bundle = model.fit(t.request_text, t.final)
model.save(bundle)
print("trained on", len(t), "rows; classes:", list(bundle["clf"].classes_))
