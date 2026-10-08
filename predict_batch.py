"""python predict_batch.py [in.csv] [out.csv]  -> predictions in sample_submission shape (+ a detail file)."""
import sys
import pandas as pd
from kestrel import model
src = sys.argv[1] if len(sys.argv) > 1 else "data/test_unlabelled.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
df = pd.read_csv(src); b = model.load()
cl, p = model.predict_proba(b, df.request_text)
out = pd.DataFrame({"request_id": df.request_id, "team": cl[p.argmax(1)]})
out.to_csv(dst, index=False)
det = out.assign(confidence=p.max(1).round(3), auto_route=p.max(1) >= b["threshold"])
det.to_csv(dst.replace(".csv", "_detail.csv"), index=False)
print(len(out), "rows ->", dst, "| auto-routable:", f"{det.auto_route.mean():.1%}")
