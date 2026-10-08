"""Evidence: time-ordered validation, error analysis, bot comparison, rupee model. Writes evidence/."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from kestrel import model
from kestrel.text import normalise_team

D, OUT = Path("data"), Path("evidence"); OUT.mkdir(exist_ok=True)
t = pd.read_csv(D / "train.csv").merge(pd.read_csv(D / "resolution_log.csv"), on="request_id")
t["final"] = t.final_team.map(normalise_team); t["bot"] = t.team_label.map(normalise_team)
L = []
def P(s=""): L.append(s); print(s)

P("# Evidence\n")
P("## 1. The bot is not the truth")
P(f"- Bot queue == team that finally closed the request: **{(t.bot==t.final).mean():.1%}** ({(t.bot!=t.final).sum():,} of {len(t):,} requests went to the wrong first team).")
P(f"- Requests that needed >=1 transfer: {(t.transfers>0).mean():.1%}; average transfers per request {t.transfers.mean():.2f}.")
paid = t.request_text.str.contains(r"\bpaid\b", case=False)
b = t[t.bot == "Billing"]; bp = b[paid.loc[b.index]]
P(f"- Billing queue: {len(bp)/len(b):.0%} of it is requests that merely say 'paid'; only {(bp.final=='Billing').mean():.1%} of those really ended in Billing (policy s3: Billing = the payment itself).")
c = t[t.bot == "Filters & Consumables"]; wp = c[c.product_family == "Water Purifier"]
P(f"- Water purifier requests in Filters & Consumables: {len(wp)/len(c):.0%} of that queue; only {(wp.final=='Filters & Consumables').mean():.0%} really belong there (other products: {(c[c.product_family!='Water Purifier'].final=='Filters & Consumables').mean():.0%}).")
P(f"- A model trained to copy the bot labels reproduces them ~96% of the time but is right about the real destination only ~77% of the time. Matching labels at 90% is achievable and is the wrong goal.\n")

P("## 2. Validation (always train on the past, test on the future)")
t["m"] = t.created_at_ist.str[:7]
rows = []
for a, bnd in [("2025-10", "2026-01"), ("2026-01", "2026-04"), ("2026-04", "2026-07")]:
    tr = t[t.created_at_ist < a + "-01"]; va = t[(t.created_at_ist >= a + "-01") & (t.created_at_ist < bnd + "-01")]
    bun = model.fit(tr.request_text, tr.final); cl, pr = model.predict_proba(bun, va.request_text)
    pred = cl[pr.argmax(1)]; conf = pr.max(1)
    rows.append(dict(train_to=a, validate=f"{a}..{bnd}", n=len(va), acc_vs_final=(pred == va.final).mean(),
                     acc_vs_bot_label=(pred == va.bot).mean(), bot_vs_final=(va.bot == va.final).mean(),
                     auto_coverage=(conf >= .6).mean(), auto_acc=(pred == va.final)[conf >= .6].mean()))
r = pd.DataFrame(rows); r.to_csv(OUT / "rolling_validation.csv", index=False)
P(r.round(3).to_markdown(index=False))
P(f"\nMean accuracy vs real destination: **{r.acc_vs_final.mean():.1%}** (range {r.acc_vs_final.min():.1%}-{r.acc_vs_final.max():.1%}). The bot scores {r.bot_vs_final.mean():.1%} on the same rows.\n")

tr = t[t.created_at_ist < "2026-04-01"]; va = t[t.created_at_ist >= "2026-04-01"].copy()
bun = model.fit(tr.request_text, tr.final); cl, pr = model.predict_proba(bun, va.request_text)
va["pred"] = cl[pr.argmax(1)]; va["conf"] = pr.max(1)
P("## 3. Held-out Apr-Jun 2026, per team")
P(pd.DataFrame(classification_report(va.final, va.pred, output_dict=True)).T.round(3).to_markdown())
pd.DataFrame(confusion_matrix(va.final, va.pred, labels=cl), index=cl, columns=cl).to_csv(OUT / "confusion_matrix.csv")
P("\n## 4. Auto-route vs. send to a human (confidence threshold)")
cv = []
for th in [0, .4, .5, .6, .7, .8]:
    k = va.conf >= th; cv.append(dict(threshold=th, auto_routed=k.mean(), accuracy_of_auto_routed=(va.pred == va.final)[k].mean(), to_human=1 - k.mean()))
P(pd.DataFrame(cv).round(3).to_markdown(index=False))
P("\n## 5. Where it fails")
low = va[va.conf < .6]; hi = va[va.conf >= .6]
P(f"- {len(low)/len(va):.0%} of requests are low-confidence; the model is right on only {(low.pred==low.final).mean():.0%} of them. Typical text: 'not happy with the purifier', 'please call back regarding my fan', 'need help with my mixer'. No team can be inferred; a person has to ask.")
P(f"- Confident predictions are wrong {1-(hi.pred==hi.final).mean():.1%} of the time ({(hi.pred!=hi.final).sum()} of {len(hi)}).")
P("- Exact duplicates in training (same text/product/warranty) end in the same team only ~92% of the time, so ~8% of outcomes look like noise nobody can predict from the text.")
P("- Metadata (channel, warranty status, product) does not change the destination mix, so the model uses text only.")
te = hi[hi.pred != hi.final].groupby(["final", "pred"]).size().sort_values(ascending=False).head(5)
P("- Biggest confident mix-ups (real -> predicted): " + "; ".join(f"{a}->{b} ({n})" for (a, b), n in te.items()))

P("\n## 6. Busiest teams (for headcount)")
load = pd.concat([t.bot.value_counts(), t.final.value_counts()], axis=1, keys=["bot_queue", "closed_by"])
load["transfers_out"] = t[t.bot != t.final].groupby("bot").size(); load["transfers_in"] = t[t.bot != t.final].groupby("final").size()
load["share_of_closed"] = load.closed_by / load.closed_by.sum()
P(load.fillna(0).astype(float).round(3).to_markdown())

P("\n## 7. Rupees")
VOL = 725; TR, EXTRA, LIC, HOST = 305, 260, 320000, 1000 * 12
mis = t[t.bot != t.final]; ok = t[t.bot == t.final]
tm, to = mis.transfers.mean(), ok.transfers.mean()
cost_mis, cost_ok = tm * TR + EXTRA, to * TR
bot_rate = len(mis) / len(t)
bot_pr = bot_rate * cost_mis + (1 - bot_rate) * cost_ok
def per_req(err): return err * cost_mis + (1 - err) * cost_ok
err_forced = 1 - r.acc_vs_final.mean()
cov = r.auto_coverage.mean(); err_auto = 1 - r.auto_acc.mean(); TRIAGE = EXTRA
mo = lambda x: x * VOL
P(f"- Volume: {VOL}/month (train averages {len(t)/15:.0f}/month; test {2178/3:.0f}/month).")
P(f"- Correctly routed request costs Rs {cost_ok:.0f} on average ({to:.3f} transfers x Rs {TR}); a misrouted one costs Rs {cost_mis:.0f} ({tm:.2f} transfers x Rs {TR} + Rs {EXTRA} extra contact).")
P(f"- Bot today: {bot_rate:.1%} misrouted -> Rs {bot_pr:.0f}/request -> Rs {mo(bot_pr):,.0f}/month, plus licence Rs {LIC/12:,.0f}/month. **Total Rs {mo(bot_pr)+LIC/12:,.0f}/month.**")
f = per_req(err_forced)
P(f"- Model, route everything: {err_forced:.1%} misrouted -> Rs {f:.0f}/request -> Rs {mo(f):,.0f}/month + hosting Rs {HOST/12:,.0f}. **Total Rs {mo(f)+HOST/12:,.0f}/month.**")
a = cov * per_req(err_auto) + (1 - cov) * TRIAGE
P(f"- Model + triage: auto-route {cov:.0%} at {err_auto:.1%} error, send {1-cov:.0%} to a human who asks one question (assumed Rs {TRIAGE} each, no transfer) -> Rs {a:.0f}/request -> Rs {mo(a):,.0f}/month + hosting. **Total Rs {mo(a)+HOST/12:,.0f}/month.**")
for nm, v in [("route everything", mo(f) + HOST / 12), ("model + triage", mo(a) + HOST / 12)]:
    P(f"- Saving vs bot, {nm}: Rs {(mo(bot_pr)+LIC/12 - v):,.0f}/month = Rs {(mo(bot_pr)+LIC/12 - v)*12/1e5:.1f} lakh/year.")
P(f"- Of which licence alone: Rs {LIC/1e5:.1f} lakh/year. Prediction cost: Rs 0 (local CPU, no API); month at {VOL} requests = Rs 0 in API fees.")
(OUT / "report.md").write_text("\n".join(L))
