# Kestrel Home: service-request router

Routes a customer message to one of Kestrel's seven teams, says how sure it is, and explains why in plain words. Runs on a laptop, CPU only, **no API key, no paid calls, no internet**.

## Run it (clean machine, Python 3.10+)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                           # then open http://localhost:8000
```

The trained model (`model/router.joblib`, 140 KB) is included, so nothing else is needed. If it is missing, the service still starts and answers `503` with a polite message.

**Endpoint**

```bash
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"request_text": "i paid extra for installation, installer did not turn up"}'
```
Returns `team`, `confidence`, `auto_route` (true/false), `send_to` ("Triage desk (human)" when unsure), `alternatives`, and `reasons` (sentences a service-desk employee can read). `request_text` is required; other fields are accepted and ignored. Bad input returns `400` with an explanation. `GET /health` shows whether the model loaded.

**The screen** is `http://localhost:8000`: paste a message, press Route.

## Reproduce the numbers

The customer data is **not in this repo** (Ops Policy s10 forbids putting it in public repositories). To retrain, put `train.csv`, `resolution_log.csv`, `test_unlabelled.csv` in `./data/`, then:

```bash
python train.py            # fits model/router.joblib
python evaluate.py         # writes evidence/report.md + csvs
python predict_batch.py    # writes predictions.csv (+ predictions_detail.csv with confidence)
```

## What it is

TF-IDF word 1-3-grams + logistic regression, trained on **where each request actually ended up** (`final_team` in the resolution log), not on the bot's label. Product names and order numbers are masked; legacy Zoho text with broken characters ("Ã©") is repaired. Anything under 60% confidence is not auto-routed.

## Decisions you should know about

- **Target = final team, not `team_label`.** The bot's label agrees with the final team only 77% of the time. A model can copy the bot at ~96% and would then inherit all its mistakes. See `evidence/report.md`.
- **Renamed teams are merged** (Installations -> Installs & Demo, Consumables -> Filters & Consumables). Predictions use the current names, as all test rows are after 15 Jan 2026.
- **Text only.** Channel, product and warranty status did not change where requests end up.
- **Output for every row, plus a "don't auto-route" flag.** `predictions.csv` has a best guess for every request as asked. About 16% have no usable detail; the service sends those to a person.
- **Zoho resolution times were never used.** They are in UTC, not IST (26% of Zoho rows resolve before they were created), so any resolution-time analysis would be wrong unless shifted by 5h30m.

## Known limits

- About 16% of messages ("please call back about my purifier") do not say what is wrong. No model can route these; the model is right on ~21% of them.
- ~8% of outcomes look random even for identical wording, which caps what any text model can reach (~92%).
- Trained on 15 months of one company's wording; if customers change how they write, accuracy will drift. Re-run `evaluate.py` monthly on fresh closed requests.
- Triage cost in the rupee model (Rs 260/request) is an assumption, not a Kestrel figure.
