"""Train, load and explain the routing model. Pure scikit-learn; no network, no API key."""
import re
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .text import clean

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "router.joblib"
AUTO_THRESHOLD = 0.60   # below this the request goes to a human triage queue (see evidence/)
PAY_WORDS = re.compile(r"\bpaid\b|\bpayment\b|\bpay\b", re.I)


def fit(texts, labels):
    vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    clf = LogisticRegression(C=10, max_iter=3000)
    X = vec.fit_transform([clean(t) for t in texts])
    clf.fit(X, labels)
    return {"vec": vec, "clf": clf, "threshold": AUTO_THRESHOLD}


def save(bundle, path=MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load(path=MODEL_PATH):
    return joblib.load(path)


def predict_proba(bundle, texts):
    X = bundle["vec"].transform([clean(t) for t in texts])
    return bundle["clf"].classes_, bundle["clf"].predict_proba(X)


def explain(bundle, raw_text, top_k=3):
    """Decision plus the reasons a service-desk employee can read."""
    vec, clf, thr = bundle["vec"], bundle["clf"], bundle["threshold"]
    X = vec.transform([clean(raw_text)])
    proba = clf.predict_proba(X)[0]
    order = np.argsort(proba)[::-1]
    team, conf = str(clf.classes_[order[0]]), float(proba[order[0]])
    names = vec.get_feature_names_out()
    row = X.tocoo()
    contrib = sorted(((clf.coef_[order[0]][j] * v, names[j]) for j, v in zip(row.col, row.data)), reverse=True)
    phrases = []
    for c, p in contrib:                       # strongest first; skip fragments of a phrase already chosen
        if c > 0.15 and p.strip() != "prod" and not any(p in q for q in phrases):
            phrases.append(p)
        if len(phrases) == top_k + 2:
            break

    phrases = [p for p in phrases if not any(p != q and p in q for q in phrases)]
    phrases = [p.replace("prod", "[product]") for p in phrases]
    auto = conf >= thr
    reasons = []
    if phrases and auto:
        reasons.append("Wording that points to %s: %s." % (team, ", ".join('"%s"' % p for p in phrases)))
    if not auto:
        reasons.append("The message does not say what the problem is (confidence %.0f%%). "
                       "Do not auto-route: send to the triage desk to ask the customer one question." % (conf * 100))
    if PAY_WORDS.search(raw_text) and team != "Billing":
        reasons.append("Customer mentions paying, but the problem is not the payment itself, "
                       "so it is not Billing (Ops Policy section 3).")
    if auto and not phrases:
        reasons.append("Matches the pattern of past requests that ended in %s." % team)
    alt = [{"team": str(clf.classes_[i]), "confidence": round(float(proba[i]), 3)} for i in order[1:3]]
    return {"team": team, "confidence": round(conf, 3), "auto_route": bool(auto),
            "send_to": team if auto else "Triage desk (human)", "alternatives": alt, "reasons": reasons}
