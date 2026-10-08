"""Kestrel routing service.  python app.py  ->  http://localhost:8000"""
from flask import Flask, jsonify, request, render_template
from kestrel import model

app = Flask(__name__)
try:
    BUNDLE = model.load()
except Exception as e:           # missing/corrupt model file: start anyway, answer politely
    BUNDLE = None
    print("WARNING: model not loaded (%s). Run `python train.py` with data in ./data." % e)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify(status="ok" if BUNDLE else "model_missing")

@app.post("/route")
def route():
    if BUNDLE is None:
        return jsonify(error="The routing model is not installed. Ask IT to run train.py (see README)."), 503
    rec = request.get_json(silent=True)
    if not isinstance(rec, dict) or not str(rec.get("request_text", "")).strip():
        return jsonify(error="Send JSON like {\"request_text\": \"...\"}. request_text is required; "
                             "product_family, warranty_status, channel are optional and ignored by the model."), 400
    out = model.explain(BUNDLE, str(rec["request_text"])[:2000])
    out["request_id"] = rec.get("request_id")
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
