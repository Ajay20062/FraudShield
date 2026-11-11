"""Credit Card Fraud Detection using Ensemble Learning - Flask application."""
import json
import os
import sqlite3
from datetime import datetime

from flask import Flask, g, jsonify, redirect, render_template, request, url_for

import ml

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "history.db")
app = Flask(__name__)

FIELDS = {  # name: (type, min, max)
    "amount": (float, 0.01, 1_000_000), "hour": (int, 0, 23), "day_of_week": (int, 0, 6),
    "distance_from_home_km": (float, 0, 20000), "distance_from_last_txn_km": (float, 0, 20000),
    "txn_count_24h": (int, 0, 500), "avg_amount_30d": (float, 0.01, 1_000_000),
    "card_present": (int, 0, 1), "international": (int, 0, 1),
}


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
        g.db.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, input TEXT,
            probability REAL, verdict TEXT, risk TEXT, details TEXT)""")
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d is not None:
        d.close()


def parse_form(data):
    row = {}
    for name, (typ, lo, hi) in FIELDS.items():
        try:
            v = typ(data.get(name))
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a valid number")
        if not lo <= v <= hi:
            raise ValueError(f"'{name}' must be between {lo} and {hi}")
        row[name] = v
    cat = data.get("merchant_category")
    if cat not in ml.CATEGORIES:
        raise ValueError("Invalid merchant category")
    row["merchant_category"] = cat
    return row


@app.route("/")
def dashboard():
    ml.load()
    with open(ml.METRICS_PATH) as f:
        info = json.load(f)
    return render_template("dashboard.html", info=info, page="dashboard")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, error, row = None, None, None
    if request.method == "POST":
        try:
            row = parse_form(request.form)
            result = ml.predict(row)
            verdict = "FRAUD" if result["is_fraud"] else "LEGITIMATE"
            db().execute(
                "INSERT INTO history (created,input,probability,verdict,risk,details) VALUES (?,?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(row),
                 result["final_probability"], verdict, result["risk"], json.dumps(result["probabilities"])))
            db().commit()
        except ValueError as e:
            error = str(e)
    return render_template("predict.html", categories=ml.CATEGORIES, result=result,
                           error=error, row=row, page="predict")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(ml.predict(parse_form(request.get_json(force=True, silent=True) or {})))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/history")
def history():
    rows = []
    for r in db().execute("SELECT * FROM history ORDER BY id DESC LIMIT 200"):
        rows.append({**dict(r), "input": json.loads(r["input"])})
    return render_template("history.html", rows=rows, page="history")


@app.route("/history/clear", methods=["POST"])
def clear_history():
    db().execute("DELETE FROM history")
    db().commit()
    return redirect(url_for("history"))


@app.route("/retrain", methods=["POST"])
def retrain():
    ml.train()
    ml._bundle = None
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    ml.load()
    app.run(debug=True)
