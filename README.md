# Credit Card Fraud Detection using Ensemble Learning (Flask)

Models: CART, Random Forest, AdaBoost, Gradient Boosting, Soft-Voting ensemble, Bayesian-model-averaging ensemble.
Pages: Dashboard (metrics + interactive charts), Predict, History (SQLite).

## Install & run
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python generate_data.py        # creates data/transactions.csv   (already included)
python ml.py                   # trains models, prints metrics    (auto-runs on first start)
python app.py                  # open http://127.0.0.1:5000
```
JSON API: `POST /api/predict` with the form fields as JSON.
Charts use a local copy of Chart.js, so the app works offline.

## Files
- `generate_data.py` synthetic dataset · `ml.py` training/inference · `app.py` Flask routes
- `templates/` HTML · `static/` CSS + JS · `docs/` report + viva questions
