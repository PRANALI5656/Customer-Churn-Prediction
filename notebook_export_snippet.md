# Notebook to API handoff

The exploratory notebook is retained at `notebooks/EDA.ipynb`. Model training
and serving preprocessing now live in `src/model.py`; the notebook's selected
XGBoost parameters, stratified 80/20 split, and recall-focused 0.3 decision
threshold are recorded there and in `artifacts/metadata.json`.

Run this from the project root after changing the data or model settings in
the notebook:

```powershell
python -m scripts.train_model
```

The command regenerates `artifacts/churn_xgb.joblib` and its metadata. The
joblib contains both fitted preprocessing and the classifier, so API requests
must provide the original customer feature fields rather than notebook-only
one-hot columns. Do not load untrusted joblib files; loading a joblib artifact
can execute code.
