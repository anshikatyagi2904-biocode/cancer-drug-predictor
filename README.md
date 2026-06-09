# Cancer Cell Line Drug Response Predictor

> ML-powered web app to predict cancer cell line drug sensitivity using GDSC and DepMap data.

🔗 **Live Demo:** [huggingface.co/spaces/ansh1kac0re/cancer-drug-response](https://huggingface.co/spaces/ansh1kac0re/cancer-drug-response)

---

## What This Does

Given a tumour type, drug, and cell line — the app predicts the **LN_IC50** (log-transformed drug concentration needed to inhibit 50% of cell growth).

User flow: `Select Tumour Type → Select Drug → Select Cell Line → View Prediction`

Outputs:
- Predicted LN_IC50
- Actual LN_IC50 (from GDSC data, where available)
- Model R² and RMSE
- Top gene importance features driving the prediction

---

## Data Sources

- **GDSC2** (Genomics of Drug Sensitivity in Cancer) — drug response (LN_IC50 values)
- **DepMap** — cancer cell line gene expression profiles

---

## Model

- Algorithm: **XGBoost Regressor** (outperforms Random Forest baseline)
- Features: Gene expression values per cell line
- Target: LN_IC50 per drug
- Trained: one model per drug (286 drugs total in training)
- Deployed: 5 curated drugs in v1 demo (lazy-loaded on drug selection)

### V1 Demo Drugs — Model Performance

| Drug | R² | RMSE |
|---|---|---|
| Erlotinib | — | — |
| Olaparib | — | — |
| Irinotecan | — | — |
| 5-Fluorouracil | 0.444 | 1.360 |
| Palbociclib | — | — |

> **Note on model quality:** R² values vary by drug. The deployed demo is a proof-of-concept. Models with low R² should be interpreted carefully — the predicted LN_IC50 reflects general trends, not clinical dosing.

---

## Why Only 5 Drugs in the Demo?

All 286 models were trained. The v1 app deploys 5 curated drugs to keep memory usage within Hugging Face free-tier limits (lazy-loading one model at a time). Expanding to more drugs is planned for v2.

---

## Repo Structure

```
cancer-drug-predictor/
├── data/               # Data loading scripts, GDSC2 exploration
├── models/             # Trained XGBoost models (.pkl) + deploy-safe files (.json, .npy)
├── notebooks/          # Exploratory analysis, model comparison (Day 1–9)
├── app.py              # Streamlit app (deployed on HF Spaces)
├── data_prep.py        # Data cleaning and feature engineering
├── train_models.py     # XGBoost training loop across 286 drugs
└── requirements.txt    # Python dependencies
```

---

## Run Locally

```bash
git clone https://github.com/anshikatyagi2904-biocode/cancer-drug-predictor
cd cancer-drug-predictor
pip install -r requirements.txt
streamlit run app.py
```

> The app loads model files from Hugging Face dataset on startup. Internet connection required.

---

## Key Notebooks

| Notebook | What it shows |
|---|---|
| Day 6 | XGBoost vs RF baseline — XGBoost wins on R² |
| Day 8 | Full training loop: 286 drugs, results saved |
| Day 9 | README methodology writeup |

---

## Limitations

- Models trained on cancer cell lines (in vitro) — not clinical data
- Gene expression features only; no mutation, CNV, or methylation data
- Some drugs have poor model fit (low R²) — prediction confidence varies
- V1 limited to 5 drugs

---

## Author

**Anshika** — aspiring bioinformatician  
Built as a self-directed 10-day ML project  
GitHub: [@anshikatyagi2904-biocode](https://github.com/anshikatyagi2904-biocode)

---

## What's Next (V2 Roadmap)

- [ ] Expand to 20+ drugs
- [ ] Add predicted vs actual scatter plot
- [ ] Include mutation features
- [ ] Add model confidence intervals

