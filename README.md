# Cancer Cell Line Drug Response Predictor

Predicts drug sensitivity (LN_IC50) for cancer cell lines using gene expression data from DepMap and GDSC2.

## What it does
Given a cancer cell line, the app predicts how sensitive it is to a drug — using only gene expression (RNA-seq TPM) as input. Lower LN_IC50 = more sensitive = drug is effective at lower concentration.

## Data Sources
- **GDSC2** (Genomics of Drug Sensitivity in Cancer): 242,036 drug response measurements, 286 drugs, 969 cell lines
- **DepMap 26Q1**: Gene expression (19,216 genes, 1,775 cell lines), somatic mutations, cell line metadata
- **Overlap**: 824 cell lines with both expression and drug response data

## Method
1. Built master feature table: 156,307 rows × 39,368 columns (cell line × drug combinations with expression features)
2. Per drug: selected top 1,000 most variable genes as features
3. Trained XGBoost regressor (200 estimators, max_depth=4, lr=0.05)
4. Evaluated on 20% held-out test set

## Results
Trained models for 286 drugs. Top 10 most predictable:

| Drug | R² | RMSE | n |
|------|----|------|---|
| Acetalax | 0.731 | 1.037 | 956 |
| CHIR-99021 | 0.666 | 0.977 | 117 |
| Irinotecan | 0.613 | 1.325 | 614 |
| AZD5991 | 0.607 | 1.867 | 477 |
| Olaparib | 0.597 | 0.792 | 615 |
| Nutlin-3a | 0.594 | 1.152 | 616 |
| Camptothecin | 0.593 | 1.160 | 616 |
| Rucaparib | 0.564 | 0.679 | 601 |
| Tozasertib | 0.555 | 1.475 | 152 |
| Vorinostat | 0.552 | 0.808 | 610 |

Erlotinib (EGFR inhibitor, used as initial test case): R²=0.269, RMSE=1.182. EGFR gene expression was the top predictive feature — biologically consistent.

## Stack
Python, pandas, scikit-learn, XGBoost, Streamlit, joblib, pyarrow

## Repo Structure
## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Note on data files
`data/` files are not on GitHub due to size. Download from:
- GDSC2: https://www.cancerrxgene.org/downloads/bulk_download
- DepMap 26Q1: https://depmap.org/portal/download/
