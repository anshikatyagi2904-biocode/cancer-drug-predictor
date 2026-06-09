import pandas as pd
import numpy as np
import joblib
import os
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

print("Loading master features...")
t = time.time()
df_full = pd.read_parquet("data/master_features.parquet")
gene_cols = [c for c in df_full.columns if c not in 
             ['ModelID','CELL_LINE_NAME','DRUG_NAME','LN_IC50','TCGA_DESC',
              'PATHWAY_NAME','PUTATIVE_TARGET']]
print(f"Loaded in {time.time()-t:.1f}s — {len(gene_cols)} gene columns")

drug_counts = df_full.groupby('DRUG_NAME').size()
drugs = drug_counts[drug_counts >= 100].index.tolist()
print(f"Training on {len(drugs)} drugs with >=100 cell lines")

os.makedirs("models", exist_ok=True)
results = []

for i, drug in enumerate(drugs):
    drug_df = df_full[df_full['DRUG_NAME'] == drug].dropna(subset=['LN_IC50'])
    expr = drug_df[gene_cols]
    top_genes = expr.var().nlargest(1000).index.tolist()
    X = drug_df[top_genes].values
    y = drug_df['LN_IC50'].values
    if len(X) < 50:
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    safe_name = drug.replace("/", "_").replace(" ", "_")
    joblib.dump(model, f"models/xgb_{safe_name}.pkl")
    joblib.dump(top_genes, f"models/top_features_{safe_name}.pkl")
    results.append({'drug': drug, 'n_samples': len(X), 'r2': round(r2,4), 'rmse': round(rmse,4)})
    print(f"[{i+1}/{len(drugs)}] {drug}: R²={r2:.3f}, RMSE={rmse:.3f}, n={len(X)}")

results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
results_df.to_csv("models/drug_model_results.csv", index=False)
print(f"\nDone. Top 10 most predictable drugs:")
print(results_df.head(10).to_string())
