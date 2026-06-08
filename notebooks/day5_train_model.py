import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# ── 1. Load master table ──────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("data/master_features.parquet")
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])

# ── 2. Pick one drug to start ─────────────────────────────────────────────────
DRUG = "Erlotinib"
df_drug = df[df['DRUG_NAME'] == DRUG].copy()
print(f"Samples for {DRUG}: {len(df_drug)}")

# ── 3. Features and target ────────────────────────────────────────────────────
drop_cols = ['ModelID', 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'TCGA_DESC', 'PATHWAY_NAME']
X = df_drug.drop(columns=drop_cols)
y = df_drug['LN_IC50']
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

# ── 4. Reduce features — top 500 most variable genes ─────────────────────────
variances = X.var()
top_features = variances.nlargest(500).index
X = X[top_features]
print(f"After feature selection: {X.shape}")

# ── 5. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ── 6. Train Random Forest ────────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nResults for {DRUG}:")
print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# ── 8. Top important features ─────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=top_features)
print(f"\nTop 10 predictive genes:")
print(importances.nlargest(10))

# ── 9. Save model ─────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, f"models/rf_{DRUG.lower()}.pkl")
print(f"\nModel saved to models/rf_{DRUG.lower()}.pkl")