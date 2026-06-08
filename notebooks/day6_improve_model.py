import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("data/master_features.parquet")
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])

DRUG = "Erlotinib"
df_drug = df[df['DRUG_NAME'] == DRUG].copy()

# ── 2. Expression-only features ───────────────────────────────────────────────
# Expression columns are numeric gene names like "EGFR (1956)"
meta_cols = ['ModelID', 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'TCGA_DESC', 'PATHWAY_NAME']
all_features = [c for c in df_drug.columns if c not in meta_cols]

# Expression cols have format "GENESYMBOL (number)"
expr_cols = [c for c in all_features if '(' in str(c) and ')' in str(c)]
print(f"Expression features: {len(expr_cols)}")

X = df_drug[expr_cols]
y = df_drug['LN_IC50']

# ── 3. Top 1000 most variable expression features ─────────────────────────────
top_features = X.var().nlargest(1000).index
X = X[top_features]
print(f"After selection: {X.shape}")

# ── 4. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. XGBoost ────────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.3,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost  R²: {r2_xgb:.3f}  RMSE: {rmse_xgb:.3f}")

# ── 6. Random Forest with expression only ────────────────────────────────────
print("Training Random Forest (expression only)...")
rf = RandomForestRegressor(n_estimators=200, max_features=0.3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"RF       R²: {r2_rf:.3f}  RMSE: {rmse_rf:.3f}")

# ── 7. Save best model ────────────────────────────────────────────────────────
if r2_xgb >= r2_rf:
    best, best_name = xgb, "xgb"
    print(f"\nBest: XGBoost")
else:
    best, best_name = rf, "rf_v2"
    print(f"\nBest: Random Forest")

joblib.dump(best, f"models/{best_name}_erlotinib.pkl")
joblib.dump(top_features.tolist(), "models/top_features_erlotinib.pkl")
print(f"Saved models/{best_name}_erlotinib.pkl")

# ── 8. Top genes ──────────────────────────────────────────────────────────────
if best_name == "xgb":
    importances = pd.Series(xgb.feature_importances_, index=top_features)
else:
    importances = pd.Series(rf.feature_importances_, index=top_features)
print(f"\nTop 10 genes:")
print(importances.nlargest(10))