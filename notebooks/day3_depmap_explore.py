import pandas as pd
import numpy as np

# ── 1. Expression Data ────────────────────────────────────────────────────────
print("Loading expression data...")
expr = pd.read_csv("data/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
print(f"Shape: {expr.shape}")
print(f"Cell lines (first 5): {expr.index[:5].tolist()}")
print(f"Genes (first 5): {expr.columns[:5].tolist()}")
numeric_cols = expr.select_dtypes(include='number')
print(f"Value range: {numeric_cols.values.min():.2f} to {numeric_cols.values.max():.2f}")
print(f"Missing values: {expr.isnull().sum().sum()}")

# ── 2. Mutation Data ──────────────────────────────────────────────────────────
print("\nLoading mutation data...")
mut = pd.read_csv("data/OmicsSomaticMutations.csv")
print(f"Shape: {mut.shape}")
print(f"Columns: {mut.columns.tolist()}")
print(f"Unique cell lines: {mut['ModelID'].nunique()}")
print(f"Unique genes: {mut['HugoSymbol'].nunique()}")

# ── 3. Model Mapping ──────────────────────────────────────────────────────────
print("\nLoading Model.csv...")
model = pd.read_csv("data/Model.csv")
print(f"Shape: {model.shape}")
print(f"Columns: {model.columns.tolist()}")
print(model[['ModelID', 'CellLineName', 'OncotreePrimaryDisease']].head(5))

# ── 4. Overlap Check with GDSC ───────────────────────────────────────────────
print("\nChecking GDSC overlap...")
gdsc = pd.read_csv("data/gdsc2_raw.csv")
gdsc_lines = set(gdsc['CELL_LINE_NAME'].str.upper())
model_names = set(model['CellLineName'].str.upper())
overlap = gdsc_lines & model_names
print(f"GDSC cell lines: {len(gdsc_lines)}")
print(f"DepMap cell lines: {len(model_names)}")
print(f"Overlap: {len(overlap)}")
print(f"Example matches: {list(overlap)[:5]}")