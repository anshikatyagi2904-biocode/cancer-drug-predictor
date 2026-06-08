import pandas as pd
import numpy as np

# ── 1. Load all data ──────────────────────────────────────────────────────────
print("Loading data...")
gdsc = pd.read_csv("data/gdsc2_raw.csv")
expr = pd.read_csv("data/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
model = pd.read_csv("data/Model.csv")
mut = pd.read_csv("data/OmicsSomaticMutations.csv", low_memory=False)

# ── 2. Fix expression index ───────────────────────────────────────────────────
# First column is ModelConditionID, ModelID is a column — find it
print(f"Expression columns 0-5: {expr.columns[:5].tolist()}")
# Set ModelID as index
expr = expr.set_index('ModelID')
# Drop non-gene metadata columns
meta_cols = ['SequencingID', 'ModelConditionID', 'IsDefaultEntryForMC', 'IsDefaultEntryForModel']
expr = expr.drop(columns=[c for c in meta_cols if c in expr.columns])
print(f"Expression after cleanup: {expr.shape}")  # should be (~1775, ~19000)

# ── 3. Build mutation binary matrix ──────────────────────────────────────────
print("\nBuilding mutation matrix...")
# Keep only damaging mutations
damaging = mut[mut['VepImpact'].isin(['HIGH', 'MODERATE'])]
# Pivot: rows = ModelID, cols = gene, value = 1 if mutated
mut_matrix = damaging.groupby(['ModelID', 'HugoSymbol']).size().unstack(fill_value=0)
mut_matrix = (mut_matrix > 0).astype(int)  # binary
print(f"Mutation matrix: {mut_matrix.shape}")

# ── 4. Map GDSC cell line names to DepMap ModelID ────────────────────────────
print("\nMapping cell line names...")
name_to_id = dict(zip(model['CellLineName'].str.upper(), model['ModelID']))
gdsc['ModelID'] = gdsc['CELL_LINE_NAME'].str.upper().map(name_to_id)
gdsc_mapped = gdsc.dropna(subset=['ModelID'])
print(f"GDSC rows with ModelID: {len(gdsc_mapped)} / {len(gdsc)}")

# ── 5. Merge everything ───────────────────────────────────────────────────────
print("\nMerging...")
df = gdsc_mapped[['ModelID', 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'TCGA_DESC', 'PATHWAY_NAME']]
df = df.merge(expr, on='ModelID', how='inner')
df = df.merge(mut_matrix, on='ModelID', how='left')
df = df.fillna(0)

print(f"\nFinal master table shape: {df.shape}")
print(f"Columns sample: {df.columns[:10].tolist()}")
print(f"LN_IC50 range: {df['LN_IC50'].min():.2f} to {df['LN_IC50'].max():.2f}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])
df['TCGA_DESC'] = df['TCGA_DESC'].astype(str)
df['PATHWAY_NAME'] = df['PATHWAY_NAME'].astype(str)
df.to_parquet("data/master_features.parquet", index=False)
print("\nSaved to data/master_features.parquet")
print(f"Shape confirmed: {df.shape}")