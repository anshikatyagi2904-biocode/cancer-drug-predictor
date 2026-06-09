import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cancer Drug Response Predictor", layout="wide")

st.title("🧬 Cancer Cell Line Drug Response Predictor")
st.markdown("Predict **LN_IC50** (drug sensitivity) for a cancer cell line using gene expression data.")

# ── Load model and features ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/xgb_erlotinib.pkl")
    features = joblib.load("models/top_features_erlotinib.pkl")
    return model, features

@st.cache_data
def load_data():
    features = joblib.load("models/top_features_erlotinib.pkl")
    cols = ['ModelID', 'CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'TCGA_DESC', 'PATHWAY_NAME'] + list(features)
    df = pd.read_parquet("hf://datasets/ansh1kac0re/cancer-drug-predictor-data/master_features_slim.parquet", columns=cols)
    return df.drop(columns=[c for c in df.columns if 'Unnamed' in c])

model, top_features = load_model()
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Select Input")
cell_lines = sorted(df['CELL_LINE_NAME'].unique())
selected_cell = st.sidebar.selectbox("Cell Line", cell_lines)

# ── Get expression for selected cell line ─────────────────────────────────────
cell_data = df[df['CELL_LINE_NAME'] == selected_cell].iloc[0]
X_input = pd.DataFrame([cell_data[top_features]], columns=top_features)

# ── Predict ───────────────────────────────────────────────────────────────────
prediction = model.predict(X_input)[0]

# ── Show actual values for this cell line + Erlotinib ─────────────────────────
actual_row = df[(df['CELL_LINE_NAME'] == selected_cell) & (df['DRUG_NAME'] == 'Erlotinib')]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Cell Line", selected_cell)
with col2:
    st.metric("Drug", "Erlotinib")
with col3:
    st.metric("Predicted LN_IC50", f"{prediction:.3f}")

if not actual_row.empty:
    actual = actual_row['LN_IC50'].values[0]
    st.metric("Actual LN_IC50", f"{actual:.3f}", delta=f"{prediction - actual:.3f} error")

# ── Interpretation ────────────────────────────────────────────────────────────
st.divider()
if prediction < 2:
    st.success("🟢 Low LN_IC50 — cell line is likely **sensitive** to Erlotinib")
elif prediction < 5:
    st.warning("🟡 Medium LN_IC50 — moderate sensitivity")
else:
    st.error("🔴 High LN_IC50 — cell line is likely **resistant** to Erlotinib")

st.caption("LN_IC50: log of drug concentration needed to inhibit 50% cell growth. Lower = more sensitive.")

# ── Top predictive genes for this cell line ───────────────────────────────────
st.divider()
st.subheader("Top 10 Predictive Gene Expression Values")
importances = pd.Series(model.feature_importances_, index=top_features)
top10 = importances.nlargest(10).index
gene_vals = cell_data[top10].reset_index()
gene_vals.columns = ['Gene', 'Expression (TPM log)']
st.dataframe(gene_vals, use_container_width=True)