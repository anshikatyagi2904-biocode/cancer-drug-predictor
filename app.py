import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Cancer Drug Response Predictor", layout="wide")

DATA_REPO = "ansh1kac0re/cancer-drug-predictor-data"

SUPPORTED_DRUGS = {
    "Erlotinib": {
        "model_file": "models/xgb_erlotinib.json",
        "features_file": "models/top_features_erlotinib.npy",
    },
    "Olaparib": {
        "model_file": "models/xgb_olaparib.json",
        "features_file": "models/top_features_olaparib.npy",
    },
    "Irinotecan": {
        "model_file": "models/xgb_irinotecan.json",
        "features_file": "models/top_features_irinotecan.npy",
    },
    "5-Fluorouracil": {
        "model_file": "models/xgb_5-fluorouracil.json",
        "features_file": "models/top_features_5-fluorouracil.npy",
    },
    "Palbociclib": {
        "model_file": "models/xgb_palbociclib.json",
        "features_file": "models/top_features_palbociclib.npy",
    },
}

# Gene → biological pathway mapping
# Covers common genes that appear in GDSC expression-based models
GENE_PATHWAYS = {
    "EGFR": "EGFR signalling — cell growth & proliferation",
    "ERBB2": "HER2/ErbB signalling — cell growth",
    "ERBB3": "ErbB signalling — cell survival",
    "KRAS": "RAS/MAPK — cell division control",
    "NRAS": "RAS/MAPK — cell division control",
    "BRAF": "MAPK signalling — tumour growth",
    "PIK3CA": "PI3K/AKT — cell survival & metabolism",
    "PTEN": "PI3K/AKT suppressor — tumour suppressor",
    "AKT1": "PI3K/AKT — cell survival",
    "MYC": "MYC oncogene — gene transcription & growth",
    "MYCN": "MYC family — neuroblastoma amplicon",
    "TP53": "p53 pathway — DNA damage response",
    "RB1": "Cell cycle control — tumour suppressor",
    "CCND1": "Cell cycle — G1/S transition",
    "CDK4": "Cell cycle kinase — G1 progression",
    "CDK6": "Cell cycle kinase — G1 progression",
    "CDKN2A": "Cell cycle inhibitor — tumour suppressor",
    "BCL2": "Apoptosis — anti-cell death",
    "BCL2L1": "Apoptosis — anti-cell death (BCL-XL)",
    "BAX": "Apoptosis — pro-cell death",
    "BRCA1": "DNA repair — homologous recombination",
    "BRCA2": "DNA repair — homologous recombination",
    "ATM": "DNA damage response — checkpoint kinase",
    "CHEK1": "DNA damage checkpoint",
    "CHEK2": "DNA damage checkpoint",
    "VEGFA": "Angiogenesis — blood vessel growth",
    "MET": "MET signalling — invasion & metastasis",
    "FGFR1": "FGFR signalling — cell growth",
    "FGFR2": "FGFR signalling — cell growth",
    "FGFR3": "FGFR signalling — bladder cancer relevant",
    "ALK": "ALK signalling — lung cancer relevant",
    "RET": "RET signalling — thyroid cancer relevant",
    "JAK1": "JAK/STAT — immune & growth signalling",
    "JAK2": "JAK/STAT — haematopoietic signalling",
    "STAT3": "JAK/STAT — transcription factor",
    "MTOR": "mTOR pathway — cell growth & metabolism",
    "TSC1": "mTOR suppressor",
    "TSC2": "mTOR suppressor",
    "VHL": "HIF/hypoxia — tumour suppressor",
    "HIF1A": "Hypoxia response",
    "MDM2": "p53 regulator — oncogene",
    "NOTCH1": "Notch signalling — cell fate",
    "WNT5A": "WNT signalling — development & cancer",
    "CTNNB1": "WNT/beta-catenin — transcription",
    "APC": "WNT suppressor — colorectal cancer",
    "SMAD4": "TGF-beta signalling — tumour suppressor",
    "TGFB1": "TGF-beta — growth inhibition/invasion",
    "FOXO3": "FOXO — apoptosis & stress response",
    "E2F1": "Cell cycle transcription factor",
    "TOP1": "DNA topoisomerase — Irinotecan target",
    "TOP2A": "DNA topoisomerase II — replication",
}

def get_confidence_label(r2):
    if r2 is None:
        return None, None
    if r2 >= 0.6:
        return "✅ Good fit", "green"
    elif r2 >= 0.35:
        return "⚠️ Moderate fit", "orange"
    else:
        return "❌ Weak fit", "red"


@st.cache_data
def load_data():
    parquet_path = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename="master_features_slim.parquet",
    )
    return pd.read_parquet(parquet_path)


@st.cache_data
def load_results():
    results_path = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename="models/drug_model_results.csv",
    )
    return pd.read_csv(results_path)


@st.cache_resource
def load_model(model_file):
    model_path = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=model_file,
    )
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


@st.cache_data
def load_features(features_file):
    features_path = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=features_file,
    )
    return np.load(features_path, allow_pickle=True).tolist()


st.title("Cancer Drug Response Predictor")
st.caption("Demo supports 5 curated drugs with lazy-loaded models")

try:
    df = load_data()
    results_df = load_results()

    available_df = df[df["DRUG_NAME"].isin(SUPPORTED_DRUGS.keys())].copy()

    if available_df.empty:
        st.error("No supported drugs found in dataset.")
        st.stop()

    tumour_types = sorted(available_df["TCGA_DESC"].dropna().unique())
    selected_tumour = st.selectbox("Select tumour type", tumour_types)

    tumour_df = available_df[available_df["TCGA_DESC"] == selected_tumour].copy()

    drugs_for_tumour = sorted(tumour_df["DRUG_NAME"].dropna().unique())
    selected_drug = st.selectbox("Select drug", drugs_for_tumour)

    drug_df = tumour_df[tumour_df["DRUG_NAME"] == selected_drug].copy()

    cell_lines = sorted(drug_df["CELL_LINE_NAME"].dropna().unique())
    selected_cell_line = st.selectbox("Select cell line", cell_lines)

    row = drug_df[drug_df["CELL_LINE_NAME"] == selected_cell_line].iloc[0:1].copy()

    model_info = SUPPORTED_DRUGS[selected_drug]
    model = load_model(model_info["model_file"])
    model_features = load_features(model_info["features_file"])

    X = row.reindex(columns=model_features, fill_value=0).fillna(0)
    pred = float(model.predict(X)[0])

    actual = float(row["LN_IC50"].iloc[0]) if "LN_IC50" in row.columns else None

    result_row = results_df[results_df["drug"] == selected_drug]
    r2_value = float(result_row["r2"].iloc[0]) if not result_row.empty else None
    rmse_value = float(result_row["rmse"].iloc[0]) if not result_row.empty else None

    # — Metrics row ————————————————————————————————
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted LN_IC50", f"{pred:.4f}")
    with col2:
        if actual is not None:
            st.metric("Actual LN_IC50", f"{actual:.4f}")
    with col3:
        if r2_value is not None:
            st.metric("Model R²", f"{r2_value:.3f}")

    if rmse_value is not None:
        st.caption(f"Model RMSE: {rmse_value:.3f}")

    # — Model confidence indicator ————————————————————————————————
    st.divider()
    st.subheader("Model Confidence")

    confidence_label, confidence_color = get_confidence_label(r2_value)

    if confidence_label:
        if confidence_color == "green":
            st.success(f"{confidence_label} — predictions for **{selected_drug}** are reliable. R² = {r2_value:.3f}")
        elif confidence_color == "orange":
            st.warning(f"{confidence_label} — predictions for **{selected_drug}** show real signal but have meaningful error. R² = {r2_value:.3f}")
        else:
            st.error(f"{confidence_label} — model for **{selected_drug}** explains little variance. Treat prediction as indicative only. R² = {r2_value:.3f}")

        with st.expander("What does this mean?"):
            st.markdown("""
**R² (R-squared)** measures how well the model explains the variation in drug response across cell lines.

| R² range | Meaning |
|---|---|
| ≥ 0.6 | Good — model captures most of the biological signal |
| 0.35 – 0.6 | Moderate — useful signal, but not highly precise |
| < 0.35 | Weak — prediction is a rough estimate only |

A weak fit doesn't mean the biology is wrong — it may mean gene expression alone isn't sufficient to predict this drug's response, or the drug has a complex mechanism.
""")

    # — Selection details ————————————————————————————————
    st.divider()
    st.subheader("Selection Details")
    details_cols = [c for c in ["CELL_LINE_NAME", "TCGA_DESC", "PATHWAY_NAME", "DRUG_NAME"] if c in row.columns]
    st.dataframe(row[details_cols], use_container_width=True)

    # — Top gene importance with pathway labels ————————————————————————————————
    st.divider()
    st.subheader("Top Gene Importance")
    st.caption("What the model relied on most to make this prediction, and what each gene does biologically.")

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Gene": model_features,
        "Importance Score": importances
    }).sort_values("Importance Score", ascending=False).head(20).reset_index(drop=True)

    importance_df["Biological Role"] = importance_df["Gene"].map(
        lambda g: GENE_PATHWAYS.get(g, "—")
    )

    # highlight genes we have pathway info for
    known = importance_df["Biological Role"] != "—"
    st.dataframe(
        importance_df,
        use_container_width=True,
        column_config={
            "Importance Score": st.column_config.ProgressColumn(
                "Importance Score",
                min_value=0,
                max_value=float(importance_df["Importance Score"].max()),
                format="%.4f",
            )
        }
    )

    known_count = known.sum()
    if known_count > 0:
        st.caption(f"{known_count} of top 20 genes have known pathway annotations. '—' means the gene is not in the current annotation dictionary.")

except Exception as e:
    st.error("App failed")
    st.exception(e)