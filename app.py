import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Cancer Drug Response Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Clean academic styling ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base font and background */
    html, body, [class*="css"] {
        font-family: 'Georgia', serif;
        background-color: #fafafa;
        color: #1a1a1a;
    }

    /* Main container padding */
    .block-container {
        padding: 2.5rem 4rem 2rem 4rem;
        max-width: 1100px;
    }

    /* Title */
    h1 {
        font-size: 1.9rem;
        font-weight: 600;
        color: #1a1a1a;
        border-bottom: 2px solid #2c5f8a;
        padding-bottom: 0.5rem;
        margin-bottom: 0.2rem;
    }

    /* Section headers */
    h2, h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c5f8a;
        margin-top: 1.5rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Selectbox label */
    label {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 0.85rem;
        color: #444;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #dde3ea;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #666;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1a1a1a;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #ffffff;
        border: 1px solid #ccd3db;
        border-radius: 4px;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 0.9rem;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e0e5ea;
        margin: 1.5rem 0;
    }

    /* Caption */
    .stCaption, small {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 0.8rem;
        color: #777;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 0.85rem;
        color: #2c5f8a;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #dde3ea;
        border-radius: 4px;
    }

    /* Success/warning/error boxes */
    .stAlert {
        border-radius: 4px;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 0.88rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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
    "JAK1": "JAK/STAT — immune & growth signalling",
    "JAK2": "JAK/STAT — haematopoietic signalling",
    "STAT3": "JAK/STAT — transcription factor",
    "MTOR": "mTOR pathway — cell growth & metabolism",
    "VHL": "HIF/hypoxia — tumour suppressor",
    "MDM2": "p53 regulator — oncogene",
    "NOTCH1": "Notch signalling — cell fate",
    "CTNNB1": "WNT/beta-catenin — transcription",
    "APC": "WNT suppressor — colorectal cancer",
    "SMAD4": "TGF-beta signalling — tumour suppressor",
    "TOP1": "DNA topoisomerase — Irinotecan target",
    "TOP2A": "DNA topoisomerase II — replication",
    "AGR2": "Endoplasmic reticulum — protein folding; overexpressed in breast, lung & GI cancers",
    "C19orf33": "Chromosome 19 open reading frame — stress response; linked to drug resistance",
    "CD74": "MHC class II chaperone — immune signalling; highly expressed in haematological cancers",
    "EPCAM": "Epithelial cell adhesion molecule — tumour marker; drives epithelial cancer proliferation",
    "FN1": "Fibronectin — extracellular matrix & cell adhesion; promotes invasion & metastasis",
    "FXYD3": "FXYD domain protein — ion transport regulator; overexpressed in colon & breast cancer",
    "KRT17": "Keratin 17 — cytoskeletal structural protein; marker of basal-like tumours",
    "KRT18": "Keratin 18 — epithelial cytoskeleton; loss linked to chemotherapy resistance",
    "KRT19": "Keratin 19 — epithelial marker; elevated in circulating tumour cells",
    "KRT7": "Keratin 7 — epithelial cytoskeleton; used to classify carcinoma origin",
    "KRT8": "Keratin 8 — epithelial cytoskeleton; protects cancer cells from apoptosis",
    "RPS4Y1": "Ribosomal protein S4 Y-linked — protein synthesis; Y-chromosome gene",
    "S100A14": "S100 calcium-binding protein — cell motility & invasion; upregulated in epithelial cancers",
    "S100A6": "S100 calcium-binding protein — cell cycle & apoptosis regulation",
    "S100P": "S100 calcium-binding protein — drug resistance signal; high in pancreatic & breast cancer",
    "SLPI": "Secretory leukocyte protease inhibitor — immune modulation; promotes tumour immune evasion",
    "SPARC": "Extracellular matrix protein — affects drug delivery & tumour microenvironment",
    "TACSTD2": "Tumour-associated calcium signal transducer 2 — cell adhesion; therapeutic target",
    "TGFBI": "TGF-beta induced protein — extracellular matrix; linked to chemotherapy resistance",
    "UCHL1": "Ubiquitin carboxyl-terminal hydrolase L1 — protein degradation pathway",
    "VIM": "Vimentin — mesenchymal cytoskeleton; key marker of epithelial-to-mesenchymal transition",
    "IKZF1": "Ikaros transcription factor — lymphocyte development regulator; mutated in leukaemia & ALL",
    "PTPN7": "Protein tyrosine phosphatase — negative MAPK regulator; expressed in haematopoietic cancers",
    "PXDC1": "PX domain containing protein — vesicle trafficking & signalling",
    "RTL8C": "Retrotransposon-like protein — epigenetic gene regulation; cancer-testis antigen family",
    "COL7A1": "Collagen type VII alpha 1 — basement membrane structure; altered in epithelial cancers",
    "EVA1A": "Epithelial V-like antigen 1 — autophagy & apoptosis regulator; tumour suppressor",
    "INA": "Alpha-internexin — neuronal intermediate filament; expressed in neuroendocrine tumours",
    "APP": "Amyloid precursor protein — cell signalling & adhesion; expressed across multiple cancer types",
    "SGCE": "Sarcoglycan epsilon — dystrophin-associated complex; imprinted gene chr7q21",
    "PPIC": "Peptidyl-prolyl isomerase C (Cyclophilin C) — protein folding; implicated in drug resistance",
    "WAS": "Wiskott-Aldrich syndrome protein — actin cytoskeleton regulation; haematopoietic cells",
    "NCKAP1L": "NCK-associated protein — actin polymerisation; immune cell migration",
    "SASH3": "SAM and SH3 domain protein — immune signalling scaffold; B & T cell function",
    "MT1F": "Metallothionein 1F — heavy metal detox & oxidative stress; induced by chemotherapy",
    "PCDH1": "Protocadherin 1 — cell adhesion; epithelial barrier function",
    "SDC4": "Syndecan-4 — heparan sulfate proteoglycan; regulates cell adhesion & migration",
    "PTPRF": "Protein tyrosine phosphatase receptor F — cell adhesion & growth factor signalling",
    "FXYD5": "Dysadherin — Na+/K+ ATPase regulator; promotes cancer invasion",
    "PLEK2": "Pleckstrin 2 — actin cytoskeleton remodelling; platelet & immune cell signalling",
    "CFI": "Complement factor I — immune complement regulation; tumour immune evasion",
    "MATN2": "Matrilin-2 — extracellular matrix assembly; downregulated in some cancers",
    "SMARCA1": "SWI/SNF chromatin remodelling — gene transcription regulation",
    "GPR87": "G protein-coupled receptor 87 — stress response; overexpressed in lung & bladder cancer",
    "SIRPA": "Signal regulatory protein alpha — immune checkpoint; inhibits macrophage phagocytosis",
    "APLP1": "Amyloid precursor-like protein 1 — cell adhesion & neurotrophic signalling",
    "CDKN2B": "Cyclin-dependent kinase inhibitor 2B (p15) — cell cycle brake; tumour suppressor",
    "CRIP2": "Cysteine-rich intestinal protein 2 — transcription factor; LIM domain protein",
    "LAMA4": "Laminin alpha 4 — basement membrane component; promotes tumour angiogenesis",
    "PHLDA2": "Pleckstrin homology-like domain A2 — imprinted growth suppressor; PI3K/AKT inhibitor",
    "VGF": "VGF nerve growth factor — neuropeptide precursor; expressed in neuroendocrine tumours",
    "KRT16": "Keratin 16 — epithelial cytoskeleton; stress-response keratin in squamous cancers",
    "IL32": "Interleukin 32 — pro-inflammatory cytokine; promotes tumour immune infiltration",
    "CSTA": "Cystatin A — cysteine protease inhibitor; epithelial barrier & cancer invasion",
    "CNN3": "Calponin 3 — actin-binding protein; cytoskeletal regulation",
    "SLFN13": "Schlafen 13 — immune regulation & cell cycle; interferon-stimulated gene",
    "KRT6A": "Keratin 6A — epithelial stress marker; expressed in squamous cell carcinomas",
    "DAB2": "Disabled homolog 2 — endocytic adaptor; tumour suppressor in ovarian & prostate cancer",
    "BCAM": "Basal cell adhesion molecule — laminin receptor; overexpressed in ovarian cancer",
    "CHMP4C": "Charged multivesicular body protein 4C — ESCRT pathway; cytokinesis & DNA damage",
    "PPP1R1B": "DARPP-32 — dopamine signalling; amplified in gastric cancer",
    "NT5E": "CD73 — immune checkpoint; adenosine production promotes immune evasion",
    "TMEM30B": "Transmembrane protein 30B — phospholipid flippase subunit",
    "GNA15": "G protein alpha 15 — promiscuous Gq signalling; haematopoietic cells",
    "STC2": "Stanniocalcin 2 — calcium/phosphate regulation; hypoxia-induced tumour survival",
    "GJB3": "Gap junction beta 3 (Connexin 31) — intercellular communication",
    "TNFRSF12A": "TNF receptor superfamily 12A — apoptosis & angiogenesis; overexpressed in tumours",
    "LIMD2": "LIM domain protein 2 — cell migration & invasion; metastasis regulator",
    "MYOF": "Myoferlin — membrane repair & VEGFR2 recycling; overexpressed in breast & lung cancer",
    "RHOH": "Ras homolog H — haematopoietic GTPase; tumour suppressor in lymphomas",
    "ELMO1": "Engulfment & motility protein 1 — Rac1 activation; promotes cancer cell invasion",
    "MLPH": "Melanophilin — vesicle transport; expressed in ER+ breast cancer",
    "CLDN1": "Claudin 1 — tight junction protein; altered expression drives invasion & metastasis",
    "HID1": "HID1 domain containing — Golgi membrane protein; vesicle trafficking",
    "CADM1": "Cell adhesion molecule 1 — tumour suppressor; loss promotes invasion",
    "TCF4": "Transcription factor 4 — Wnt/E-box transcription; neural & epithelial cancers",
    "ITGA3": "Integrin alpha 3 — extracellular matrix receptor; promotes invasion",
    "STC1": "Stanniocalcin 1 — calcium signalling; hypoxia-regulated tumour cell survival",
    "LRATD2": "Lecithin retinol acyltransferase domain 2 — lipid metabolism",
    "EREG": "Epiregulin — EGFR/HER4 ligand; promotes tumour growth in colorectal & lung cancer",
    "SMOC1": "SPARC-related modular calcium binding 1 — extracellular matrix; Wnt modulator",
    "ITGB5": "Integrin beta 5 — vitronectin receptor; promotes tumour angiogenesis & invasion",
    "KDELR3": "KDEL ER retention receptor 3 — protein retention in endoplasmic reticulum",
    "EPHA2": "Ephrin type-A receptor 2 — cell migration; overexpressed in many cancers",
    "ARHGAP29": "Rho GTPase activating protein 29 — RhoA inhibitor; regulates cell migration",
    "TPM1": "Tropomyosin 1 — actin cytoskeleton stabiliser; tumour suppressor activity",
    "S100A10": "S100 calcium-binding protein A10 — plasminogen receptor; promotes tumour invasion",
    "AEBP1": "Adipocyte enhancer binding protein 1 — transcription repressor; promotes invasion",
    "HMGA2": "High mobility group AT-hook 2 — chromatin remodelling; oncogene in many cancers",
    "JAG1": "Jagged 1 — Notch ligand; promotes tumour stem cell maintenance & metastasis",
    "APOC1": "Apolipoprotein C1 — lipid metabolism; expressed in macrophages & some tumours",
    "SLFN11": "Schlafen 11 — DNA replication stress response; predicts sensitivity to DNA-damaging drugs",
    "COL18A1": "Collagen type XVIII — endostatin precursor; anti-angiogenic when cleaved",
    "PPL": "Periplakin — cornified envelope protein; cytoskeletal linker in epithelial cells",
    "OLFM1": "Olfactomedin 1 — secreted glycoprotein; regulates apoptosis & neural development",
    "VSIR": "VISTA immune checkpoint — suppresses T cell activation",
    "IER3": "Immediate early response 3 — NF-kB target; pro-survival signal after DNA damage",
    "TGFB1I1": "TGF-beta 1 induced transcript 1 (Hic-5) — focal adhesion scaffold; EMT promoter",
    "GSN": "Gelsolin — actin filament severing; tumour suppressor in breast & bladder cancer",
    "B3GNT3": "Beta-1,3-N-acetylglucosaminyltransferase 3 — glycosylation; immune cell homing",
    "LAPTM5": "Lysosomal protein transmembrane 5 — lysosomal trafficking; immune cell marker",
}


def get_confidence_label(r2):
    if r2 is None:
        return None, None
    if r2 >= 0.6:
        return "Good fit", "green"
    elif r2 >= 0.35:
        return "Moderate fit", "orange"
    else:
        return "Weak fit", "red"


@st.cache_data
def load_data():
    parquet_path = hf_hub_download(
        repo_id=DATA_REPO, repo_type="dataset",
        filename="master_features_slim.parquet",
    )
    return pd.read_parquet(parquet_path)


@st.cache_data
def load_results():
    results_path = hf_hub_download(
        repo_id=DATA_REPO, repo_type="dataset",
        filename="models/drug_model_results.csv",
    )
    return pd.read_csv(results_path)


@st.cache_resource
def load_model(model_file):
    model_path = hf_hub_download(
        repo_id=DATA_REPO, repo_type="dataset", filename=model_file,
    )
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


@st.cache_data
def load_features(features_file):
    features_path = hf_hub_download(
        repo_id=DATA_REPO, repo_type="dataset", filename=features_file,
    )
    return np.load(features_path, allow_pickle=True).tolist()


@st.cache_data
def get_all_predictions(drug_name, model_file, model_features, _df):
    model = xgb.XGBRegressor()
    model_path = hf_hub_download(
        repo_id=DATA_REPO, repo_type="dataset", filename=model_file,
    )
    model.load_model(model_path)
    drug_df = _df[_df["DRUG_NAME"] == drug_name].copy()
    X_all = drug_df.reindex(columns=model_features, fill_value=0).fillna(0)
    drug_df = drug_df.copy()
    drug_df["Predicted_LN_IC50"] = model.predict(X_all)
    return drug_df[["CELL_LINE_NAME", "LN_IC50", "Predicted_LN_IC50", "TCGA_DESC"]].dropna()


def make_scatter(all_preds_df, selected_cell_line, selected_pred, selected_actual, drug_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    ax.scatter(
        all_preds_df["LN_IC50"],
        all_preds_df["Predicted_LN_IC50"],
        alpha=0.4, s=18, color="#2c5f8a",
        label="All cell lines", zorder=2,
    )

    all_vals = pd.concat([all_preds_df["LN_IC50"], all_preds_df["Predicted_LN_IC50"]])
    vmin, vmax = all_vals.min(), all_vals.max()
    ax.plot([vmin, vmax], [vmin, vmax], color="#aaaaaa", linestyle="--",
            linewidth=1, label="Perfect prediction", zorder=1)

    ax.scatter(
        selected_actual, selected_pred,
        color="#c0392b", s=90, zorder=5,
        label=f"{selected_cell_line} (selected)",
        edgecolors="#7b241c", linewidths=0.8,
    )

    ax.set_xlabel("Actual LN_IC50", fontsize=10, color="#333", fontfamily="sans-serif")
    ax.set_ylabel("Predicted LN_IC50", fontsize=10, color="#333", fontfamily="sans-serif")
    ax.set_title(
        f"{drug_name} — Predicted vs Actual LN_IC50  ({len(all_preds_df)} cell lines)",
        fontsize=11, color="#1a1a1a", fontfamily="sans-serif", pad=12,
    )
    ax.tick_params(colors="#555", labelsize=9)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.8, edgecolor="#dddddd")
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    return fig


# ── App layout ─────────────────────────────────────────────────────────────────

st.markdown("## Cancer Drug Response Predictor")
st.markdown(
    "<p style='font-family:Helvetica Neue, sans-serif; font-size:0.92rem; color:#555; margin-top:-0.5rem;'>"
    "Predicts LN_IC50 drug sensitivity for cancer cell lines using gene expression data. "
    "Trained on GDSC2 · DepMap · XGBoost · v1 demo: 5 drugs"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

try:
    df = load_data()
    results_df = load_results()

    available_df = df[df["DRUG_NAME"].isin(SUPPORTED_DRUGS.keys())].copy()
    if available_df.empty:
        st.error("No supported drugs found in dataset.")
        st.stop()

    # ── Selection row ─────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tumour_types = sorted(available_df["TCGA_DESC"].dropna().unique())
        selected_tumour = st.selectbox("Tumour Type", tumour_types)
    with col_b:
        tumour_df = available_df[available_df["TCGA_DESC"] == selected_tumour].copy()
        drugs_for_tumour = sorted(tumour_df["DRUG_NAME"].dropna().unique())
        selected_drug = st.selectbox("Drug", drugs_for_tumour)
    with col_c:
        drug_df = tumour_df[tumour_df["DRUG_NAME"] == selected_drug].copy()
        cell_lines = sorted(drug_df["CELL_LINE_NAME"].dropna().unique())
        selected_cell_line = st.selectbox("Cell Line", cell_lines)

    st.divider()

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

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted LN_IC50", f"{pred:.4f}")
    with m2:
        if actual is not None:
            st.metric("Actual LN_IC50", f"{actual:.4f}")
    with m3:
        if r2_value is not None:
            st.metric("Model R²", f"{r2_value:.3f}")

    if rmse_value is not None:
        st.caption(f"Model RMSE: {rmse_value:.3f}")

    # ── Model confidence ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Model Confidence")

    confidence_label, confidence_color = get_confidence_label(r2_value)
    if confidence_label:
        if confidence_color == "green":
            st.success(f"**{confidence_label}** — predictions for {selected_drug} are reliable. R² = {r2_value:.3f}")
        elif confidence_color == "orange":
            st.warning(f"**{confidence_label}** — predictions for {selected_drug} show real signal but have meaningful error. R² = {r2_value:.3f}")
        else:
            st.error(f"**{confidence_label}** — model for {selected_drug} explains little variance. Treat prediction as indicative only. R² = {r2_value:.3f}")

        with st.expander("What does this mean?"):
            st.markdown("""
**R²** measures how well the model explains variation in drug response across cell lines.

| R² | Interpretation |
|---|---|
| ≥ 0.6 | Good — model captures most biological signal |
| 0.35 – 0.6 | Moderate — useful signal, not highly precise |
| < 0.35 | Weak — prediction is a rough estimate only |

A weak fit may mean gene expression alone is insufficient for this drug — other data modalities (mutations, CNV) are likely needed.
""")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Predicted vs Actual LN_IC50")
    st.caption("Each point represents one cell line. Red point = selected cell line. Dashed line = perfect prediction.")

    if actual is not None:
        with st.spinner("Loading predictions across all cell lines..."):
            all_preds = get_all_predictions(
                selected_drug, model_info["model_file"], model_features, df
            )
        if not all_preds.empty:
            fig = make_scatter(all_preds, selected_cell_line, pred, actual, selected_drug)
            st.pyplot(fig)
            plt.close(fig)

            percentile = (all_preds["LN_IC50"] < actual).mean() * 100
            sensitivity = (
                "lower than most — more sensitive" if percentile < 40
                else "higher than most — more resistant" if percentile > 60
                else "near the median sensitivity"
            )
            st.caption(
                f"**{selected_cell_line}** actual LN_IC50 is at the **{percentile:.0f}th percentile** "
                f"among all {selected_drug} cell lines — {sensitivity}."
            )
    else:
        st.info("Actual LN_IC50 not available for this selection.")

    # ── Selection details ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Selection Details")
    details_cols = [c for c in ["CELL_LINE_NAME", "TCGA_DESC", "PATHWAY_NAME", "DRUG_NAME"] if c in row.columns]
    st.dataframe(row[details_cols], use_container_width=True)

    # ── Gene importance ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Top Gene Importance")
    st.caption("Genes ranked by contribution to this prediction, with biological context.")

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Gene": model_features,
        "Importance Score": importances,
    }).sort_values("Importance Score", ascending=False).head(20).reset_index(drop=True)

    importance_df["Biological Role"] = importance_df["Gene"].map(
        lambda g: GENE_PATHWAYS.get(g.split(" ")[0], "—")
    )

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
        },
    )

    known_count = (importance_df["Biological Role"] != "—").sum()
    st.caption(f"{known_count} of top 20 genes have pathway annotations.")

    st.divider()
    st.markdown(
        "<p style='font-family:Helvetica Neue,sans-serif;font-size:0.78rem;color:#999;'>"
        "⚠️ This tool is not a clinical instrument. Predictions are based on cancer cell line data (GDSC2) "
        "and should not be used to guide treatment decisions."
        "</p>",
        unsafe_allow_html=True
    )

except Exception as e:
    st.error("App failed to load.")
    st.exception(e)
