# -------------------------------------------
# Nestlé AI Model Prediction System (Final Version)
# -------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.cross_decomposition import PLSRegression
import re
import ast
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION (All paths updated to projectFinal)
# =====================================================
BASE = Path("C:/Users/risha/projectFinal")

PASSWORD = "Password@123"

TRAIN_REFERENCE_FILE = BASE / "Nestle test.xlsx"
TRAIN_SHEET = "A1B1C1E1E2_Building"

OUTPUT_DETAILS_DIR = BASE / "detailed_analysis_results"
MODELS_DIR = BASE / "saved_models"
FEATURE_IMPORTANCE_DIR = BASE / "feature_importance"

LOGO_PATH = BASE / "logo.svg"
BUILDER_LOGO = BASE / "download.png"

# Output mapping
OUTPUT_NAME_MAP = {
    1: "filtration_rate",
    2: "leaching_rate",
    3: "yield",
    4: "starch"
}

OUTPUT_LABEL_MAP = {
    1: "Filtration Rate",
    2: "Leaching Rate",
    3: "Yield",
    4: "Starch"
}

PLS_COMPONENTS_MAP = {1: 5, 2: 6, 3: 5, 4: 6}

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(page_title="Nestlé AI System", layout="wide")

# =====================================================
# PASSWORD GATE
# =====================================================
st.markdown("### 🔒 Secure Access")

pw = st.text_input("Enter password to open dashboard", type="password")
if pw != PASSWORD:
    if pw != "":
        st.error("Incorrect password.")
    st.stop()

# =====================================================
# TOP HEADER WITH LOGOS
# =====================================================
col1, col2 = st.columns([7, 1])

with col1:
    st.markdown(
        "<h1 style='color:#1b3e8b;'>Nestlé’s AI Model Prediction System</h1>",
        unsafe_allow_html=True
    )

with col2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)

st.markdown("#### Powered by Polymerize ")
if BUILDER_LOGO.exists():
    st.image(str(BUILDER_LOGO), width=180)

# =====================================================
# UTILS
# =====================================================
def split_into_groups(df):
    cols = list(df.columns)

    outputs_idx = [1, 2, 3, 49]
    A1_1_idx = slice(4, 12)
    C1_idx = slice(12, 15)
    A1_2_idx = slice(15, 28)
    B1_idx = slice(28, 47)
    E1_idx = slice(47, 50)
    E2_idx = slice(50, 53)
    NIR_idx = slice(54, None)

    outputs_cols = [cols[i] for i in outputs_idx]
    A1_cols = cols[A1_1_idx] + cols[A1_2_idx]
    C1_cols = cols[C1_idx]
    B1_cols = cols[B1_idx]
    E1_cols = cols[E1_idx]
    E2_cols = cols[E2_idx]
    NIR_cols = cols[NIR_idx]

    return {
        'Outputs': df[outputs_cols],
        'A1': df[A1_cols],
        'C1': df[C1_cols],
        'B1': df[B1_cols],
        'E1': df[E1_cols],
        'E2': df[E2_cols],
        'NIR': df[NIR_cols]
    }


def preprocess_groups(groups, ref_groups, y_ref, n_comp):
    processed = {}

    # Normal groups
    for g in [g for g in groups if g not in ["NIR", "Outputs"]]:
        scaler = RobustScaler()
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        X_ref = ref_groups[g].values
        X_test = groups[g].values
        scaler.fit(X_ref)
        pt.fit(scaler.transform(X_ref))
        processed[g] = pt.transform(scaler.transform(X_test))

    # NIR with PLS
    nir_train = ref_groups["NIR"].values
    nir_test = groups["NIR"].values

    comp = min(n_comp, nir_train.shape[1], max(1, nir_train.shape[0] - 1))
    pls = PLSRegression(n_components=comp)
    pls.fit(nir_train, y_ref)

    nir_train_scores = pls.transform(nir_train)
    nir_test_scores = pls.transform(nir_test)

    scaler = RobustScaler()
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    scaler.fit(nir_train_scores)
    pt.fit(scaler.transform(nir_train_scores))
    processed["NIR"] = pt.transform(scaler.transform(nir_test_scores))

    return processed


# =====================================================
# LOAD TRAIN REFERENCE
# =====================================================
train_df = pd.read_excel(TRAIN_REFERENCE_FILE, sheet_name=TRAIN_SHEET)
train_groups = split_into_groups(train_df)

# =====================================================
# OUTPUT SELECTION
# =====================================================
selected_output = st.selectbox(
    "🎯 Select Output Property",
    [1, 2, 3, 4],
    format_func=lambda x: OUTPUT_LABEL_MAP[x]
)

output_name = OUTPUT_NAME_MAP[selected_output]
pls_comp = PLS_COMPONENTS_MAP[selected_output]

# =====================================================
# LOAD TOP50 MODEL TABLE
# =====================================================
detail_file = OUTPUT_DETAILS_DIR / f"{output_name}_Detailed_Results.xlsx"

detail_df = pd.read_excel(detail_file, sheet_name="Top50_R2")
detail_df["Groups"] = detail_df["Groups"].apply(lambda x: ast.literal_eval(x))

st.subheader(f"📘 Top 50 Models for {OUTPUT_LABEL_MAP[selected_output]}")
st.dataframe(detail_df)

# =====================================================
# LOAD MODEL FILES (Top50)
# =====================================================
folder = MODELS_DIR / f"{output_name}_Top50_R2"
model_files = sorted(folder.glob("*.pkl"))

pattern = r"(\d+)_([A-Za-z0-9]+)_R2_([0-9.]+)_MAPE_([0-9.]+).pkl"

rows = []
for f in model_files:
    m = re.search(pattern, f.name)
    if m:
        rank = int(m.group(1))
        name = m.group(2)
        r2 = float(m.group(3))
        mape = float(m.group(4))
        rows.append([rank, name, r2, mape, f])

model_df = pd.DataFrame(rows, columns=["Rank", "Model", "R2", "MAPE", "File"])
model_df = model_df.sort_values("Rank")

st.subheader("📂 Saved Models")
st.dataframe(model_df)

# Select Model Rank
selected_rank = st.selectbox(
    "Select Model Rank",
    model_df["Rank"].tolist()
)

row = model_df[model_df["Rank"] == selected_rank].iloc[0]
model_path = row["File"]
selected_model_name = row["Model"]

st.success(f"Selected model: Rank {selected_rank} — {selected_model_name}")

# Required groups
required_groups = detail_df.iloc[selected_rank - 1]["Groups"]

# =====================================================
# FEATURE IMPORTANCE IMAGE
# =====================================================

st.subheader("📊 Feature Importance")

feature_img = FEATURE_IMPORTANCE_DIR / output_name / f"{selected_rank:02d}_{selected_model_name}_importance.png"

if feature_img.exists():
    st.image(str(feature_img), use_column_width=True)
else:
    st.warning("Feature importance not found.")

# =====================================================
# UPLOAD TEST FILE (One time only)
# =====================================================

uploaded_file = st.file_uploader("📁 Upload Testing Excel File", type=["xlsx"])
test_df = None

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    st_sheet = st.selectbox("Select Sheet", xls.sheet_names)
    test_df = pd.read_excel(xls, sheet_name=st_sheet)
    test_groups = split_into_groups(test_df)

# =====================================================
# PREDICT
# =====================================================

if st.button("🚀 Predict Now"):
    if test_df is None:
        st.error("Upload a testing Excel file first.")
        st.stop()

    model = joblib.load(model_path)

    y_train_ref = train_groups["Outputs"].values[:, selected_output - 1]
    processed_test = preprocess_groups(test_groups, train_groups, y_train_ref, pls_comp)

    X_input = np.hstack([processed_test[g] for g in required_groups])

    y_pred = model.predict(X_input)

    test_df[f"Predicted_{OUTPUT_LABEL_MAP[selected_output]}"] = y_pred

    st.subheader("✅ Predictions")
    st.dataframe(test_df.head(50))

    st.download_button(
        "📥 Download Predictions",
        test_df.to_csv(index=False).encode("utf-8"),
        file_name=f"Predicted_{output_name}.csv"
    )
