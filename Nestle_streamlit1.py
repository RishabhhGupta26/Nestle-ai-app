# -------------------------------------------
# Nestlé AI Model Prediction System (Top50 only version - FINAL FIXED)
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
# CONFIGURATION
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

OUTPUT_NAME_MAP = {1: "filtration_rate", 2: "leaching_rate", 3: "yield", 4: "starch"}
OUTPUT_LABEL_MAP = {1: "Filtration Rate", 2: "Leaching Rate", 3: "Yield", 4: "Starch"}

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
# HEADER WITH LOGOS
# =====================================================
col1, col2 = st.columns([7, 1])

with col1:
    st.markdown("<h1 style='color:#1b3e8b;'>Nestlé’s AI Model Prediction System</h1>", unsafe_allow_html=True)

with col2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)

st.markdown("#### Powered by Polymerize")
if BUILDER_LOGO.exists():
    st.image(str(BUILDER_LOGO), width=180)


# =====================================================
# UTILS
# =====================================================
def split_into_groups(df):
    cols = list(df.columns)

    outputs_idx = [1, 2, 3, 49]  # 4 output columns

    A1_1_idx = slice(4, 12)
    C1_idx  = slice(12, 15)
    A1_2_idx = slice(15, 28)
    B1_idx  = slice(28, 47)
    E1_idx  = slice(47, 50)
    E2_idx  = slice(50, 53)
    NIR_idx = slice(54, None)

    outputs_cols = [cols[i] for i in outputs_idx]

    return {
        'Outputs': df[outputs_cols],
        'A1': pd.concat([df[A1_1_idx], df[A1_2_idx]], axis=1),
        'C1': df[C1_idx],
        'B1': df[B1_idx],
        'E1': df[E1_idx],
        'E2': df[E2_idx],
        'NIR': df[NIR_idx]
    }


def preprocess_groups(groups, ref_groups, y_ref, n_comp):
    processed = {}

    # Normal groups
    for g in [g for g in groups if g not in ["NIR", "Outputs"]]:
        X_ref = ref_groups[g].values
        X_test = groups[g].values

        scaler = RobustScaler()
        scaler.fit(X_ref)

        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        pt.fit(scaler.transform(X_ref))

        processed[g] = pt.transform(scaler.transform(X_test))

    # NIR (PLS)
    nir_train = ref_groups["NIR"].values
    nir_test = groups["NIR"].values

    comp = min(n_comp, nir_train.shape[1], max(1, nir_train.shape[0] - 1))

    pls = PLSRegression(n_components=comp)
    pls.fit(nir_train, y_ref)

    nir_train_scores = pls.transform(nir_train)
    nir_test_scores = pls.transform(nir_test)

    scaler = RobustScaler()
    scaler.fit(nir_train_scores)

    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    pt.fit(scaler.transform(nir_train_scores))

    processed["NIR"] = pt.transform(scaler.transform(nir_test_scores))

    return processed


# =====================================================
# LOAD TRAIN REFERENCE
# =====================================================
train_df = pd.read_excel(TRAIN_REFERENCE_FILE, sheet_name=TRAIN_SHEET)

# Remove text column (ID column)
if train_df.iloc[:, 0].dtype == "object":
    train_df = train_df.drop(train_df.columns[0], axis=1)

train_groups = split_into_groups(train_df)


# =====================================================
# OUTPUT SELECTION
# =====================================================
selected_output = st.selectbox("🎯 Select Output Property",
                               [1, 2, 3, 4],
                               format_func=lambda x: OUTPUT_LABEL_MAP[x])

output_name = OUTPUT_NAME_MAP[selected_output]
pls_comp = PLS_COMPONENTS_MAP[selected_output]


# =====================================================
# LOAD TOP50 MODELS ONLY
# =====================================================
detail_file = OUTPUT_DETAILS_DIR / f"{output_name}_Detailed_Results.xlsx"

detail_df = pd.read_excel(detail_file, sheet_name="Top50_R2")
detail_df["Groups"] = detail_df["Groups"].apply(lambda x: ast.literal_eval(x))

st.subheader(f"📘 Top 50 Models for {OUTPUT_LABEL_MAP[selected_output]}")
st.dataframe(detail_df)


# =====================================================
# SELECT MODEL RANK FROM TOP50
# =====================================================
rank_list = detail_df.index + 1
selected_rank = st.selectbox("Select Model Rank from Top 50", rank_list)

required_groups = detail_df.iloc[selected_rank - 1]["Groups"]
selected_model_name = detail_df.iloc[selected_rank - 1]["Model"]


# =====================================================
# LOAD MODEL FILE
# =====================================================
model_folder = MODELS_DIR / f"{output_name}_Top50_R2"
model_pattern = f"{selected_rank:02d}_{selected_model_name}_R2_*.pkl"

matched_files = list(model_folder.glob(model_pattern))
if not matched_files:
    st.error("❌ Model file missing for this rank.")
    st.stop()

model_path = matched_files[0]

st.success(f"Selected model: Rank {selected_rank} — {selected_model_name}")


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.subheader("📊 Feature Importance")

feature_img = FEATURE_IMPORTANCE_DIR / output_name / f"{selected_rank:02d}_{selected_model_name}_importance.png"

if feature_img.exists():
    st.image(str(feature_img), use_column_width=True)

    with open(feature_img, "rb") as f:
        st.download_button("📥 Download Feature Importance Image",
                           f,
                           file_name=f"feature_importance_{output_name}.png")
else:
    st.warning("⚠ Feature importance image not found.")


# =====================================================
# UPLOAD TEST FILE
# =====================================================
uploaded_file = st.file_uploader("📁 Upload Testing Excel File", type=["xlsx"])
test_df = None

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

    test_df = pd.read_excel(xls, sheet_name=sheet)

    # Remove ID column if it is text
    if test_df.iloc[:, 0].dtype == "object":
        test_df = test_df.drop(test_df.columns[0], axis=1)

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
