# Nestle_streamlit_final.py
# -------------------------------------------
# Nestlé AI Model Prediction System (Top50 final - Deploy Ready)
# -------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.cross_decomposition import PLSRegression
import ast
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# CONFIGURATION (STREAMLIT + GITHUB DEPLOY SAFE)
# -------------------------
# Base directory = folder where this .py file is located
BASE = Path(__file__).resolve().parent

PASSWORD = "Nestle@123"

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

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="Nestlé AI System", layout="wide")

# -------------------------
# AUTH
# -------------------------
st.markdown("### 🔒 Secure Access")
pw = st.text_input("Enter password", type="password")
if pw != PASSWORD:
    if pw != "":
        st.error("Incorrect password.")
    st.stop()

# -------------------------
# HEADER
# -------------------------
c1, c2 = st.columns([7, 1])
with c1:
    st.markdown("<h1 style='color:#1b3e8b;'>Nestlé’s AI Model Prediction System</h1>", unsafe_allow_html=True)
with c2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=160)
st.markdown("#### Powered by Polymerize")
if BUILDER_LOGO.exists():
    st.image(str(BUILDER_LOGO), width=140)

# -------------------------
# HELPERS
# -------------------------
def drop_first_column(df):
    if df is None or df.shape[1] == 0:
        return df
    return df.drop(df.columns[0], axis=1)

def numericize_and_fill(df):
    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors='coerce')
    medians = df_num.median(numeric_only=True)
    return df_num.fillna(medians)

def safe_get_cols_by_index(df_cols, indices):
    names = []
    for i in indices:
        if 0 <= i < len(df_cols):
            names.append(df_cols[i])
    return names

def adjusted_index(i):
    return i - 1

# -------------------------
# SPLIT INTO GROUPS
# -------------------------
def split_into_groups(df):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)

    # Outputs detection
    try:
        orig_outputs = [1, 2, 3, 49]
        adj = [adjusted_index(i) for i in orig_outputs]
        output_cols = safe_get_cols_by_index(cols, adj)

        if len(output_cols) != 4:
            output_cols = [c for c in cols if any(k in c.lower() for k in ["filtration", "leaching", "yield", "starch"])]
    except:
        output_cols = [c for c in cols if any(k in c.lower() for k in ["filtration", "leaching", "yield", "starch"])]

    if len(output_cols) != 4:
        st.warning("Unable to detect output columns. Select manually.")
        ch = st.multiselect("Select 4 output columns", cols)
        if len(ch) != 4:
            st.stop()
        output_cols = ch

    # Group slices
    def slice_to_names(s):
        start = adjusted_index(s.start) if s.start is not None else 0
        stop = adjusted_index(s.stop) if s.stop is not None else len(cols)
        return safe_get_cols_by_index(cols, list(range(start, stop)))

    A1_cols = slice_to_names(slice(4, 12)) + slice_to_names(slice(15, 28))
    C1_cols = slice_to_names(slice(12, 15))
    B1_cols = slice_to_names(slice(28, 47))
    E1_cols = slice_to_names(slice(47, 50))
    E2_cols = slice_to_names(slice(50, 53))
    NIR_cols = slice_to_names(slice(54, None))

    return {
        "Outputs": df[output_cols],
        "A1": df[A1_cols] if A1_cols else pd.DataFrame(index=df.index),
        "C1": df[C1_cols] if C1_cols else pd.DataFrame(index=df.index),
        "B1": df[B1_cols] if B1_cols else pd.DataFrame(index=df.index),
        "E1": df[E1_cols] if E1_cols else pd.DataFrame(index=df.index),
        "E2": df[E2_cols] if E2_cols else pd.DataFrame(index=df.index),
        "NIR": df[NIR_cols] if NIR_cols else pd.DataFrame(index=df.index),
    }

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess_groups_for_prediction(groups, train_groups, y_ref, n_comp):
    processed = {}

    for g in [x for x in groups if x not in ["NIR", "Outputs"]]:
        X_ref = numericize_and_fill(train_groups[g])
        X_test = numericize_and_fill(groups[g])

        scaler = RobustScaler()
        pt = PowerTransformer(method="yeo-johnson", standardize=False)

        scaler.fit(X_ref.values)
        pt.fit(scaler.transform(X_ref.values))

        processed[g] = pt.transform(scaler.transform(X_test.values))

    nir_train = numericize_and_fill(train_groups["NIR"]).values
    nir_test = numericize_and_fill(groups["NIR"]).values

    if nir_train.size == 0 or nir_test.size == 0:
        processed["NIR"] = np.zeros((len(list(groups.values())[0].index), 1))
    else:
        comp = min(n_comp, nir_train.shape[1], max(1, nir_train.shape[0] - 1))
        pls = PLSRegression(n_components=comp)
        y_use = y_ref if y_ref.ndim == 1 else y_ref[:, 0]
        pls.fit(nir_train, y_use)

        nir_train_s = pls.transform(nir_train)
        nir_test_s = pls.transform(nir_test)

        scaler = RobustScaler()
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        scaler.fit(nir_train_s)
        pt.fit(scaler.transform(nir_train_s))

        processed["NIR"] = pt.transform(scaler.transform(nir_test_s))

    return processed

# -------------------------
# LOAD TRAINING REF
# -------------------------
try:
    train_ref_df = pd.read_excel(TRAIN_REFERENCE_FILE, sheet_name=TRAIN_SHEET)
except Exception as e:
    st.error(f"Failed to read training file: {e}")
    st.stop()

train_ref_df = drop_first_column(train_ref_df)
train_ref_df = numericize_and_fill(train_ref_df)
train_ref_groups = split_into_groups(train_ref_df)

# -------------------------
# SELECT OUTPUT PROPERTY
# -------------------------
selected_output = st.selectbox("🎯 Select Output", [1, 2, 3, 4], format_func=lambda x: OUTPUT_LABEL_MAP[x])
output_name = OUTPUT_NAME_MAP[selected_output]
pls_comp = PLS_COMPONENTS_MAP[selected_output]

# -------------------------
# # LOAD TOP50 DETAILS
# # -------------------------
# detail_file = OUTPUT_DETAILS_DIR / f"{output_name}_Detailed_Results.xlsx"
# detail_df = pd.read_excel(detail_file, sheet_name="Top50_R2")
# detail_df["Groups"] = detail_df["Groups"].apply(lambda x: ast.literal_eval(x))

# st.subheader(f"📘 Top 50 Models for {OUTPUT_LABEL_MAP[selected_output]}")
# st.dataframe(detail_df)

# selected_rank = st.selectbox("Select Model Rank", list(range(1, len(detail_df) + 1)))
# required_groups = detail_df.iloc[selected_rank - 1]["Groups"]
# selected_model_name = detail_df.iloc[selected_rank - 1]["Model"]

# -------------------------
# LOAD TOP50 DETAILS
# -------------------------
detail_file = OUTPUT_DETAILS_DIR / f"{output_name}_Detailed_Results.xlsx"
detail_df = pd.read_excel(detail_file, sheet_name="Top50_R2")
detail_df["Groups"] = detail_df["Groups"].apply(lambda x: ast.literal_eval(x))

st.subheader(f"📘 Top 50 Models for {OUTPUT_LABEL_MAP[selected_output]}")


# REMOVE the Output_Index column if present
detail_df = detail_df.drop(columns=["Output_Index"], errors="ignore")

# FIX → Index from 1 to 50
detail_df_display = detail_df.copy()
detail_df_display.index = detail_df_display.index + 1

st.dataframe(detail_df_display)

# Select Rank (1–50)
selected_rank = st.selectbox("Select Model Rank", list(range(1, len(detail_df) + 1)))

required_groups = detail_df.iloc[selected_rank - 1]["Groups"]
selected_model_name = detail_df.iloc[selected_rank - 1]["Model"]


model_folder = MODELS_DIR / f"{output_name}_Top50_R2"
model_path = list(model_folder.glob(f"{selected_rank:02d}_{selected_model_name}_R2_*.pkl"))[0]

st.success(f"Selected model: Rank {selected_rank} — {selected_model_name}")

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.subheader("📊 Feature Importance")
feature_img = FEATURE_IMPORTANCE_DIR / output_name / f"{selected_rank:02d}_{selected_model_name}_importance.png"
if feature_img.exists():
    st.image(str(feature_img), use_column_width=True)

# -------------------------
# TEST FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📁 Upload Testing Excel File", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    test_df = pd.read_excel(xls, sheet_name=sheet)

    test_df = drop_first_column(test_df)
    test_df = numericize_and_fill(test_df)
    test_groups = split_into_groups(test_df)

    st.success("Test file loaded.")

# -------------------------
# PREDICT
# -------------------------
if st.button("🚀 Predict Now"):
    if uploaded_file is None:
        st.error("Upload test file first.")
        st.stop()

    model = joblib.load(model_path)
    y_train_ref = train_ref_groups["Outputs"].values[:, selected_output - 1]

    processed_test = preprocess_groups_for_prediction(test_groups, train_ref_groups, y_train_ref, pls_comp)

    X = np.hstack([processed_test[g] for g in required_groups])

    y_pred = model.predict(X)

    out_df = pd.DataFrame({f"Predicted {OUTPUT_LABEL_MAP[selected_output]}": y_pred})

    st.subheader("✅ Predictions")
    st.dataframe(out_df)

    st.download_button("📥 Download Predictions CSV", out_df.to_csv(index=False), file_name="predictions.csv")
