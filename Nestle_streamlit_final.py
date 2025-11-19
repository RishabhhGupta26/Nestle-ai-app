# Nestle_streamlit_final.py
# -------------------------------------------
# Nestlé AI Model Prediction System (Top50 final)
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
# CONFIGURATION
# -------------------------
BASE = Path("C:/Users/risha/projectFinal")

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
    """Always drop first column (ID like D_7) to avoid string leakage into numeric pipelines."""
    if df is None or df.shape[1] == 0:
        return df
    return df.drop(df.columns[0], axis=1)

def numericize_and_fill(df):
    """Convert all columns to numeric (coerce non-numeric to NaN) and fill NaNs with column median."""
    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors='coerce')
    # Fill NaNs with median per column
    medians = df_num.median(numeric_only=True)
    df_num = df_num.fillna(medians)
    return df_num

def safe_get_cols_by_index(df_cols, indices):
    """Return list of column names for given indices. Ignore indices out of range."""
    names = []
    for i in indices:
        if 0 <= i < len(df_cols):
            names.append(df_cols[i])
    return names

def adjusted_index(i):
    """Given original code indices that assumed a leading ID column, adjust them after dropping that column.
    Original indices (in your code) were for dataframe including ID at index 0.
    After dropping first column, shift all positive indices by -1.
    """
    return i - 1

# -------------------------
# SPLIT INTO GROUPS (robust)
# -------------------------
def split_into_groups(df):
    """
    This function:
    - expects df to have had its first column removed (we do that before calling)
    - returns dict of groups: Outputs, A1, C1, B1, E1, E2, NIR
    It tries to use the original index schema but adjusted after dropping the ID column.
    """
    df = df.copy()
    # ensure column names are strings (avoid int.lower errors)
    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)

    # Original indices used in your code (before dropping ID) were:
    # outputs_idx = [1, 2, 3, 49]
    # We'll adjust them because we dropped the first column
    try:
        orig_outputs_idx = [1, 2, 3, 49]
        adj_outputs_idx = [adjusted_index(i) for i in orig_outputs_idx]
        output_cols = safe_get_cols_by_index(cols, adj_outputs_idx)
        # Verify we found exactly 4 outputs; if not, fallback to name matching
        if len(output_cols) != 4:
            # try to detect by name keywords (case-insensitive)
            output_cols = [c for c in cols if ("filtration" in c.lower() or
                                               "leaching" in c.lower() or
                                               "yield" in c.lower() or
                                               "starch" in c.lower())]
    except Exception:
        output_cols = [c for c in cols if ("filtration" in c.lower() or
                                           "leaching" in c.lower() or
                                           "yield" in c.lower() or
                                           "starch" in c.lower())]

    # final fallback: if still not 4, ask user to pick 4 columns
    if len(output_cols) != 4:
        st.warning("Could not automatically identify 4 output columns. Please select them manually.")
        chosen = st.multiselect("Select the 4 output columns (in order)", cols, default=cols[:4])
        if len(chosen) != 4:
            st.stop()
        output_cols = chosen

    # Now determine other groups using adjusted slices (original slices minus 1)
    # Original slices (pre-drop): A1_1_idx = slice(4,12), C1_idx = slice(12,15), A1_2_idx = slice(15,28)
    # B1_idx = slice(28,47), E1_idx = slice(47,50), E2_idx = slice(50,53), NIR_idx = slice(54,None)
    def slice_to_names(s):
        if s.start is None and s.stop is None:
            return []
        start = adjusted_index(s.start) if s.start is not None else 0
        stop  = adjusted_index(s.stop) if s.stop is not None else len(cols)
        return safe_get_cols_by_index(cols, list(range(start, stop)))

    A1_cols = slice_to_names(slice(4, 12)) + slice_to_names(slice(15, 28))
    C1_cols = slice_to_names(slice(12, 15))
    B1_cols = slice_to_names(slice(28, 47))
    E1_cols = slice_to_names(slice(47, 50))
    E2_cols = slice_to_names(slice(50, 53))
    NIR_cols = slice_to_names(slice(54, None))

    # If above yielded empty groups, attempt keyword-based fallback
    if len(A1_cols) == 0:
        A1_cols = [c for c in cols if "a1" in c.lower()]
    if len(C1_cols) == 0:
        C1_cols = [c for c in cols if "c1" in c.lower()]
    if len(B1_cols) == 0:
        B1_cols = [c for c in cols if "b1" in c.lower()]
    if len(E1_cols) == 0:
        E1_cols = [c for c in cols if "e1" in c.lower()]
    if len(E2_cols) == 0:
        E2_cols = [c for c in cols if "e2" in c.lower()]
    if len(NIR_cols) == 0:
        NIR_cols = [c for c in cols if "nir" in c.lower()]

    groups = {
        "Outputs": df[output_cols],
        "A1": df[A1_cols] if len(A1_cols) > 0 else pd.DataFrame(index=df.index),
        "C1": df[C1_cols] if len(C1_cols) > 0 else pd.DataFrame(index=df.index),
        "B1": df[B1_cols] if len(B1_cols) > 0 else pd.DataFrame(index=df.index),
        "E1": df[E1_cols] if len(E1_cols) > 0 else pd.DataFrame(index=df.index),
        "E2": df[E2_cols] if len(E2_cols) > 0 else pd.DataFrame(index=df.index),
        "NIR": df[NIR_cols] if len(NIR_cols) > 0 else pd.DataFrame(index=df.index),
    }
    return groups

# -------------------------
# PREPROCESSING FIXED (uses numericized data)
# -------------------------
def preprocess_groups_for_prediction(groups, reference_train, y_reference, n_components):
    processed = {}

    # For each non-NIR group: convert to numeric, fill NA with medians, then scale+yeo-johnson
    for g in [n for n in groups if n not in ["NIR", "Outputs"]]:
        # ensure numeric & fill missing
        X_ref = numericize_and_fill(reference_train[g])
        X_test = numericize_and_fill(groups[g])

        scaler = RobustScaler()
        pt = PowerTransformer(method='yeo-johnson', standardize=False)

        scaler.fit(X_ref.values)            # fit on numeric reference
        pt.fit(scaler.transform(X_ref.values))

        processed[g] = pt.transform(scaler.transform(X_test.values))

    # NIR handling: PLS on raw numeric NIR reference -> transform test -> scale
    nir_train = numericize_and_fill(reference_train["NIR"]).values
    nir_test = numericize_and_fill(groups["NIR"]).values

    # ensure valid shapes
    if nir_train.size == 0 or nir_test.size == 0:
        processed["NIR"] = np.zeros((len(list(groups.values())[0].index), 1))
    else:
        n_comp = min(n_components, nir_train.shape[1], max(1, nir_train.shape[0] - 1))
        pls = PLSRegression(n_components=n_comp)
        # y_reference may be 1d or 2d: use 1d if single-output
        y_fit = y_reference if y_reference.ndim == 1 else y_reference[:, 0]
        pls.fit(nir_train, y_fit)

        nir_train_scores = pls.transform(nir_train)
        nir_test_scores = pls.transform(nir_test)

        scaler = RobustScaler()
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        scaler.fit(nir_train_scores)
        pt.fit(scaler.transform(nir_train_scores))
        processed["NIR"] = pt.transform(scaler.transform(nir_test_scores))

    return processed

# -------------------------
# LOAD TRAINING REFERENCE (drop first col always)
# -------------------------
try:
    train_ref_df = pd.read_excel(TRAIN_REFERENCE_FILE, sheet_name=TRAIN_SHEET)
except Exception as e:
    st.error(f"Failed to read training reference file: {e}")
    st.stop()

# ALWAYS drop first column (ID)
train_ref_df = drop_first_column(train_ref_df)
# Convert to numeric and fill
train_ref_df = numericize_and_fill(train_ref_df)
train_ref_groups = split_into_groups(train_ref_df)

# -------------------------
# OUTPUT SELECTION
# -------------------------
selected_output = st.selectbox("🎯 Select Output to Predict", [1, 2, 3, 4],
                               format_func=lambda x: OUTPUT_LABEL_MAP[x])
pls_components = PLS_COMPONENTS_MAP[selected_output]
output_name = OUTPUT_NAME_MAP[selected_output]

# -------------------------
# LOAD Top50 details
# -------------------------
detail_file = OUTPUT_DETAILS_DIR / f"{output_name}_Detailed_Results.xlsx"
if not detail_file.exists():
    st.error(f"Detail file not found: {detail_file}")
    st.stop()

detail_df = pd.read_excel(detail_file, sheet_name="Top50_R2")
detail_df["Groups"] = detail_df["Groups"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

st.subheader(f"📘 Top 50 Models for {OUTPUT_LABEL_MAP[selected_output]}")
st.dataframe(detail_df)

# Select rank from Top50
rank_list = list(range(1, len(detail_df) + 1))
selected_rank = st.selectbox("Select Model Rank from Top50", rank_list, index=0)

required_groups = detail_df.iloc[selected_rank - 1]["Groups"]
selected_model_name = detail_df.iloc[selected_rank - 1]["Model"]

# -------------------------
# find & load model file (Top50 folder)
# -------------------------
model_folder = MODELS_DIR / f"{output_name}_Top50_R2"
if not model_folder.exists():
    st.error(f"Model folder not found: {model_folder}")
    st.stop()

model_files = list(model_folder.glob(f"{selected_rank:02d}_{selected_model_name}_R2_*.pkl"))
if not model_files:
    st.error("Model file for selected rank not found.")
    st.stop()

# We won't load model until predict button, but keep path
model_path = model_files[0]

st.success(f"Selected model: Rank {selected_rank} — {selected_model_name}")

# -------------------------
# Feature importance image
# -------------------------
st.subheader("📊 Feature Importance")
feature_img = FEATURE_IMPORTANCE_DIR / output_name / f"{selected_rank:02d}_{selected_model_name}_importance.png"
if feature_img.exists():
    st.image(str(feature_img), use_column_width=True)
    with open(feature_img, "rb") as f:
        st.download_button("📥 Download Feature Importance Image", f, file_name=f"feature_importance_{output_name}.png")
else:
    st.info("Feature importance image not found for this model.")

# -------------------------
# Upload test file (single uploader)
# -------------------------
uploaded_file = st.file_uploader("📁 Upload your Testing Excel file (XLSX)", type=["xlsx"])
test_df = None
test_groups = None

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Select Testing Sheet", xls.sheet_names)
        test_df = pd.read_excel(xls, sheet_name=sheet)
        # drop first ID column always
        test_df = drop_first_column(test_df)
        # convert to numeric & fill
        test_df = numericize_and_fill(test_df)
        test_groups = split_into_groups(test_df)
        st.success("Testing sheet loaded and cleaned.")
    except Exception as e:
        st.error(f"Failed to read/prepare test file: {e}")
        test_df = None

# -------------------------
# Predict button
# -------------------------
if st.button("🚀 Predict Now"):
    if test_df is None or test_groups is None:
        st.error("Please upload a testing Excel file first.")
        st.stop()

    try:
        # load model
        model = joblib.load(model_path)

        # y reference for PLS (from training groups)
        y_train_ref = train_ref_groups["Outputs"].values[:, selected_output - 1]

        # preprocess test groups using training references
        processed_test = preprocess_groups_for_prediction(test_groups, train_ref_groups, y_train_ref, pls_components)

        # Build X_input in the same order as required_groups
        # required_groups is a list of group names (e.g., ['A1','NIR'])
        X_parts = []
        for g in required_groups:
            if g not in processed_test:
                st.error(f"Required group '{g}' not present in test data after splitting.")
                st.stop()
            X_parts.append(processed_test[g])
        X_input = np.hstack(X_parts)

        # predict
        y_pred = model.predict(X_input)

        # prepare output df (only predicted values)
        out_df = pd.DataFrame({
            f"Predicted {OUTPUT_LABEL_MAP[selected_output]}": y_pred
        })

        st.subheader("✅ Predictions")
        st.dataframe(out_df)

        st.download_button("📥 Download Predictions CSV",
                           out_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"Predicted_{output_name}.csv")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
