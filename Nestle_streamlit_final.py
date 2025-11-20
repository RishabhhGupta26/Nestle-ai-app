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
    df_num = df_num.fillna(df_num.median(numeric_only=True))
    return df_num

def safe_get_cols_by_index(df_cols, indices):
    names = []
    for i in indices:
        if 0 <= i < len(df_cols):
            names.append(df_cols[i])
    return names

def adjusted_index(i):
    return i - 1

# -------------------------
# SPLIT GROUPS
# -------------------------
def split_into_groups(df):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)

    try:
        orig_idx = [1, 2, 3, 49]
        adj_idx = [adjusted_index(i) for i in orig_idx]
        output_cols = safe_get_cols_by_index(cols, adj_idx)
        if len(output_cols) != 4:
            output_cols = [c for c in cols if any(k in c.lower() for k in ["filtration","leaching","yield","starch"])]
    except:
        output_cols = [c for c in cols if any(k in c.lower() for k in ["filtration","leaching","yield","starch"])]

    if len(output_cols) != 4:
        st.warning("Select 4 output columns manually")
        selected = st.multiselect("Select 4 output columns", cols)
        if len(selected) != 4:
            st.stop()
        output_cols = selected

    def slice_to_names(s):
        start = adjusted_index(s.start) if s.start else 0
        stop = adjusted_index(s.stop) if s.stop else len(cols)
        return safe_get_cols_by_index(cols, list(range(start, stop)))

    groups = {
        "Outputs": df[output_cols],
        "A1": df[slice_to_names(slice(4,12)) + slice_to_names(slice(15,28))],
        "C1": df[slice_to_names(slice(12,15))],
        "B1": df[slice_to_names(slice(28,47))],
        "E1": df[slice_to_names(slice(47,50))],
        "E2": df[slice_to_names(slice(50,53))],
        "NIR": df[slice_to_names(slice(54,None))]
    }
    return groups

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess_groups_for_prediction(groups, reference_train, y_reference, n_components):
    processed = {}
    for g in [x for x in groups if x not in ["NIR","Outputs"]]:
        X_ref = numericize_and_fill(reference_train[g])
        X_test = numericize_and_fill(groups[g])

        scaler = RobustScaler()
        pt = PowerTransformer(method="yeo-johnson", standardize=False)

        scaler.fit(X_ref.values)
        pt.fit(scaler.transform(X_ref.values))
        processed[g] = pt.transform(scaler.transform(X_test.values))

    nir_train = numericize_and_fill(reference_train["NIR"]).values
    nir_test = numericize_and_fill(groups["NIR"]).values

    if nir_train.size == 0:
        processed["NIR"] = np.zeros((len(groups["Outputs"]),1))
    else:
        n_comp = min(n_components, nir_train.shape[1], max(1,nir_train.shape[0]-1))
        pls = PLSRegression(n_components=n_comp)
        pls.fit(nir_train, y_reference)
        nir_train_s = pls.transform(nir_train)
        nir_test_s = pls.transform(nir_test)

        scaler = RobustScaler()
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        scaler.fit(nir_train_s)
        pt.fit(scaler.transform(nir_train_s))
        processed["NIR"] = pt.transform(scaler.transform(nir_test_s))

    return processed

# -------------------------
# LOAD TRAINING
# -------------------------
train_ref_df = pd.read_excel(TRAIN_REFERENCE_FILE, sheet_name=TRAIN_SHEET)
train_ref_df = drop_first_column(train_ref_df)
train_ref_df = numericize_and_fill(train_ref_df)
train_ref_groups = split_into_groups(train_ref_df)

# -------------------------
# OUTPUT SELECTION
# -------------------------
selected_output = st.selectbox("🎯 Select Output", [1,2,3,4], format_func=lambda x: OUTPUT_LABEL_MAP[x])
output_name = OUTPUT_NAME_MAP[selected_output]
pls_components = PLS_COMPONENTS_MAP[selected_output]

# -------------------------
# TOP50 DETAILS
# -------------------------
# -------------------------
# TOP50 DETAILS  ✅ FULLY FIXED
# -------------------------
detail_file = OUTPUT_DETAILS_DIR / f"{output_name}_Detailed_Results.xlsx"
detail_df = pd.read_excel(detail_file, sheet_name="Top50_R2")

# Convert "Groups" string → actual Python list
detail_df["Groups"] = detail_df["Groups"].apply(lambda x: ast.literal_eval(x))

# --- Fix 2: Remove Output_Index column ---
detail_df = detail_df.drop(columns=["Output_Index"], errors="ignore")

# --- Fix 3: Convert MAPE to % and rename ---
if "Test_MAPE" in detail_df.columns:
    detail_df["Test_MAPE"] = detail_df["Test_MAPE"] * 100
    detail_df = detail_df.rename(columns={"Test_MAPE": "Test_MAPE (% error)"})

# --- Fix 1: Set index to 1–50 instead of 0–49 ---
detail_df.index = detail_df.index + 1

# Show fixed table
st.subheader(f"📘 Top 50 Models for {OUTPUT_LABEL_MAP[selected_output]}")
st.dataframe(detail_df)

# Rank selection (uses new 1–50 index)
selected_rank = st.selectbox("Select Rank", detail_df.index.tolist())

required_groups = detail_df.loc[selected_rank, "Groups"]
selected_model_name = detail_df.loc[selected_rank, "Model"]

model_folder = MODELS_DIR / f"{output_name}_Top50_R2"
model_path = list(model_folder.glob(f"{selected_rank:02d}_{selected_model_name}_R2_*.pkl"))[0]

# -------------------------
# FEATURE IMPORTANCE  ✅ FIXED
# -------------------------
st.subheader("📊 Feature Importance")

feature_img = FEATURE_IMPORTANCE_DIR / output_name / f"{selected_rank:02d}_{selected_model_name}_importance.png"
if feature_img.exists():
    st.image(str(feature_img), use_column_width=True)
else:
    st.info("Feature importance not available for this model.")

# -------------------------
# Upload Test File
# -------------------------
uploaded_file = st.file_uploader("Upload Test Excel", type=["xlsx"])
test_df = None
sample_names = None
actual_df = None

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

    test_df_raw = pd.read_excel(xls, sheet_name=sheet)

    sample_names = test_df_raw.iloc[:,0].astype(str).tolist()

    actual_cols = []
    for idx in [1,2,3,49]:
        try:
            actual_cols.append(test_df_raw.iloc[:,idx])
        except:
            actual_cols.append(pd.Series([np.nan]*len(test_df_raw)))

    actual_df = pd.concat(actual_cols, axis=1)
    actual_df.columns = ["Actual1","Actual2","Actual3","Actual4"]

    test_df = drop_first_column(test_df_raw)
    test_df = numericize_and_fill(test_df)
    test_groups = split_into_groups(test_df)
# -------------------------
# Prediction
# -------------------------
if st.button("🚀 Predict Now"):
    # Run prediction and create the initial result table only once
    model = joblib.load(model_path)

    y_train_ref = train_ref_groups["Outputs"].iloc[:, selected_output - 1].values

    processed_test = preprocess_groups_for_prediction(
        test_groups, train_ref_groups, y_train_ref, pls_components
    )

    X = np.hstack([processed_test[g] for g in required_groups])
    y_pred = model.predict(X)

    # Build table
    result = pd.DataFrame({
        "Sample Name": sample_names,
        "Actual Value": actual_df.iloc[:, selected_output - 1].astype(float),
        f"Predicted {OUTPUT_LABEL_MAP[selected_output]}": y_pred.astype(float)
    })

    result["% Error"] = np.where(
        result["Actual Value"] > 0,
        np.abs(
            (result["Actual Value"] -
             result[f"Predicted {OUTPUT_LABEL_MAP[selected_output]}"])
            / result["Actual Value"]
        ) * 100,
        np.nan
    )

    # Store table in session_state
    st.session_state["result_table"] = result


# -------------------------
# SHOW + EDIT TABLE (PERSISTENT)
# -------------------------
if "result_table" in st.session_state:

    df_edit = st.session_state["result_table"].copy()

    edited = st.data_editor(
        df_edit,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Sample Name": st.column_config.TextColumn(
                "Sample Name",
                disabled=True
            ),
            f"Predicted {OUTPUT_LABEL_MAP[selected_output]}": st.column_config.NumberColumn(
                "Predicted",
                disabled=True,
                format="%.4f"
            ),
            "% Error": st.column_config.NumberColumn(
                "% Error",
                disabled=True,
                format="%.3f"
            ),
            "Actual Value": st.column_config.NumberColumn(
                "Actual Value (Editable)",
                help="Enter your lab value; % Error will update",
                step=0.01,
                format="%.4f"
            )
        }
    )

    # LIVE RECALCULATION of error
    edited["Actual Value"] = pd.to_numeric(edited["Actual Value"], errors="coerce")

    edited["% Error"] = np.where(
        edited["Actual Value"] > 0,
        np.abs(
            (edited["Actual Value"] -
             edited[f"Predicted {OUTPUT_LABEL_MAP[selected_output]}"])
            / edited["Actual Value"]
        ) * 100,
        np.nan
    )

    # Save UPDATE back to session_state
    st.session_state["result_table"] = edited

    # Show final table
    st.write("### Final Results")
    st.dataframe(edited, hide_index=True)

    # Download
    st.download_button(
        "📥 Download Results",
        edited.to_csv(index=False).encode("utf-8"),
        file_name="Predictions.csv"
    )
