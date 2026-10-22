# -----------------------------------------------------------
# Isothermal VLE Predictor (y1 & Pressure) — Optimized Layout
# Based on RDKit Descriptors and AutoGluon
# -----------------------------------------------------------

import os
import io
import traceback
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ===========================================================
# 1. Page configuration — centered, compact
# ===========================================================
st.set_page_config(
    layout="centered",
    page_title="Isothermal VLE Predictor",
    page_icon="️",
)

# ===========================================================
# 2. Import RDKit
# ===========================================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    from rdkit.ML.Descriptors import MoleculeDescriptors

    RDKIT_AVAILABLE = True
    RDKIT_ERROR = None

except Exception as e:
    Chem = None
    Descriptors = None
    Draw = None
    MoleculeDescriptors = None

    RDKIT_AVAILABLE = False
    RDKIT_ERROR = str(e)

# ===========================================================
# 3. Import AutoGluon
# ===========================================================
try:
    from autogluon.tabular import TabularPredictor

    AUTOGLUON_AVAILABLE = True
    AUTOGLUON_ERROR = None

except Exception as e:
    TabularPredictor = None
    AUTOGLUON_AVAILABLE = False
    AUTOGLUON_ERROR = str(e)

# ===========================================================
# 4. Model paths
# ===========================================================
MODEL_Y1_PATH = "./final_42_y1"
MODEL_P_PATH = "./final_42_p"

# ===========================================================
# 5. Fixed model ranking (best → worst)
# ===========================================================
Y1_TOP6_MODELS = [
    "NeuralNetTorch_r22_BAG_L1",
    "NeuralNetTorch_r41_BAG_L1",
    "NeuralNetTorch_r79_BAG_L1",
    "NeuralNetTorch_r197_BAG_L1",
    "NeuralNetTorch_BAG_L1",
    "NeuralNetTorch_r158_BAG_L1",
]

P_TOP6_MODELS = [
    "XGBoost_r89_BAG_L2",
    "LightGBM_BAG_L2",
    "XGBoost_BAG_L2",
    "LightGBMXT_BAG_L1",
    "LightGBM_r131_BAG_L2",
    "LightGBMLarge_BAG_L2",
]

# ===========================================================
# 6. Page style — fixed viewport, no scroll
# ===========================================================
st.markdown(
    """
    <style>
    /* Lock page height to viewport, hide scrollbars */
    html, body, [class*="css"] {
        overflow: hidden !important;
        height: 100vh !important;
    }
    .stApp {
        max-width: 960px;
        margin: 0 auto;
        padding: 18px 24px 10px 24px;
        box-sizing: border-box;
        overflow: hidden;
    }
    /* Compact headers */
    .stApp h1, .stApp h2, .stApp h3 {
        margin-top: 0;
        margin-bottom: 6px;
    }
    /* Compact inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        font-size: 0.85rem !important;
    }
    /* Compact dataframe */
    .stDataFrame {
        font-size: 0.82rem !important;
    }
    /* Hide default Streamlit footer / menu clutter */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    /* Title block */
    .title-block {
        text-align: center;
        background: linear-gradient(135deg, #e8eaf6, #f3e5f5);
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .title-block h2 {
        margin: 0;
        font-size: 1.35rem;
    }
    .title-block p {
        margin: 4px 0 0 0;
        font-size: 0.82rem;
        color: #555;
    }
    /* Section headers */
    .section-header {
        font-size: 1.0rem;
        font-weight: 700;
        border-left: 4px solid #6c63ff;
        padding-left: 8px;
        margin: 6px 0 8px 0;
    }
    /* Molecule card */
    .mol-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 8px;
        text-align: center;
        margin-bottom: 8px;
    }
    .mol-card .mol-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 2px;
    }
    .mol-card .mol-smiles {
        font-size: 0.72rem;
        color: #888;
        font-family: monospace;
        word-break: break-all;
    }
    /* Champion badge */
    .champion-badge {
        display: inline-block;
        background: #fff3cd;
        color: #856404;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 6px;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 7. Title
# ===========================================================
st.markdown(
    """
    <div class="title-block">
        <h2>⚗️ Isothermal VLE Predictor</h2>
        <p>Predicts vapor-phase mole fraction (y₁) and equilibrium pressure (P)
           for binary mixtures using RDKit descriptors &amp; AutoGluon models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 8. Model loading (cached)
# ===========================================================
@st.cache_resource(show_spinner="Loading y₁ model...")
def load_predictor_y1():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(f"AutoGluon import failed: {AUTOGLUON_ERROR}")
    if not os.path.isdir(MODEL_Y1_PATH):
        raise FileNotFoundError(f"Y1 model directory not found: {MODEL_Y1_PATH}")
    return TabularPredictor.load(MODEL_Y1_PATH)


@st.cache_resource(show_spinner="Loading Pressure model...")
def load_predictor_p():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(f"AutoGluon import failed: {AUTOGLUON_ERROR}")
    if not os.path.isdir(MODEL_P_PATH):
        raise FileNotFoundError(f"Pressure model directory not found: {MODEL_P_PATH}")
    return TabularPredictor.load(MODEL_P_PATH)


# ===========================================================
# 9. RDKit descriptor calculator
# ===========================================================
if RDKIT_AVAILABLE:
    desc_names = [item[0] for item in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
else:
    desc_names = []
    calculator = None


# ===========================================================
# 10. Molecular descriptors
# ===========================================================
def get_mol_features(smiles: str) -> dict:
    if not RDKIT_AVAILABLE or calculator is None:
        raise RuntimeError(f"RDKit is unavailable: {RDKIT_ERROR}")

    smiles = str(smiles).strip()
    if not smiles:
        raise ValueError("SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    descs = calculator.CalcDescriptors(mol)
    clean = []
    for v in descs:
        try:
            v = float(v)
            if not np.isfinite(v):
                v = np.nan
        except (TypeError, ValueError, OverflowError):
            v = np.nan
        clean.append(v)
    return dict(zip(desc_names, clean))


# ===========================================================
# 11. RDKit 2D molecule image (PNG bytes)
# ===========================================================
def mol_to_png_bytes(smiles: str, size: tuple = (220, 180)) -> bytes | None:
    """Return PNG bytes of a 2D RDKit drawing, or None on failure."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ===========================================================
# 12. Y1 physical constraints
# ===========================================================
def apply_y1_constraints(y_pred, x_value):
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1).copy()
    x_value = np.asarray(x_value, dtype=float).reshape(-1)
    if len(y_pred) != len(x_value):
        raise ValueError("Prediction and x arrays have different lengths.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("The Y1 model returned a non-finite prediction.")
    y_pred = np.clip(y_pred, 0.0, 1.0)
    y_pred[x_value == 0.0] = 0.0
    y_pred[x_value == 1.0] = 1.0
    return y_pred


# ===========================================================
# 13. Align input to model features
# ===========================================================
def prepare_model_input(predictor, input_df):
    model_features = predictor.feature_metadata.get_features()
    predict_df = input_df.reindex(columns=model_features, fill_value=0)
    missing = [f for f in model_features if f not in input_df.columns]
    extra = [f for f in input_df.columns if f not in model_features]
    return predict_df, missing, extra


# ===========================================================
# 14. Two-column layout: Inputs | Molecule 2D
# ===========================================================
col_left, col_right = st.columns([1.1, 0.9], gap="small")

with col_left:
    st.markdown('<div class="section-header">System Parameters</div>', unsafe_allow_html=True)

    smiles_A = st.text_input("Component A (SMILES):", value="CCCO", placeholder="e.g., CCO")
    smiles_B = st.text_input("Component B (SMILES):", value="CCC(=O)OC", placeholder="e.g., CCC(=O)OC")

    row2 = st.columns(2)
    with row2[0]:
        x_input = st.number_input(
            "Liquid mole fraction x₁:",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.4f",
        )
    with row2[1]:
        temp_input = st.number_input(
            "Temperature T (K):",
            min_value=200.0, max_value=1000.0, value=328.15, step=0.1, format="%.2f",
        )

    row3 = st.columns(2)
    with row3[0]:
        psat1_input = st.number_input(
            "P_sat₁ (kPa):",
            min_value=0.0, value=15.63616116, format="%.6f",
        )
    with row3[1]:
        psat2_input = st.number_input(
            "P_sat₂ (kPa):",
            min_value=0.0, value=42.16941721, format="%.6f",
        )

    submit_button = st.button("🚀 Submit & Predict Both", type="primary", use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">Molecular Structures (2D)</div>', unsafe_allow_html=True)

    # Component A card
    png_A = mol_to_png_bytes(smiles_A)
    st.markdown(
        f"""
        <div class="mol-card">
            <div class="mol-name">Component A</div>
            <div class="mol-smiles">{smiles_A if smiles_A.strip() else "—"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if png_A:
        st.image(png_A, use_container_width=True)
    else:
        st.info("Enter a valid SMILES to view structure.")

    # Component B card
    png_B = mol_to_png_bytes(smiles_B)
    st.markdown(
        f"""
        <div class="mol-card">
            <div class="mol-name">Component B</div>
            <div class="mol-smiles">{smiles_B if smiles_B.strip() else "—"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if png_B:
        st.image(png_B, use_container_width=True)
    else:
        st.info("Enter a valid SMILES to view structure.")


# ===========================================================
# 15. Main prediction logic — both models at once
# ===========================================================
if submit_button:

    # ---- Dependency checks ----
    if not RDKIT_AVAILABLE:
        st.error("RDKit failed to load."); st.code(RDKIT_ERROR or "Unknown error"); st.stop()
    if not AUTOGLUON_AVAILABLE:
        st.error("AutoGluon failed to load."); st.code(AUTOGLUON_ERROR or "Unknown error"); st.stop()

    # ---- Input validation ----
    if not smiles_A.strip() or not smiles_B.strip():
        st.error("Please enter SMILES for both components."); st.stop()

    with st.spinner("Calculating descriptors & predicting with 12 models (6×y₁ + 6×P)..."):
        try:
            # ---- Step 1. Descriptors ----
            feat_A = get_mol_features(smiles_A)
            feat_B = get_mol_features(smiles_B)

            # ---- Step 2. Assemble input ----
            input_data = {
                "x": float(x_input),
                "T_K": float(temp_input),
                "P_sat_1": float(psat1_input),
                "P_sat_2": float(psat2_input),
            }
            for k, v in feat_A.items():
                input_data[f"c1_{k}"] = v
            for k, v in feat_B.items():
                input_data[f"c2_{k}"] = v
            input_df = pd.DataFrame([input_data])

            # ---- Step 3. Load both predictors ----
            pred_y1 = load_predictor_y1()
            pred_P = load_predictor_p()

            # ---- Step 4. Align features ----
            df_y1, miss_y1, extra_y1 = prepare_model_input(pred_y1, input_df)
            df_P, miss_P, extra_P = prepare_model_input(pred_P, input_df)

            # ---- Step 5. Predict with all 6+6 models ----
            rows = []

            # Y1 models
            for rank, m in enumerate(Y1_TOP6_MODELS, 1):
                try:
                    raw = float(pred_y1.predict(df_y1, model=m).iloc[0])
                    if not np.isfinite(raw):
                        raise ValueError("Non-finite prediction.")
                    constrained = apply_y1_constraints([raw], [x_input])
                    rows.append({
                        "Model": m,
                        "y₁ (vapor mole fraction)": float(constrained[0]),
                        "P (kPa)": "—",
                        "Is Champion": " Y1" if rank == 1 else "",
                    })
                except Exception as e:
                    rows.append({
                        "Model": m,
                        "y₁ (vapor mole fraction)": np.nan,
                        "P (kPa)": "—",
                        "Is Champion": "❌",
                    })

            # Pressure models
            for rank, m in enumerate(P_TOP6_MODELS, 1):
                try:
                    raw = float(pred_P.predict(df_P, model=m).iloc[0])
                    if not np.isfinite(raw):
                        raise ValueError("Non-finite prediction.")
                    rows.append({
                        "Model": m,
                        "y₁ (vapor mole fraction)": "—",
                        "P (kPa)": raw,
                        "Is Champion": "🏆 P" if rank == 1 else "",
                    })
                except Exception as e:
                    rows.append({
                        "Model": m,
                        "y₁ (vapor mole fraction)": "—",
                        "P (kPa)": np.nan,
                        "Is Champion": "❌",
                    })

            results_df = pd.DataFrame(rows)

            # ---- Step 6. Display ----
            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

            y1_champ = Y1_TOP6_MODELS[0]
            p_champ = P_TOP6_MODELS[0]
            st.caption(
                f"🏆 Champion y₁ model: **{y1_champ}** &nbsp;|&nbsp; "
                f"🏆 Champion P model: **{p_champ}**"
            )

            # Highlight champion rows
            def highlight_champ(row):
                styles = [""] * len(row)
                if row["Is Champion"].startswith(""):
                    styles = ["background-color:#fff9c4; font-weight:bold"] * len(row)
                return styles

            display_cols = ["Model", "y₁ (vapor mole fraction)", "P (kPa)", "Is Champion"]
            styled = (
                results_df[display_cols]
                .style
                .apply(highlight_champ, axis=1)
                .format({"y₁ (vapor mole fraction)": "{:.6f}", "P (kPa)": "{:.4f}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # ---- Step 7. Download CSV ----
            csv = results_df[display_cols].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ Download Results (CSV)",
                data=csv,
                file_name="VLE_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
