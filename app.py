# -----------------------------------------------------------
# Isothermal VLE Predictor (Optimized: Compact & Simultaneous)
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
# 1. Page configuration
# ===========================================================
st.set_page_config(
    layout="centered",
    page_title="Isothermal VLE Predictor",
    page_icon="⚗️",
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
    Chem = None; Descriptors = None; Draw = None; MoleculeDescriptors = None
    RDKIT_AVAILABLE = False; RDKIT_ERROR = str(e)

# ===========================================================
# 3. Import AutoGluon
# ===========================================================
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
    AUTOGLUON_ERROR = None
except Exception as e:
    TabularPredictor = None
    AUTOGLUON_AVAILABLE = False; AUTOGLUON_ERROR = str(e)

# ===========================================================
# 4. Model paths
# ===========================================================
MODEL_Y1_PATH = "./final_42_y1"
MODEL_P_PATH = "./final_42_p"

# ===========================================================
# 5. Model lists (We will only use the top 3 of each)
# ===========================================================
Y1_MODELS_FULL = [
    "NeuralNetTorch_r22_BAG_L1", "NeuralNetTorch_r41_BAG_L1",
    "NeuralNetTorch_r79_BAG_L1", "NeuralNetTorch_r197_BAG_L1",
    "NeuralNetTorch_BAG_L1", "NeuralNetTorch_r158_BAG_L1",
]
P_MODELS_FULL = [
    "XGBoost_r89_BAG_L2", "LightGBM_BAG_L2", "XGBoost_BAG_L2",
    "LightGBMXT_BAG_L1", "LightGBM_r131_BAG_L2", "LightGBMLarge_BAG_L2",
]

# Take only top 3 as requested
Y1_MODELS = Y1_MODELS_FULL[:3]
P_MODELS = P_MODELS_FULL[:3]

# ===========================================================
# 6. Page style (Compact & Fixed Height)
# ===========================================================
st.markdown(
    """
    <style>
    /* Force no scroll and compact layout */
    html, body, [class*="css"] {
        overflow: hidden !important;
        height: 100vh !important;
    }
    .stApp {
        max-width: 900px;
        margin: 0 auto;
        padding: 10px 20px;
        box-sizing: border-box;
        overflow: hidden;
    }
    /* Compact headers */
    .stApp h1, .stApp h2, .stApp h3 {
        margin-top: 0; margin-bottom: 5px;
        font-size: 1.1rem;
    }
    .stApp p { margin-bottom: 5px; }
    
    /* Compact inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        font-size: 0.8rem !important;
        padding: 0.2rem 0.5rem !important;
    }
    .stNumberInput label, .stTextInput label {
        font-size: 0.85rem;
        margin-bottom: 2px;
    }
    
    /* Title block */
    .title-block {
        text-align: center;
        background: #f0f2f6;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .title-block h2 { font-size: 1.2rem; margin: 0; }
    .title-block p { font-size: 0.75rem; color: #666; margin: 2px 0 0 0; }
    
    /* Section headers */
    .section-header {
        font-size: 0.95rem;
        font-weight: 700;
        border-left: 4px solid #6c63ff;
        padding-left: 8px;
        margin: 5px 0 8px 0;
    }
    
    /* Molecule card */
    .mol-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 4px;
        text-align: center;
        margin-bottom: 5px;
    }
    .mol-name { font-size: 0.75rem; font-weight: 600; }
    .mol-smiles { font-size: 0.65rem; color: #888; font-family: monospace; }
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
        <p>Predicts y₁ and P for binary mixtures using RDKit & AutoGluon.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 8. Model loading
# ===========================================================
@st.cache_resource(show_spinner="Loading y₁ model...")
def load_predictor_y1():
    if not AUTOGLUON_AVAILABLE: raise ImportError(AUTOGLUON_ERROR)
    if not os.path.isdir(MODEL_Y1_PATH): raise FileNotFoundError(MODEL_Y1_PATH)
    return TabularPredictor.load(MODEL_Y1_PATH)

@st.cache_resource(show_spinner="Loading Pressure model...")
def load_predictor_p():
    if not AUTOGLUON_AVAILABLE: raise ImportError(AUTOGLUON_ERROR)
    if not os.path.isdir(MODEL_P_PATH): raise FileNotFoundError(MODEL_P_PATH)
    return TabularPredictor.load(MODEL_P_PATH)

# ===========================================================
# 9. RDKit setup
# ===========================================================
if RDKIT_AVAILABLE:
    desc_names = [item[0] for item in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
else:
    desc_names = []; calculator = None

def get_mol_features(smiles: str) -> dict:
    if not RDKIT_AVAILABLE: raise RuntimeError(RDKIT_ERROR)
    smiles = str(smiles).strip()
    if not smiles: raise ValueError("SMILES empty")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    descs = calculator.CalcDescriptors(mol)
    clean = []
    for v in descs:
        try:
            v = float(v)
            if not np.isfinite(v): v = np.nan
        except: v = np.nan
        clean.append(v)
    return dict(zip(desc_names, clean))

def mol_to_png_bytes(smiles: str, size: tuple = (160, 120)) -> bytes | None:
    if not RDKIT_AVAILABLE: return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None: return None
    try:
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return buf.getvalue()
    except: return None

def apply_y1_constraints(y_pred, x_value):
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1).copy()
    x_value = np.asarray(x_value, dtype=float).reshape(-1)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    y_pred[x_value == 0.0] = 0.0; y_pred[x_value == 1.0] = 1.0
    return y_pred

def prepare_model_input(predictor, input_df):
    model_features = predictor.feature_metadata.get_features()
    return input_df.reindex(columns=model_features, fill_value=0), [], []

# ===========================================================
# 10. Layout: Inputs (Left) | Molecules (Right)
# ===========================================================
col_left, col_right = st.columns([1.2, 0.8], gap="small")

with col_left:
    st.markdown('<div class="section-header">System Parameters</div>', unsafe_allow_html=True)
    
    smiles_A = st.text_input("Component A (SMILES):", value="CCCO")
    smiles_B = st.text_input("Component B (SMILES):", value="CCC(=O)OC")
    
    row2 = st.columns(2)
    with row2[0]: x_input = st.number_input("Liquid mole fraction x₁:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.4f")
    with row2[1]: temp_input = st.number_input("Temperature T (K):", min_value=200.0, max_value=1000.0, value=328.15, step=0.1, format="%.2f")
    
    row3 = st.columns(2)
    with row3[0]: psat1_input = st.number_input("P_sat₁ (kPa):", min_value=0.0, value=15.636161, format="%.6f")
    with row3[1]: psat2_input = st.number_input("P_sat₂ (kPa):", min_value=0.0, value=42.169417, format="%.6f")
    
    submit_button = st.button("🚀 Submit & Predict Both", type="primary", use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">Molecular Structures (2D)</div>', unsafe_allow_html=True)
    
    # Component A
    png_A = mol_to_png_bytes(smiles_A, size=(160, 120))
    st.markdown(f'<div class="mol-card"><div class="mol-name">Component A</div><div class="mol-smiles">{smiles_A[:15]}...</div></div>', unsafe_allow_html=True)
    if png_A: st.image(png_A, width=160) # Fixed width to keep it small
    else: st.caption("Invalid SMILES A")

    # Component B
    png_B = mol_to_png_bytes(smiles_B, size=(160, 120))
    st.markdown(f'<div class="mol-card"><div class="mol-name">Component B</div><div class="mol-smiles">{smiles_B[:15]}...</div></div>', unsafe_allow_html=True)
    if png_B: st.image(png_B, width=160) # Fixed width to keep it small
    else: st.caption("Invalid SMILES B")


# ===========================================================
# 11. Main prediction logic
# ===========================================================
if submit_button:
    if not RDKIT_AVAILABLE: st.error("RDKit error"); st.stop()
    if not AUTOGLUON_AVAILABLE: st.error("AutoGluon error"); st.stop()
    if not smiles_A.strip() or not smiles_B.strip(): st.error("Enter SMILES"); st.stop()

    with st.spinner("Predicting top 3 models for y₁ and P..."):
        try:
            feat_A = get_mol_features(smiles_A)
            feat_B = get_mol_features(smiles_B)
            
            input_data = {"x": float(x_input), "T_K": float(temp_input), "P_sat_1": float(psat1_input), "P_sat_2": float(psat2_input)}
            for k, v in feat_A.items(): input_data[f"c1_{k}"] = v
            for k, v in feat_B.items(): input_data[f"c2_{k}"] = v
            input_df = pd.DataFrame([input_data])

            pred_y1 = load_predictor_y1()
            pred_P = load_predictor_p()
            
            df_y1, _, _ = prepare_model_input(pred_y1, input_df)
            df_P, _, _ = prepare_model_input(pred_P, input_df)

            rows = []
            
            # Y1 Models (Top 3)
            for m in Y1_MODELS:
                try:
                    raw = float(pred_y1.predict(df_y1, model=m).iloc[0])
                    if not np.isfinite(raw): raise ValueError("Non-finite")
                    val = float(apply_y1_constraints([raw], [x_input])[0])
                    rows.append({"Model": m, "y₁": val, "P (kPa)": np.nan})
                except:
                    rows.append({"Model": m, "y₁": np.nan, "P (kPa)": np.nan})

            # Pressure Models (Top 3)
            for m in P_MODELS:
                try:
                    raw = float(pred_P.predict(df_P, model=m).iloc[0])
                    if not np.isfinite(raw): raise ValueError("Non-finite")
                    rows.append({"Model": m, "y₁": np.nan, "P (kPa)": raw})
                except:
                    rows.append({"Model": m, "y₁": np.nan, "P (kPa)": np.nan})

            results_df = pd.DataFrame(rows)
            
            st.markdown('<div class="section-header">Prediction Results (Top 3 Models)</div>', unsafe_allow_html=True)
            
            # Simple table, no champion column
            display_cols = ["Model", "y₁", "P (kPa)"]
            styled = (
                results_df[display_cols]
                .style
                .format({"y₁": "{:.6f}", "P (kPa)": "{:.4f}"}, na_rep="—")
            )
            st.dataframe(styled, use_container_width=True, hide_index=True, height=180)

            # Download
            csv = results_df[display_cols].fillna("—").to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Download CSV", data=csv, file_name="VLE_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
            with st.expander("Details"): st.code(traceback.format_exc())
