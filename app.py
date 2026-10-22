# -----------------------------------------------------------
# Isothermal VLE Predictor (y1 & Pressure)
# Compact dual-target interface with RDKit 2D molecular structures
# -----------------------------------------------------------

import os
import traceback
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Isothermal VLE Predictor",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    RDKIT_ERROR = None
except Exception as e:
    Chem = Draw = Descriptors = MoleculeDescriptors = None
    RDKIT_AVAILABLE = False
    RDKIT_ERROR = str(e)

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
    AUTOGLUON_ERROR = None
except Exception as e:
    TabularPredictor = None
    AUTOGLUON_AVAILABLE = False
    AUTOGLUON_ERROR = str(e)

MODEL_Y1_PATH = "./final_42_y1"
MODEL_P_PATH = "./final_42_p"

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

st.markdown(
    """
    <style>
    /* Keep the interface compact and centered. A fixed viewport cannot be
       guaranteed on every device, but this avoids the previous full-width layout. */
    .block-container {
        max-width: 1180px !important;
        padding-top: 0.8rem !important;
        padding-bottom: 0.55rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    .main { background: #f7f8fa; }
    h1.app-title {
        text-align: center;
        font-size: 1.7rem;
        font-weight: 750;
        color: #20334d;
        background: #e8edf3;
        border-radius: 12px;
        padding: 0.55rem 0.8rem;
        margin: 0 0 0.35rem 0;
    }
    .app-subtitle {
        text-align: center;
        color: #697586;
        font-size: 0.84rem;
        margin: 0 0 0.75rem 0;
    }
    h2, h3 { color: #24364b; }
    div[data-testid="stVerticalBlock"] { gap: 0.45rem; }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label { font-size: 0.82rem; }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input { font-size: 0.88rem; }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        min-height: 2.45rem;
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stDataFrame"] { font-size: 0.78rem; }
    .molecule-card {
        border: 1px solid #e0e5eb;
        background: #ffffff;
        border-radius: 12px;
        padding: 0.45rem;
        text-align: center;
    }
    .molecule-label {
        font-weight: 650;
        color: #334155;
        font-size: 0.88rem;
        margin-bottom: 0.1rem;
    }
    @media (max-height: 850px) {
        .block-container { padding-top: 0.35rem !important; }
        h1.app-title { font-size: 1.45rem; padding: 0.35rem; }
        .app-subtitle { margin-bottom: 0.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="app-title">Isothermal VLE Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Predict vapor-phase mole fraction (y₁) and equilibrium pressure (P) '
    'for binary mixtures using RDKit molecular descriptors and AutoGluon.</div>',
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading vapor-composition model...")
def load_predictor_y1():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(f"AutoGluon import failed: {AUTOGLUON_ERROR}")
    if not os.path.isdir(MODEL_Y1_PATH):
        raise FileNotFoundError(f"Y1 model directory not found: {MODEL_Y1_PATH}")
    return TabularPredictor.load(MODEL_Y1_PATH)

@st.cache_resource(show_spinner="Loading pressure model...")
def load_predictor_p():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(f"AutoGluon import failed: {AUTOGLUON_ERROR}")
    if not os.path.isdir(MODEL_P_PATH):
        raise FileNotFoundError(f"Pressure model directory not found: {MODEL_P_PATH}")
    return TabularPredictor.load(MODEL_P_PATH)

if RDKIT_AVAILABLE:
    desc_names = [item[0] for item in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
else:
    desc_names = []
    calculator = None

def get_mol_features(smiles):
    if not RDKIT_AVAILABLE or calculator is None:
        raise RuntimeError(f"RDKit is unavailable: {RDKIT_ERROR}")
    smiles = str(smiles).strip()
    if not smiles:
        raise ValueError("SMILES cannot be empty.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    values = calculator.CalcDescriptors(mol)
    clean = []
    for value in values:
        try:
            value = float(value)
            if not np.isfinite(value):
                value = np.nan
        except (TypeError, ValueError, OverflowError):
            value = np.nan
        clean.append(value)
    return dict(zip(desc_names, clean)), mol

def apply_y1_constraints(y_pred, x_value):
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1).copy()
    x_value = np.asarray(x_value, dtype=float).reshape(-1)
    if len(y_pred) != len(x_value):
        raise ValueError("Prediction and x arrays have different lengths.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("The y1 model returned a non-finite prediction.")
    y_pred = np.clip(y_pred, 0.0, 1.0)
    y_pred[x_value == 0.0] = 0.0
    y_pred[x_value == 1.0] = 1.0
    return y_pred

def prepare_model_input(predictor, input_df):
    model_features = predictor.feature_metadata.get_features()
    missing = [f for f in model_features if f not in input_df.columns]
    extra = [f for f in input_df.columns if f not in model_features]
    aligned = input_df.reindex(columns=model_features, fill_value=0)
    return aligned, missing, extra

def make_results(predictor, predict_df, model_names, target, x_value=None):
    available = set(predictor.model_names())
    results = []
    for model_name in model_names:
        if model_name not in available:
            results.append({"Model": model_name, target: np.nan})
            continue
        try:
            raw = float(predictor.predict(predict_df, model=model_name).iloc[0])
            if not np.isfinite(raw):
                raise ValueError("Non-finite prediction.")
            if target == "Pred y1":
                raw = float(apply_y1_constraints([raw], [x_value])[0])
            results.append({"Model": model_name, target: raw})
        except Exception:
            results.append({"Model": model_name, target: np.nan})
    return pd.DataFrame(results)

def render_molecule(mol, caption):
    image = Draw.MolToImage(mol, size=(340, 230), kekulize=True)
    st.markdown(f'<div class="molecule-card"><div class="molecule-label">{caption}</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main layout: inputs on the left, 2D molecular structures on the right.
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.subheader("System Inputs")
    smiles_A = st.text_input("Component A — SMILES", value="CCCO", placeholder="e.g. CCO")
    smiles_B = st.text_input("Component B — SMILES", value="CCC(=O)OC", placeholder="e.g. CC(=O)O")
    x_input = st.number_input(
        "Liquid mole fraction of component A (x)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.4f"
    )
    temp_input = st.number_input(
        "Operating temperature (K)",
        min_value=200.0, max_value=1000.0, value=328.15, step=0.1, format="%.2f"
    )
    st.markdown("**Saturation pressures**")
    psat_col1, psat_col2 = st.columns(2, gap="small")
    with psat_col1:
        psat1_input = st.number_input("Psat₁ (kPa)", min_value=0.0, value=15.63616116, format="%.5f")
    with psat_col2:
        psat2_input = st.number_input("Psat₂ (kPa)", min_value=0.0, value=42.16941721, format="%.5f")
    submit_button = st.button("Predict y₁ and Pressure", type="primary", use_container_width=True)

with right:
    st.subheader("Molecular Structures")
    if RDKIT_AVAILABLE:
        preview_col1, preview_col2 = st.columns(2, gap="small")
        try:
            mol_a = Chem.MolFromSmiles(smiles_A.strip()) if smiles_A.strip() else None
            mol_b = Chem.MolFromSmiles(smiles_B.strip()) if smiles_B.strip() else None
            with preview_col1:
                if mol_a is not None:
                    render_molecule(mol_a, "Component A")
                else:
                    st.info("Enter a valid SMILES for A.")
            with preview_col2:
                if mol_b is not None:
                    render_molecule(mol_b, "Component B")
                else:
                    st.info("Enter a valid SMILES for B.")
        except Exception as e:
            st.warning(f"Unable to draw molecular structure: {e}")
    else:
        st.warning("RDKit is unavailable; molecular structure previews cannot be displayed.")

if submit_button:
    if not RDKIT_AVAILABLE:
        st.error(f"RDKit failed to load: {RDKIT_ERROR}")
        st.stop()
    if not AUTOGLUON_AVAILABLE:
        st.error(f"AutoGluon failed to load: {AUTOGLUON_ERROR}")
        st.stop()
    if not smiles_A.strip() or not smiles_B.strip():
        st.error("Please enter SMILES for both components.")
        st.stop()

    with st.spinner("Calculating molecular descriptors and predicting both targets..."):
        try:
            feat_A, _ = get_mol_features(smiles_A)
            feat_B, _ = get_mol_features(smiles_B)
            input_data = {
                "x": float(x_input),
                "T_K": float(temp_input),
                "P_sat_1": float(psat1_input),
                "P_sat_2": float(psat2_input),
            }
            input_data.update({f"c1_{k}": v for k, v in feat_A.items()})
            input_data.update({f"c2_{k}": v for k, v in feat_B.items()})
            input_df = pd.DataFrame([input_data])

            pred_y1 = load_predictor_y1()
            pred_p = load_predictor_p()
            y1_input, y1_missing, _ = prepare_model_input(pred_y1, input_df)
            p_input, p_missing, _ = prepare_model_input(pred_p, input_df)

            result_y1 = make_results(pred_y1, y1_input, Y1_TOP6_MODELS, "Pred y1", x_input)
            result_p = make_results(pred_p, p_input, P_TOP6_MODELS, "Predicted Pressure (kPa)")

            # Keep results compact and remove the Rank and Status fields entirely.
            out_col1, out_col2 = st.columns(2, gap="medium")
            with out_col1:
                st.markdown("#### Vapor-phase mole fraction (y₁)")
                st.dataframe(
                    result_y1.style.format({"Pred y1": "{:.6f}"}),
                    hide_index=True, use_container_width=True, height=252
                )
            with out_col2:
                st.markdown("#### Equilibrium pressure (kPa)")
                st.dataframe(
                    result_p.style.format({"Predicted Pressure (kPa)": "{:.5f}"}),
                    hide_index=True, use_container_width=True, height=252
                )

            # Download both target predictions in one file.
            combined = result_y1.rename(columns={"Model": "y1 Model"}).merge(
                result_p.rename(columns={"Model": "Pressure Model"}),
                left_index=True, right_index=True
            )
            csv_data = combined.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download Both Predictions (CSV)",
                data=csv_data,
                file_name="isothermal_vle_predictions.csv",
                mime="text/csv",
            )

            if y1_missing or p_missing:
                with st.expander("Feature compatibility details"):
                    if y1_missing:
                        st.warning(f"Y1 model: {len(y1_missing)} expected feature(s) were filled with 0.")
                        st.code("\n".join(y1_missing))
                    if p_missing:
                        st.warning(f"Pressure model: {len(p_missing)} expected feature(s) were filled with 0.")
                        st.code("\n".join(p_missing))

            if result_y1["Pred y1"].notna().sum() == 0:
                st.error("No y1 model prediction succeeded. Check model names and saved model files.")
            if result_p["Predicted Pressure (kPa)"].notna().sum() == 0:
                st.error("No pressure model prediction succeeded. Check model names and saved model files.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
