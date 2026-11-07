# -----------------------------------------------------------
# Isothermal VLE Predictor
# Multi-model predictions display (5 models each)
# -----------------------------------------------------------

import os
import io
import traceback
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ===========================================================
# 1. Page configuration
# ===========================================================
st.set_page_config(
    layout="wide",
    page_title="Isothermal VLE Predictor",
    initial_sidebar_state="collapsed",
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
# 4. Model paths and selected models
# ===========================================================
MODEL_Y1_PATH = "./final_52_y1"
MODEL_P_PATH = "./final_52_p"

# Best models (for phase diagram)
BEST_Y1_MODEL = "NeuralNetTorch_BAG_L1"
BEST_P_MODEL = "NeuralNetFastAI_BAG_L1"

# All models for comparison (5 models each)
Y1_MODELS = [
    "NeuralNetTorch_BAG_L1",
    "NeuralNetTorch_BAG_L2",
    "CatBoost_BAG_L2",
    "WeightedEnsemble_L3",
    "LightGBM_BAG_L2"
]

P_MODELS = [
    "NeuralNetFastAI_BAG_L1",
    "NeuralNetTorch_r79_BAG_L1",
    "LightGBMXT_BAG_L1",
    "NeuralNetTorch_BAG_L1",
    "XGBoost_BAG_L2"
]

# ===========================================================
# 5. Page style
# ===========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    .main-title {
        text-align: center;
        font-size: 1.85rem;
        font-weight: 750;
        color: #202938;
        padding: 0.35rem 0 0.85rem 0;
        margin-bottom: 0.9rem;
        border-bottom: 1px solid #e5e7eb;
        line-height: 1.4;
    }

    .section-header {
        font-size: 1rem;
        font-weight: 700;
        border-left: 4px solid #6c63ff;
        padding-left: 10px;
        margin: 0 0 12px 0;
        line-height: 1.5;
        color: #202938;
    }

    .mol-title {
        text-align: center;
        font-size: 0.86rem;
        font-weight: 650;
        margin: 2px 0 4px 0;
        color: #374151;
    }

    div[data-testid="stImage"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto 4px auto !important;
    }

    div[data-testid="stImage"] img {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        object-fit: contain;
    }

    .stTextInput label,
    .stNumberInput label {
        font-size: 0.84rem !important;
        padding-bottom: 2px !important;
    }

    .stTextInput input,
    .stNumberInput input {
        font-size: 0.88rem !important;
    }

    .stButton > button {
        min-height: 2.65rem;
        margin-top: 0.4rem;
        font-size: 0.93rem;
        font-weight: 650;
        border-radius: 8px;
    }
    
    .stColumn {
        display: flex;
        flex-direction: column;
    }

    @media (max-width: 950px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-title">
        ⚗️ Isothermal VLE Predictor
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 6. Load AutoGluon predictors
# ===========================================================
@st.cache_resource(show_spinner=False)
def load_predictor_y1():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(AUTOGLUON_ERROR)
    if not os.path.isdir(MODEL_Y1_PATH):
        raise FileNotFoundError(f"y1 model directory not found: {MODEL_Y1_PATH}")
    return TabularPredictor.load(MODEL_Y1_PATH)


@st.cache_resource(show_spinner=False)
def load_predictor_p():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(AUTOGLUON_ERROR)
    if not os.path.isdir(MODEL_P_PATH):
        raise FileNotFoundError(f"Pressure model directory not found: {MODEL_P_PATH}")
    return TabularPredictor.load(MODEL_P_PATH)


# ===========================================================
# 7. RDKit descriptor calculator
# ===========================================================
if RDKIT_AVAILABLE:
    desc_names = [item[0] for item in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
else:
    desc_names = []
    calculator = None


def get_mol_features(smiles: str) -> dict:
    """Calculate RDKit descriptors from a SMILES string."""
    if not RDKIT_AVAILABLE:
        raise RuntimeError(f"RDKit is unavailable: {RDKIT_ERROR}")

    smiles = str(smiles).strip()
    if not smiles:
        raise ValueError("SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    descs = calculator.CalcDescriptors(mol)
    clean = []

    for value in descs:
        try:
            value = float(value)
            if not np.isfinite(value):
                value = np.nan
        except (TypeError, ValueError, OverflowError):
            value = np.nan
        clean.append(value)

    return dict(zip(desc_names, clean))


def mol_to_png_bytes(smiles: str, size=(190, 115)):
    """Generate a 2D molecular structure image."""
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


def apply_y1_constraints(y_pred, x_value):
    """Apply 0<=y1<=1 and pure-component boundary constraints."""
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1).copy()
    x_value = np.asarray(x_value, dtype=float).reshape(-1)

    y_pred = np.clip(y_pred, 0.0, 1.0)
    y_pred[x_value == 0.0] = 0.0
    y_pred[x_value == 1.0] = 1.0
    return y_pred


def prepare_model_input(predictor, input_df):
    """Align input columns with the features used during model training."""
    model_features = predictor.feature_metadata.get_features()
    return input_df.reindex(columns=model_features, fill_value=0)


def build_feature_row(feat_A, feat_B, x_value, temperature, psat1, psat2):
    """Build one model-input row for a specified liquid composition."""
    record = {
        "x": float(x_value),
        "T_K": float(temperature),
        "P_sat_1": float(psat1),
        "P_sat_2": float(psat2),
    }

    for key, value in feat_A.items():
        record[f"c1_{key}"] = value

    for key, value in feat_B.items():
        record[f"c2_{key}"] = value

    return record


# ===========================================================
# 8. Optimized Layout
# ===========================================================

col_left_area, col_right_diagram = st.columns([1.67, 1.75], gap="medium")

with col_left_area:
    col_params, col_structures = st.columns([1.05, 0.62], gap="medium")

    with col_params:
        st.markdown(
            '<div class="section-header">System Parameters</div>',
            unsafe_allow_html=True,
        )

        smiles_A = st.text_input(
            "Component A (SMILES):",
            value="CCCO",
            key="smiles_A",
        )

        smiles_B = st.text_input(
            "Component B (SMILES):",
            value="CCC(=O)OC",
            key="smiles_B",
        )

        row_xT = st.columns(2, gap="small")
        with row_xT[0]:
            x_input = st.number_input(
                "Liquid mole fraction x₁:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                format="%.4f",
                key="x_input",
            )

        with row_xT[1]:
            temp_input = st.number_input(
                "Temperature T (K):",
                min_value=200.0,
                max_value=1000.0,
                value=328.15,
                step=0.1,
                format="%.2f",
                key="temp_input",
            )

        row_psat = st.columns(2, gap="small")
        with row_psat[0]:
            psat1_input = st.number_input(
                "P_sat₁ (kPa):",
                min_value=0.0,
                value=15.636161,
                format="%.6f",
                key="psat1_input",
            )

        with row_psat[1]:
            psat2_input = st.number_input(
                "P_sat₂ (kPa):",
                min_value=0.0,
                value=42.169417,
                format="%.6f",
                key="psat2_input",
            )

        submit_button = st.button(
            "Submit & Predict Both",
            type="primary",
            use_container_width=True,
            key="predict_button",
        )

    with col_structures:
        st.markdown(
            '<div class="section-header">Molecular Structures (2D)</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="mol-title">Component A</div>', unsafe_allow_html=True)
        png_A = mol_to_png_bytes(smiles_A, size=(190, 115))
        if png_A:
            st.image(png_A, use_container_width=True)
        else:
            st.caption("Invalid SMILES")

        st.markdown(
            '<div style="height: 10px;"></div>'
            '<div class="mol-title">Component B</div>',
            unsafe_allow_html=True,
        )
        png_B = mol_to_png_bytes(smiles_B, size=(190, 115))
        if png_B:
            st.image(png_B, use_container_width=True)
        else:
            st.caption("Invalid SMILES")
    
    results_container = st.container()

diagram_slot = col_right_diagram.empty()


# ===========================================================
# 9. Prediction Logic
# ===========================================================
if submit_button:
    st.session_state["has_prediction"] = False

    if not RDKIT_AVAILABLE:
        st.error(f"RDKit error: {RDKIT_ERROR}")
        st.stop()

    if not AUTOGLUON_AVAILABLE:
        st.error(f"AutoGluon error: {AUTOGLUON_ERROR}")
        st.stop()

    if not smiles_A.strip() or not smiles_B.strip():
        st.error("Please enter SMILES for both components.")
        st.stop()

    if psat1_input <= 0 or psat2_input <= 0:
        st.error("Both saturation vapor pressures must be greater than zero.")
        st.stop()

    try:
        with st.spinner("Predicting VLE properties and generating the phase diagram..."):
            feat_A = get_mol_features(smiles_A)
            feat_B = get_mol_features(smiles_B)

            pred_y1 = load_predictor_y1()
            pred_P = load_predictor_p()

            # Single-composition predictions
            single_row = build_feature_row(
                feat_A, feat_B, x_input, temp_input,
                psat1_input, psat2_input
            )
            single_df = pd.DataFrame([single_row])

            y_input = prepare_model_input(pred_y1, single_df)
            p_input = prepare_model_input(pred_P, single_df)

            # Predict with all y1 models
            y1_results = {}
            for model_name in Y1_MODELS:
                try:
                    y_pred_raw = float(
                        pred_y1.predict(y_input, model=model_name).iloc[0]
                    )
                    y_pred = float(
                        apply_y1_constraints([y_pred_raw], [x_input])[0]
                    )
                    y1_results[model_name] = y_pred
                except Exception as e:
                    st.warning(f"y₁ Model {model_name} failed: {str(e)}")
                    y1_results[model_name] = np.nan

            # Predict with all P models
            p_results = {}
            for model_name in P_MODELS:
                try:
                    p_pred = float(
                        pred_P.predict(p_input, model=model_name).iloc[0]
                    )
                    p_results[model_name] = p_pred
                except Exception as e:
                    st.warning(f"P Model {model_name} failed: {str(e)}")
                    p_results[model_name] = np.nan

            # Check if best models are valid
            if not np.isfinite(y1_results.get(BEST_Y1_MODEL, np.nan)) or \
               not np.isfinite(p_results.get(BEST_P_MODEL, np.nan)):
                raise ValueError("The selected best models returned non-finite predictions.")

            # Phase-diagram predictions (using best models only)
            x_grid = np.linspace(0.0, 1.0, 20)

            phase_records = [
                build_feature_row(
                    feat_A, feat_B, x_value, temp_input,
                    psat1_input, psat2_input
                )
                for x_value in x_grid
            ]
            phase_input = pd.DataFrame(phase_records)

            phase_y_input = prepare_model_input(pred_y1, phase_input)
            phase_p_input = prepare_model_input(pred_P, phase_input)

            y_grid_raw = np.asarray(
                pred_y1.predict(phase_y_input, model=BEST_Y1_MODEL),
                dtype=float,
            ).reshape(-1)

            p_grid = np.asarray(
                pred_P.predict(phase_p_input, model=BEST_P_MODEL),
                dtype=float,
            ).reshape(-1)

            y_grid = apply_y1_constraints(y_grid_raw, x_grid)

            if (
                len(y_grid) != 20
                or len(p_grid) != 20
                or not np.all(np.isfinite(y_grid))
                or not np.all(np.isfinite(p_grid))
            ):
                raise ValueError(
                    "The selected models did not return 20 valid phase-diagram predictions."
                )

            phase_df = pd.DataFrame({
                "x1 (liquid)": x_grid,
                "y1 (vapor)": y_grid,
                "P (kPa)": p_grid,
            })

            dew_df = phase_df.sort_values("y1 (vapor)").drop_duplicates(
                subset=["y1 (vapor)"], keep="first"
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=phase_df["x1 (liquid)"],
                y=phase_df["P (kPa)"],
                mode="lines",
                name="Bubble-point curve (P-x₁)",
                line=dict(width=3, shape="spline", smoothing=1.0),
                hovertemplate="x₁ = %{x:.4f}<br>P = %{y:.4f} kPa<extra></extra>",
            ))

            fig.add_trace(go.Scatter(
                x=dew_df["y1 (vapor)"],
                y=dew_df["P (kPa)"],
                mode="lines",
                name="Dew-point curve (P-y₁)",
                line=dict(width=3, dash="dash", shape="spline", smoothing=1.0),
                hovertemplate="y₁ = %{x:.4f}<br>P = %{y:.4f} kPa<extra></extra>",
            ))

            fig.update_layout(
                template="plotly_white",
                height=590,
                margin=dict(l=65, r=25, t=105, b=75),
                title=dict(
                    text=f"Isothermal P-x-y Diagram at T = {temp_input:.2f} K",
                    x=0.5,
                    xanchor="center",
                    y=0.98,
                    yanchor="top",
                    font=dict(size=17, family="Arial, sans-serif"),
                ),
                xaxis=dict(
                    title=dict(
                        text="<b>Mole fraction of component 1 (x₁ or y₁)</b>",
                        font=dict(size=14, family="Arial, sans-serif"),
                        standoff=18,
                    ),
                    range=[0, 1],
                    showgrid=True,
                    zeroline=False,
                    tickfont=dict(size=11),
                ),
                yaxis=dict(
                    title=dict(
                        text="<b>Pressure (kPa)</b>",
                        font=dict(size=14, family="Arial, sans-serif"),
                        standoff=12,
                    ),
                    showgrid=True,
                    zeroline=False,
                    tickfont=dict(size=11),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.10,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                    itemwidth=150,
                ),
                hovermode="closest",
            )

            st.session_state["phase_df"] = phase_df
            st.session_state["phase_fig"] = fig
            st.session_state["y1_results"] = y1_results
            st.session_state["p_results"] = p_results
            st.session_state["has_prediction"] = True

    except Exception as e:
        st.error(f"Prediction error: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.stop()

# ===========================================================
# 10. Display Results with Multiple Models (5 rows)
# ===========================================================
if st.session_state.get("has_prediction", False):
    y1_results = st.session_state["y1_results"]
    p_results = st.session_state["p_results"]
    
    with results_container:
        st.markdown(
            '<div style="height: 15px;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-header">Prediction Results</div>',
            unsafe_allow_html=True,
        )

        # Create result table with all 5 models
        result_data = []
        max_models = max(len(Y1_MODELS), len(P_MODELS))
        
        for i in range(max_models):
            row = {}
            
            # y1 model
            if i < len(Y1_MODELS):
                model_name = Y1_MODELS[i]
                row["y₁ model"] = model_name
                row["Predicted y₁"] = y1_results.get(model_name, np.nan)
            else:
                row["y₁ model"] = ""
                row["Predicted y₁"] = ""
            
            # P model
            if i < len(P_MODELS):
                model_name = P_MODELS[i]
                row["P model"] = model_name
                row["Predicted P (kPa)"] = p_results.get(model_name, np.nan)
            else:
                row["P model"] = ""
                row["Predicted P (kPa)"] = ""
            
            result_data.append(row)

        result_df = pd.DataFrame(result_data)

        # Fixed height for 5 rows: header (~40px) + 5 rows (~30px each) = ~190px
        st.dataframe(
            result_df.style.format({
                "Predicted y₁": lambda x: f"{x:.6f}" if pd.notna(x) and x != "" else "",
                "Predicted P (kPa)": lambda x: f"{x:.4f}" if pd.notna(x) and x != "" else "",
            }),
            use_container_width=True,
            hide_index=True,
            height=190,
        )

# ===========================================================
# 11. Display Phase Diagram (Right Side)
# ===========================================================
if st.session_state.get("has_prediction", False):
    with diagram_slot.container():
        st.markdown(
            '<div class="section-header">Predicted P-x-y Phase Diagram</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            st.session_state["phase_fig"],
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "Predicted_Pxy_diagram",
                    "scale": 2,
                },
            },
        )

        with st.expander("View the 20 predicted P-x-y data points"):
            st.dataframe(
                st.session_state["phase_df"].style.format({
                    "x1 (liquid)": "{:.4f}",
                    "y1 (vapor)": "{:.4f}",
                    "P (kPa)": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=735,
            )
else:
    with diagram_slot.container():
        st.markdown(
            '<div class="section-header">Predicted P-x-y Phase Diagram</div>',
            unsafe_allow_html=True,
        )
        st.info("Enter the system parameters and click Submit to generate the P-x-y diagram.")
