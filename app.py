# -----------------------------------------------------------
# Isothermal VLE Predictor
# Compact and Centered UI Version
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
# 4. Model paths
# ===========================================================
MODEL_Y1_PATH = "./final_42_y1"
MODEL_P_PATH = "./final_42_p"

# ===========================================================
# 5. Model lists: Top 3 models for each target
# ===========================================================
Y1_MODELS_FULL = [
    "NeuralNetTorch_r22_BAG_L1",
    "NeuralNetTorch_r41_BAG_L1",
    "NeuralNetTorch_r79_BAG_L1",
    "NeuralNetTorch_r197_BAG_L1",
    "NeuralNetTorch_BAG_L1",
    "NeuralNetTorch_r158_BAG_L1",
]

P_MODELS_FULL = [
    "XGBoost_r89_BAG_L2",
    "LightGBM_BAG_L2",
    "XGBoost_BAG_L2",
    "LightGBMXT_BAG_L1",
    "LightGBM_r131_BAG_L2",
    "LightGBMLarge_BAG_L2",
]

Y1_MODELS = Y1_MODELS_FULL[:3]
P_MODELS = P_MODELS_FULL[:3]

# ===========================================================
# 6. Compact page style
# ===========================================================
st.markdown(
    """
    <style>
    /* ---------- Overall page ---------- */

    .block-container {
        max-width: 1000px !important;
        padding-top: 0.65rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Reduce default vertical gaps between Streamlit elements */
    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.8rem;
        align-items: flex-start;
    }

    /* ---------- Main title ---------- */

    .title-block {
        text-align: center;
        background: #f0f2f6;
        padding: 7px 8px;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    .title-block h2 {
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.35;
        margin: 0;
        padding: 0;
        color: #202938;
    }

    /* ---------- Section headings ---------- */

    .section-header {
        font-size: 0.92rem;
        font-weight: 700;
        line-height: 1.3;
        border-left: 4px solid #6c63ff;
        padding-left: 8px;
        margin: 1px 0 5px 0;
        color: #202938;
    }

    /* ---------- Input labels ---------- */

    .stTextInput label,
    .stNumberInput label {
        font-size: 0.78rem !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    /* ---------- Input boxes ---------- */

    .stTextInput,
    .stNumberInput {
        margin-top: 0 !important;
        margin-bottom: 0.1rem !important;
    }

    .stTextInput > div,
    .stNumberInput > div {
        margin-top: 0 !important;
    }

    .stTextInput input,
    .stNumberInput input {
        font-size: 0.8rem !important;
    }

    /* Compact input field height */
    .stTextInput div[data-baseweb="input"],
    .stNumberInput div[data-baseweb="input"] {
        min-height: 2.15rem !important;
    }

    /* ---------- Molecular structure images ---------- */

    .mol-title {
        text-align: center;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0 0 1px 0;
        color: #333333;
    }

    /* Center the image container and image itself */
    div[data-testid="stImage"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto 2px auto !important;
    }

    div[data-testid="stImage"] img {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        object-fit: contain;
    }

    /* Center invalid-SMILES messages */
    div[data-testid="stCaptionContainer"] {
        text-align: center;
    }

    /* ---------- Prediction button ---------- */

    .stButton {
        margin-top: 1px;
        margin-bottom: 2px;
    }

    .stButton > button {
        min-height: 2.35rem;
        padding: 0.35rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 7px;
    }

    /* ---------- Results table ---------- */

    div[data-testid="stDataFrame"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    /* ---------- Download button ---------- */

    .stDownloadButton {
        margin-top: 1px !important;
    }

    .stDownloadButton > button {
        min-height: 2.1rem;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
    }

    /* ---------- Reduce excess paragraph spacing ---------- */

    .stMarkdown p {
        margin-bottom: 0.15rem;
    }

    /* ---------- Responsive layout ---------- */

    @media (max-width: 700px) {
        .block-container {
            padding-top: 0.5rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }

        .title-block h2 {
            font-size: 1.1rem;
        }

        .section-header {
            font-size: 0.82rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 7. Main title
# ===========================================================
st.markdown(
    """
    <div class="title-block">
        <h2>⚗️ Isothermal VLE Predictor</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 8. Load AutoGluon models
# ===========================================================
@st.cache_resource(show_spinner=False)
def load_predictor_y1():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(AUTOGLUON_ERROR)

    if not os.path.isdir(MODEL_Y1_PATH):
        raise FileNotFoundError(
            f"y₁ model directory not found: {MODEL_Y1_PATH}"
        )

    return TabularPredictor.load(MODEL_Y1_PATH)


@st.cache_resource(show_spinner=False)
def load_predictor_p():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(AUTOGLUON_ERROR)

    if not os.path.isdir(MODEL_P_PATH):
        raise FileNotFoundError(
            f"Pressure model directory not found: {MODEL_P_PATH}"
        )

    return TabularPredictor.load(MODEL_P_PATH)


# ===========================================================
# 9. RDKit descriptor calculator
# ===========================================================
if RDKIT_AVAILABLE:
    desc_names = [
        item[0] for item in Descriptors._descList
    ]

    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
        desc_names
    )
else:
    desc_names = []
    calculator = None


def get_mol_features(smiles: str) -> dict:
    """Calculate RDKit molecular descriptors from SMILES."""

    if not RDKIT_AVAILABLE:
        raise RuntimeError(
            f"RDKit is unavailable: {RDKIT_ERROR}"
        )

    smiles = str(smiles).strip()

    if not smiles:
        raise ValueError("SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

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


def mol_to_png_bytes(
    smiles: str,
    size: tuple = (180, 105),
):
    """Generate a 2D molecular structure image."""

    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    try:
        img = Draw.MolToImage(
            mol,
            size=size,
        )

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return buf.getvalue()

    except Exception:
        return None


def apply_y1_constraints(y_pred, x_value):
    """
    Apply physical boundary constraints:
    0 <= y1 <= 1
    x1 = 0 -> y1 = 0
    x1 = 1 -> y1 = 1
    """

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1).copy()

    x_value = np.asarray(
        x_value,
        dtype=float,
    ).reshape(-1)

    y_pred = np.clip(
        y_pred,
        0.0,
        1.0,
    )

    y_pred[x_value == 0.0] = 0.0
    y_pred[x_value == 1.0] = 1.0

    return y_pred


def prepare_model_input(predictor, input_df):
    """Align input columns with the trained model features."""

    model_features = (
        predictor.feature_metadata.get_features()
    )

    model_input = input_df.reindex(
        columns=model_features,
        fill_value=0,
    )

    return model_input, [], []


# ===========================================================
# 10. Main layout: 1:1 ratio
# ===========================================================
col_left, col_right = st.columns(
    [1, 1],
    gap="medium",
)

# -----------------------------------------------------------
# Left: System Parameters
# -----------------------------------------------------------
with col_left:

    st.markdown(
        '<div class="section-header">'
        'System Parameters'
        '</div>',
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

    # Liquid mole fraction and temperature
    row2 = st.columns(2, gap="small")

    with row2[0]:
        x_input = st.number_input(
            "Liquid mole fraction x₁:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.4f",
            key="x_input",
        )

    with row2[1]:
        temp_input = st.number_input(
            "Temperature T (K):",
            min_value=200.0,
            max_value=1000.0,
            value=328.15,
            step=0.1,
            format="%.2f",
            key="temp_input",
        )

    # Saturation vapor pressures
    row3 = st.columns(2, gap="small")

    with row3[0]:
        psat1_input = st.number_input(
            "P_sat₁ (kPa):",
            min_value=0.0,
            value=15.636161,
            format="%.6f",
            key="psat1_input",
        )

    with row3[1]:
        psat2_input = st.number_input(
            "P_sat₂ (kPa):",
            min_value=0.0,
            value=42.169417,
            format="%.6f",
            key="psat2_input",
        )

    submit_button = st.button(
        "🚀 Submit & Predict Both",
        type="primary",
        use_container_width=True,
        key="predict_button",
    )


# -----------------------------------------------------------
# Right: Molecular Structures
# -----------------------------------------------------------
with col_right:

    st.markdown(
        '<div class="section-header">'
        'Molecular Structures (2D)'
        '</div>',
        unsafe_allow_html=True,
    )

    # Component A
    st.markdown(
        '<div class="mol-title">Component A</div>',
        unsafe_allow_html=True,
    )

    png_A = mol_to_png_bytes(
        smiles_A,
        size=(180, 105),
    )

    if png_A:
        st.image(
            png_A,
            width=180,
        )
    else:
        st.caption("Invalid SMILES for Component A")

    # Component B
    st.markdown(
        '<div class="mol-title">Component B</div>',
        unsafe_allow_html=True,
    )

    png_B = mol_to_png_bytes(
        smiles_B,
        size=(180, 105),
    )

    if png_B:
        st.image(
            png_B,
            width=180,
        )
    else:
        st.caption("Invalid SMILES for Component B")


# ===========================================================
# 11. Prediction logic
# ===========================================================
if submit_button:

    # Check dependencies and inputs
    if not RDKIT_AVAILABLE:
        st.error(
            f"RDKit error: {RDKIT_ERROR}"
        )
        st.stop()

    if not AUTOGLUON_AVAILABLE:
        st.error(
            f"AutoGluon error: {AUTOGLUON_ERROR}"
        )
        st.stop()

    if not smiles_A.strip() or not smiles_B.strip():
        st.error(
            "Please enter valid SMILES for both components."
        )
        st.stop()

    try:

        with st.spinner("Predicting VLE properties..."):

            # Calculate descriptors for both components
            feat_A = get_mol_features(smiles_A)
            feat_B = get_mol_features(smiles_B)

            # Construct the model input
            input_data = {
                "x": float(x_input),
                "T_K": float(temp_input),
                "P_sat_1": float(psat1_input),
                "P_sat_2": float(psat2_input),
            }

            for key, value in feat_A.items():
                input_data[f"c1_{key}"] = value

            for key, value in feat_B.items():
                input_data[f"c2_{key}"] = value

            input_df = pd.DataFrame(
                [input_data]
            )

            # Load trained predictors
            pred_y1 = load_predictor_y1()
            pred_P = load_predictor_p()

            # Match the trained model feature columns
            df_y1, _, _ = prepare_model_input(
                pred_y1,
                input_df,
            )

            df_P, _, _ = prepare_model_input(
                pred_P,
                input_df,
            )

            rows = []

            # ------------------------------------------------
            # y1 predictions: Top 3 models
            # ------------------------------------------------
            for model_name in Y1_MODELS:

                try:
                    raw_prediction = float(
                        pred_y1.predict(
                            df_y1,
                            model=model_name,
                        ).iloc[0]
                    )

                    if not np.isfinite(raw_prediction):
                        raise ValueError(
                            "Prediction is not finite."
                        )

                    prediction = float(
                        apply_y1_constraints(
                            [raw_prediction],
                            [x_input],
                        )[0]
                    )

                    rows.append({
                        "Model": model_name,
                        "y₁": prediction,
                        "P (kPa)": np.nan,
                    })

                except Exception:
                    rows.append({
                        "Model": model_name,
                        "y₁": np.nan,
                        "P (kPa)": np.nan,
                    })

            # ------------------------------------------------
            # Pressure predictions: Top 3 models
            # ------------------------------------------------
            for model_name in P_MODELS:

                try:
                    prediction = float(
                        pred_P.predict(
                            df_P,
                            model=model_name,
                        ).iloc[0]
                    )

                    if not np.isfinite(prediction):
                        raise ValueError(
                            "Prediction is not finite."
                        )

                    rows.append({
                        "Model": model_name,
                        "y₁": np.nan,
                        "P (kPa)": prediction,
                    })

                except Exception:
                    rows.append({
                        "Model": model_name,
                        "y₁": np.nan,
                        "P (kPa)": np.nan,
                    })

            # Build results table
            results_df = pd.DataFrame(
                rows,
                columns=[
                    "Model",
                    "y₁",
                    "P (kPa)",
                ],
            )

        # ===================================================
        # 12. Display prediction results
        # ===================================================
        st.markdown(
            '<div class="section-header">'
            'Prediction Results (Top 3 Models Each)'
            '</div>',
            unsafe_allow_html=True,
        )

        display_cols = [
            "Model",
            "y₁",
            "P (kPa)",
        ]

        styled = (
            results_df[display_cols]
            .style
            .format(
                {
                    "y₁": "{:.6f}",
                    "P (kPa)": "{:.4f}",
                },
                na_rep="—",
            )
        )

        # Six prediction rows plus the header
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=255,
        )

        # ===================================================
        # 13. Download prediction results
        # ===================================================
        csv = (
            results_df[display_cols]
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="VLE_results.csv",
            mime="text/csv",
            key="download_results",
        )

        # Warn if any model prediction failed
        failed_models = results_df[
            results_df[["y₁", "P (kPa)"]]
            .isna()
            .all(axis=1)
        ]["Model"].tolist()

        if failed_models:
            st.warning(
                "Some models did not return predictions: "
                + ", ".join(failed_models)
                + ". Please check the model files and dependencies."
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        with st.expander("Technical details"):
            st.code(
                traceback.format_exc()
            )
