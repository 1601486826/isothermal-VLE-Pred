# -----------------------------------------------------------
# Isothermal VLE Predictor (y1 & Pressure)
# Based on RDKit Descriptors and AutoGluon
# -----------------------------------------------------------

import os
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
    layout="wide",
    page_title="Isothermal VLE Predictor"
)

# ===========================================================
# 2. Import RDKit
# ===========================================================
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    RDKIT_AVAILABLE = True
    RDKIT_ERROR = None

except Exception as e:
    rdkit = None
    Chem = None
    Descriptors = None
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
# 5. Fixed model ranking: best to worst
#
# The order below is the display ranking.
# The first model is designated as the champion.
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
# 6. Page style
# ===========================================================
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 20px auto;
        max-width: 100%;
        background-color: #f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }

    .rounded-container h2 {
        text-align: center;
        background-color: #e0e0e0;
        padding: 12px;
        border-radius: 10px;
        margin-top: 0;
    }

    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 12px;
        font-size: 1.05em;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 7. Page title
# ===========================================================
st.markdown(
    """
    <div class="rounded-container">
        <h2>Isothermal VLE Predictor (y1 & Pressure)</h2>
        <blockquote>
            This web app predicts the vapor-phase mole fraction (y1)
            and equilibrium pressure (P) for binary mixtures using
            RDKit molecular descriptors and trained AutoGluon models.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 8. Model loading
# ===========================================================
@st.cache_resource(show_spinner="Loading Y1 Model...")
def load_predictor_y1():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(
            f"AutoGluon import failed: {AUTOGLUON_ERROR}"
        )

    if not os.path.isdir(MODEL_Y1_PATH):
        raise FileNotFoundError(
            f"Y1 model directory not found: {MODEL_Y1_PATH}"
        )

    return TabularPredictor.load(MODEL_Y1_PATH)


@st.cache_resource(show_spinner="Loading Pressure Model...")
def load_predictor_p():
    if not AUTOGLUON_AVAILABLE:
        raise ImportError(
            f"AutoGluon import failed: {AUTOGLUON_ERROR}"
        )

    if not os.path.isdir(MODEL_P_PATH):
        raise FileNotFoundError(
            f"Pressure model directory not found: {MODEL_P_PATH}"
        )

    return TabularPredictor.load(MODEL_P_PATH)


# ===========================================================
# 9. RDKit descriptor initialization
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


# ===========================================================
# 10. Calculate molecular descriptors
# ===========================================================
def get_mol_features(smiles):
    """
    Convert a SMILES string into RDKit molecular descriptors.
    """
    if not RDKIT_AVAILABLE or calculator is None:
        raise RuntimeError(
            f"RDKit is unavailable: {RDKIT_ERROR}"
        )

    smiles = str(smiles).strip()

    if not smiles:
        raise ValueError("SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES string: {smiles}"
        )

    descs = calculator.CalcDescriptors(mol)

    clean_descs = []

    for value in descs:
        try:
            value = float(value)

            if not np.isfinite(value):
                value = np.nan

        except (TypeError, ValueError, OverflowError):
            value = np.nan

        clean_descs.append(value)

    return dict(zip(desc_names, clean_descs))


# ===========================================================
# 11. Y1 physical constraints
# ===========================================================
def apply_y1_constraints(y_pred, x_value):
    """
    Apply physical constraints to vapor-phase mole fraction.

    1. Clip predictions to [0, 1].
    2. If x == 0, set y = 0.
    3. If x == 1, set y = 1.
    """
    y_pred = np.asarray(
        y_pred, dtype=float
    ).reshape(-1).copy()

    x_value = np.asarray(
        x_value, dtype=float
    ).reshape(-1)

    if len(y_pred) != len(x_value):
        raise ValueError(
            "Prediction and x arrays have different lengths."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "The Y1 model returned a non-finite prediction."
        )

    y_pred = np.clip(y_pred, 0.0, 1.0)

    y_pred[x_value == 0.0] = 0.0
    y_pred[x_value == 1.0] = 1.0

    return y_pred


# ===========================================================
# 12. Validate model feature names
# ===========================================================
def prepare_model_input(predictor, input_df):
    """
    Align input columns to the features expected by the predictor.

    Missing columns are filled with 0.
    Verify training/deployment feature consistency if any are missing.
    """
    model_features = predictor.feature_metadata.get_features()

    missing_features = [
        feature
        for feature in model_features
        if feature not in input_df.columns
    ]

    extra_features = [
        feature
        for feature in input_df.columns
        if feature not in model_features
    ]

    predict_df = input_df.reindex(
        columns=model_features,
        fill_value=0
    )

    return predict_df, missing_features, extra_features


# ===========================================================
# 13. Display formatting
# ===========================================================
def display_results_table(results_df, target_name):
    """
    Display results in fixed ranking order.

    Champion:
        The first model in the corresponding model list.

    Y1:
        Model | Pred y1 | Status

    Pressure:
        Model | Predicted Pressure (kPa) | Status
    """
    results_df = results_df.copy()

    # Remove raw predictions if present.
    results_df = results_df.drop(
        columns=["Raw Prediction", "Constrained y1"],
        errors="ignore"
    )

    # Use the requested Y1 column name.
    if target_name == "y1":
        if "Pred y1" not in results_df.columns:
            raise ValueError(
                "The results table is missing the 'Pred y1' column."
            )

        display_columns = [
            "Rank",
            "Model",
            "Pred y1",
            "Status",
        ]

    else:
        display_columns = [
            "Rank",
            "Model",
            "Predicted Pressure (kPa)",
            "Status",
        ]

    # Keep only the intended display columns.
    results_df = results_df[display_columns]

    # Bold the champion row.
    def highlight_champion(row):
        if row.name == 0:
            return [
                "font-weight: bold"
                for _ in row
            ]

        return ["" for _ in row]

    styled_df = (
        results_df.style
        .apply(highlight_champion, axis=1)
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )


# ===========================================================
# 14. Input interface
# ===========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Parameters")

    model_choice = st.selectbox(
        "Select Target to Predict:",
        [
            "y1 (Vapor mole fraction)",
            "P (Equilibrium pressure)"
        ]
    )

    smiles_A = st.text_input(
        "Component A (Input SMILES):",
        value="CCCO",
        placeholder="e.g., CCO"
    )

    smiles_B = st.text_input(
        "Component B (Input SMILES):",
        value="CCC(=O)OC",
        placeholder="e.g., O"
    )

    x_input = st.number_input(
        "Liquid-phase mole fraction of component A (x):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.4f"
    )

    temp_input = st.number_input(
        "Operating Temperature (T_K):",
        min_value=200.0,
        max_value=1000.0,
        value=328.15,
        step=0.1,
        format="%.2f"
    )

with col2:
    st.subheader("Saturation Pressures")

    psat1_input = st.number_input(
        "Psat1 (kPa):",
        min_value=0.0,
        value=15.63616116,
        format="%.8f"
    )

    psat2_input = st.number_input(
        "Psat2 (kPa):",
        min_value=0.0,
        value=42.16941721,
        format="%.8f"
    )

submit_button = st.button(
    "Submit and Predict",
    type="primary",
    use_container_width=True
)


# ===========================================================
# 15. Main prediction logic
# ===========================================================
if submit_button:

    # -------------------- Dependency checks --------------------
    if not RDKIT_AVAILABLE:
        st.error("RDKit failed to load.")
        st.code(RDKIT_ERROR or "Unknown RDKit error")
        st.stop()

    if not AUTOGLUON_AVAILABLE:
        st.error("AutoGluon failed to load.")
        st.code(AUTOGLUON_ERROR or "Unknown AutoGluon error")
        st.stop()

    # -------------------- Input validation --------------------
    if not smiles_A.strip() or not smiles_B.strip():
        st.error(
            "Please enter SMILES for both components."
        )
        st.stop()

    with st.spinner(
        "Calculating descriptors and predicting with six models..."
    ):
        try:
            # -----------------------------------------------
            # Step 1. Calculate RDKit descriptors
            # -----------------------------------------------
            feat_A = get_mol_features(smiles_A)
            feat_B = get_mol_features(smiles_B)

            # -----------------------------------------------
            # Step 2. Assemble input features
            # -----------------------------------------------
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

            input_df = pd.DataFrame([input_data])

            # -----------------------------------------------
            # Step 3. Load the selected Predictor
            # -----------------------------------------------
            if model_choice.startswith("y1"):
                predictor = load_predictor_y1()
                target_name = "y1"
                selected_models = Y1_TOP6_MODELS
            else:
                predictor = load_predictor_p()
                target_name = "Pressure"
                selected_models = P_TOP6_MODELS

            # -----------------------------------------------
            # Step 4. Align model input features
            # -----------------------------------------------
            (
                predict_df,
                missing_features,
                extra_features,
            ) = prepare_model_input(
                predictor,
                input_df
            )

            with st.expander(
                "View Input Features",
                expanded=False
            ):
                st.dataframe(
                    predict_df.T.rename(
                        columns={0: "Input Value"}
                    ),
                    use_container_width=True
                )

                if missing_features:
                    st.warning(
                        f"{len(missing_features)} expected model "
                        "features were not generated and were filled "
                        "with 0. Verify feature consistency."
                    )
                    st.code("\n".join(missing_features))

                if extra_features:
                    st.caption(
                        f"{len(extra_features)} extra input features "
                        "were not used by this Predictor."
                    )

            # -----------------------------------------------
            # Step 5. Check specified model names
            # -----------------------------------------------
            available_models = predictor.model_names()

            missing_models = [
                name
                for name in selected_models
                if name not in available_models
            ]

            if missing_models:
                st.warning(
                    "Some selected model names were not found "
                    "in the loaded Predictor. Check the saved "
                    "model directory and its metadata."
                )
                st.code("\n".join(missing_models))

            # -----------------------------------------------
            # Step 6. Predict with each specified model
            # -----------------------------------------------
            st.subheader(
                "Six-Model Prediction Results"
            )

            # The first model is the designated champion.
            champion_model = selected_models[0]

            st.caption(
                f"🏆 Champion model: {champion_model}"
            )

            results = []

            for rank, model_name in enumerate(
                selected_models,
                start=1
            ):
                # Mark champion with an icon in the display name.
                display_model_name = (
                    f"🏆 {model_name}"
                    if model_name == champion_model
                    else model_name
                )

                try:
                    raw_pred = predictor.predict(
                        predict_df,
                        model=model_name
                    )

                    raw_value = float(raw_pred.iloc[0])

                    if not np.isfinite(raw_value):
                        raise ValueError(
                            "The model returned a non-finite prediction."
                        )

                    if target_name == "y1":

                        constrained_pred = apply_y1_constraints(
                            [raw_value],
                            [x_input]
                        )

                        final_value = float(
                            constrained_pred[0]
                        )

                        results.append({
                            "Rank": rank,
                            "Model": display_model_name,
                            "Pred y1": final_value,
                            "Status": "Success",
                        })

                    else:
                        # Do not apply Y1 constraints to pressure.
                        results.append({
                            "Rank": rank,
                            "Model": display_model_name,
                            "Predicted Pressure (kPa)": raw_value,
                            "Status": "Success",
                        })

                except Exception as model_error:

                    if target_name == "y1":
                        results.append({
                            "Rank": rank,
                            "Model": display_model_name,
                            "Pred y1": np.nan,
                            "Status": (
                                f"Failed: {str(model_error)}"
                            ),
                        })

                    else:
                        results.append({
                            "Rank": rank,
                            "Model": display_model_name,
                            "Predicted Pressure (kPa)": np.nan,
                            "Status": (
                                f"Failed: {str(model_error)}"
                            ),
                        })

            # -----------------------------------------------
            # Step 7. Display results in fixed ranking order
            # -----------------------------------------------
            results_df = pd.DataFrame(results)

            # Explicitly sort by the preset rank.
            results_df = (
                results_df
                .sort_values("Rank", ascending=True)
                .reset_index(drop=True)
            )

            display_results_table(
                results_df,
                target_name
            )

            success_count = int(
                (results_df["Status"] == "Success").sum()
            )

            st.write(
                f"Successful predictions: "
                f"{success_count}/{len(selected_models)}"
            )

            if success_count == 0:
                st.error(
                    "All six model predictions failed. "
                    "Please inspect the error messages in the table."
                )

            elif success_count < len(selected_models):
                st.warning(
                    "Some models failed. Only successful model "
                    "predictions should be used."
                )

            # -----------------------------------------------
            # Step 8. Download results as CSV
            # -----------------------------------------------
            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            file_name = (
                "Y1_top6_predictions.csv"
                if target_name == "y1"
                else "Pressure_top6_predictions.csv"
            )

            st.download_button(
                label="Download Predictions (CSV)",
                data=csv_data,
                file_name=file_name,
                mime="text/csv"
            )

            # -----------------------------------------------
            # Step 9. Display system parameters
            # -----------------------------------------------
            with st.expander(
                "View System Parameters",
                expanded=False
            ):
                st.write({
                    "Component A SMILES": smiles_A,
                    "Component B SMILES": smiles_B,
                    "Liquid mole fraction x1": x_input,
                    "Temperature (K)": temp_input,
                    "Psat1 (kPa)": psat1_input,
                    "Psat2 (kPa)": psat2_input,
                    "Prediction target": target_name,
                    "Champion model": champion_model,
                })

        except Exception as e:
            st.error(
                f"An error occurred during prediction: {str(e)}"
            )

            with st.expander("Technical details"):
                st.code(traceback.format_exc())
