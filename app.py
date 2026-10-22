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
