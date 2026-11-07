    .stTextInput label,
    .stNumberInput label {
        font-size: 0.85rem !important;
        padding-bottom: 2px !important;
    }

    /* ---------- Input spacing ---------- */

    .stTextInput,
    .stNumberInput {
        margin-bottom: 0.2rem !important;
    }

    .stTextInput input,
    .stNumberInput input {
        font-size: 0.88rem !important;
    }

    /* ---------- Prediction button ---------- */

    .stButton > button {
        min-height: 2.7rem;
        margin-top: 0.45rem;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 8px;
    }

    /* ---------- Molecular structure labels ---------- */

    .mol-title {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 0 8px 0;
        color: #374151;
    }

    /* ---------- Center molecular images ---------- */

    div[data-testid="stImage"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto 8px auto !important;
    }

    div[data-testid="stImage"] img {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        object-fit: contain;
    }

    /* ---------- Results table ---------- */

    div[data-testid="stDataFrame"] {
        margin-top: 4px !important;
        margin-bottom: 0 !important;
    }

    /* ---------- Download button ---------- */

    .stDownloadButton {
        margin-top: 5px;
    }

    .stDownloadButton > button {
        min-height: 2.3rem;
        font-size: 0.85rem;
        border-radius: 7px;
    }

    /* ---------- Responsive layout ---------- */

    @media (max-width: 700px) {
        .block-container {
            padding: 0.8rem !important;
        }

        .main-title {
            font-size: 1.35rem;
        }

        .section-header {
            font-size: 0.85rem;
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
    <div class="main-title">
        ⚗️ Isothermal VLE Predictor
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# 8. Load AutoGluon predictors
# ===========================================================
@st.cache_resource(show_spinner=False)
def load_predictor_y1():

    if not AUTOGLUON_AVAILABLE:
        raise ImportError(AUTOGLUON_ERROR)

    if not os.path.isdir(MODEL_Y1_PATH):
        raise FileNotFoundError(
            f"y1 model directory not found: {MODEL_Y1_PATH}"
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
    """Calculate RDKit descriptors from SMILES."""

    if not RDKIT_AVAILABLE:
        raise RuntimeError(
