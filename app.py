
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
