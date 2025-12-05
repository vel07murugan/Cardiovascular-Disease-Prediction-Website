import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from tensorflow.keras.models import load_model

# ===============================
# Load Preprocessor & Models
# ===============================
preprocessor = joblib.load("preprocessor.pkl")
rf = joblib.load("rf_model.pkl")
lgbm = joblib.load("lgbm_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
tabnet = joblib.load("tabnet_model.pkl")
mlp = load_model("mlp_model.h5")

feature_names = joblib.load("feature_names.pkl")
numeric_features = joblib.load("numeric_features.pkl")

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="CVD Risk Prediction", layout="wide")

st.title("🫀 Cardiovascular Disease (CVD) Risk Prediction")
st.write("Enter patient details below to predict CVD risk using multiple AI models.")

# Dropdown options for categorical & binary features
smoking_options = ["Never", "Past", "Current"]
alcohol_options = ["None", "Moderate", "High"]
physical_activity_options = ["Low", "Moderate", "High"]
dietary_options = ["Healthy", "Unhealthy"]
stress_options = ["Low", "Moderate", "High"]
ekg_options = ["Normal", "Abnormal"]
gender_options = ["Male", "Female"]
binary_options = [0, 1]  # for previous_heart_disease, hypertension, diabetes, obesity, family_history

user_input = {}
cols = st.columns(2)

for i, col in enumerate(feature_names):
    if col in numeric_features:
        user_input[col] = cols[i % 2].number_input(f"{col}", value=0.0)
    else:
        col_lower = col.lower()
        # Special dropdowns for known categorical or binary variables
        if "smoking" in col_lower:
            user_input[col] = cols[i % 2].selectbox(f"{col}", smoking_options)
        elif "alcohol" in col_lower:
            user_input[col] = cols[i % 2].selectbox(f"{col}", alcohol_options)
        elif "physical" in col_lower:
            user_input[col] = cols[i % 2].selectbox(f"{col}", physical_activity_options)
        elif "dietary" in col_lower:
            user_input[col] = cols[i % 2].selectbox(f"{col}", dietary_options)
        elif "stress" in col_lower:
            user_input[col] = cols[i % 2].selectbox(f"{col}", stress_options)
        elif "ekg" in col_lower:
            user_input[col] = cols[i % 2].selectbox(f"{col}", ekg_options)
        elif "gender" in col_lower:
            user_input[col] = cols[i % 2].selectbox(f"{col}", gender_options)
        elif any(x in col_lower for x in ["previous_heart_disease", "hypertension", "diabetes", "obesity", "family_history"]):
            user_input[col] = cols[i % 2].selectbox(f"{col}", binary_options)
        else:
            user_input[col] = cols[i % 2].text_input(f"{col}", value="unknown")

# ===============================
# Prediction Logic
# ===============================
if st.button("Predict CVD Risk"):
    user_df = pd.DataFrame([user_input])
    user_transformed = preprocessor.transform(user_df)

    # Extract transformed feature names
    try:
        transformed_feature_names = preprocessor.get_feature_names_out()
    except:
        transformed_feature_names = [f"Feature_{i}" for i in range(user_transformed.shape[1])]

    # Individual Model Predictions
    rf_prob = rf.predict_proba(user_transformed)[:, 1]
    lgbm_prob = lgbm.predict_proba(user_transformed)[:, 1]
    xgb_prob = xgb_model.predict_proba(user_transformed)[:, 1]
    tabnet_prob = tabnet.predict_proba(user_transformed)[:, 1]
    mlp_prob = mlp.predict(user_transformed).flatten()

    results = {
        "RandomForest": rf_prob[0],
        "LightGBM": lgbm_prob[0],
        "XGBoost": xgb_prob[0],
        "TabNet": tabnet_prob[0],
        "MLP": mlp_prob[0],
    }

    st.subheader("Model Predictions")
    for model, prob in results.items():
        pred = "High Risk" if prob >= 0.5 else "Low Risk"
        st.write(f"**{model}:** {pred} (Probability: {prob:.2f})")

    # ===============================
    # Ensemble Result
    # ===============================
    ensemble_prob = np.mean(list(results.values()))
    ensemble_pred = "High Risk" if ensemble_prob >= 0.5 else "Low Risk"

    st.markdown("---")
    st.subheader("🧠 Ensemble Result")
    st.write(f"**{ensemble_pred}** (Probability: {ensemble_prob:.2f})")

    # ===============================
    # SHAP Explainability (for Ensemble)
    # ===============================
    st.markdown("---")
    st.subheader("🔍 Explainable AI - Feature Impact (Ensemble Model)")

    rf_explainer = shap.TreeExplainer(rf)
    lgbm_explainer = shap.TreeExplainer(lgbm)
    xgb_explainer = shap.TreeExplainer(xgb_model)

    rf_shap = rf_explainer.shap_values(user_transformed)
    lgbm_shap = lgbm_explainer.shap_values(user_transformed)
    xgb_shap = xgb_explainer.shap_values(user_transformed)

    # Handle multi-class SHAP outputs
    def extract_shap_class(shap_values):
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
            shap_values = shap_values[:, :, 1]
        return shap_values

    rf_shap = extract_shap_class(rf_shap)
    lgbm_shap = extract_shap_class(lgbm_shap)
    xgb_shap = extract_shap_class(xgb_shap)

    # Average SHAP values
    shap_values_ensemble = (rf_shap + lgbm_shap + xgb_shap) / 3
    shap_values_flat = shap_values_ensemble.flatten()[:len(transformed_feature_names)]

    shap_importance = pd.DataFrame({
        "Feature": transformed_feature_names,
        "SHAP Value": shap_values_flat
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.bar_chart(shap_importance.set_index("Feature")["SHAP Value"])
    st.info("Higher positive SHAP values indicate features contributing to higher CVD risk.")
   