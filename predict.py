import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# ------------------- Load preprocessor & models -------------------
preprocessor = joblib.load("preprocessor.pkl")

rf = joblib.load("rf_model.pkl")
lgbm = joblib.load("lgbm_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
tabnet = joblib.load("tabnet_model.pkl")
mlp = load_model("mlp_model.h5")

feature_names = joblib.load("feature_names.pkl")
numeric_features = joblib.load("numeric_features.pkl")

# ------------------- Prediction function -------------------
def predict_all_models(user_input_dict):
    user_df = pd.DataFrame([user_input_dict])
    user_transformed = preprocessor.transform(user_df)

    # Model predictions
    rf_prob = rf.predict_proba(user_transformed)[:,1]
    lgbm_prob = lgbm.predict_proba(user_transformed)[:,1]
    xgb_prob = xgb_model.predict_proba(user_transformed)[:,1]
    tabnet_prob = tabnet.predict_proba(user_transformed)[:,1]
    mlp_prob = mlp.predict(user_transformed).flatten()

    model_probs = {
        "RandomForest": rf_prob,
        "LightGBM": lgbm_prob,
        "XGBoost": xgb_prob,
        "TabNet": tabnet_prob,
        "MLP": mlp_prob
    }

    print("\n----- Individual Model Predictions -----")
    for name, prob in model_probs.items():
        pred = int(prob >= 0.5)
        print(f"{name}: Prediction={pred}, Probability={prob[0]:.2f}")

    # Ensemble prediction (average probability)
    avg_prob = np.mean(list(model_probs.values()), axis=0)
    ensemble_pred = int(avg_prob >= 0.5)
    print(f"\nEnsemble Prediction: {ensemble_pred}, Probability={avg_prob[0]:.2f}")

# ------------------- Main -------------------
if __name__ == "__main__":
    print("Enter patient details for CVD prediction:")
    user_input = {}
    for col in feature_names:
        val = input(f"{col}: ")
        user_input[col] = float(val) if col in numeric_features else val if val.strip() != "" else "unknown"

    predict_all_models(user_input)
