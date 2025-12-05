import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model

# ------------------- Load preprocessor and models -------------------
preprocessor = joblib.load("preprocessor.pkl")

rf = joblib.load("rf_model.pkl")
lgbm = joblib.load("lgbm_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
tabnet = joblib.load("tabnet_model.pkl")
mlp = load_model("mlp_model.h5")

# ------------------- Load dataset -------------------
data = pd.read_csv("cvd_prediction.csv").drop(columns=[
    'region', 'income_level', 'air_pollution_exposure', 'participated_in_free_screening', 'medication_usage'
])
y = data["heart_attack"]
X_data = data.drop("heart_attack", axis=1)
X_transformed = preprocessor.transform(X_data)

# ------------------- Predictions -------------------
rf_probs = rf.predict_proba(X_transformed)[:,1]
lgbm_probs = lgbm.predict_proba(X_transformed)[:,1]
xgb_probs = xgb_model.predict_proba(X_transformed)[:,1]
tabnet_probs = tabnet.predict_proba(X_transformed)[:,1]
mlp_probs = mlp.predict(X_transformed).flatten()

model_probs_dict = {
    "RandomForest": rf_probs,
    "LightGBM": lgbm_probs,
    "XGBoost": xgb_probs,
    "TabNet": tabnet_probs,
    "MLP": mlp_probs
}

# ------------------- Evaluation -------------------
print("------- Individual Model Performance -------")
for name, probs in model_probs_dict.items():
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    roc = roc_auc_score(y, probs)
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y, preds))

# ------------------- Ensemble Performance -------------------
avg_probs = np.mean(list(model_probs_dict.values()), axis=0)
ensemble_preds = (avg_probs >= 0.5).astype(int)
print("\n------- Ensemble Model Performance -------")
print(f"Accuracy: {accuracy_score(y, ensemble_preds):.4f}")
print(f"Precision: {precision_score(y, ensemble_preds):.4f}")
print(f"Recall: {recall_score(y, ensemble_preds):.4f}")
print(f"F1-score: {f1_score(y, ensemble_preds):.4f}")
print(f"ROC-AUC: {roc_auc_score(y, avg_probs):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y, ensemble_preds))
