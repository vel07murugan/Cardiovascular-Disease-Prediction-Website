import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
preprocessor = joblib.load("preprocessor.pkl")
X = joblib.load("feature_names.pkl")
numeric_features = joblib.load("numeric_features.pkl")

data = pd.read_csv("cvd_prediction.csv").drop(columns=['region','income_level','air_pollution_exposure','participated_in_free_screening','medication_usage'])
y = data["heart_attack"]
X_data = data.drop("heart_attack", axis=1)
X_transformed = preprocessor.transform(X_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# ------------------- RandomForest -------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "rf_model.pkl")

# ------------------- LightGBM -------------------
lgbm = lgb.LGBMClassifier(n_estimators=500, device='gpu')
lgbm.fit(X_train, y_train)
joblib.dump(lgbm, "lgbm_model.pkl")

# ------------------- XGBoost -------------------
xgb_model = xgb.XGBClassifier(n_estimators=500, tree_method='gpu_hist', predictor='gpu_predictor')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgb_model.pkl")

# ------------------- TabNet -------------------
tabnet = TabNetClassifier(device_name='cuda')
tabnet.fit(X_train, y_train.values, max_epochs=50, patience=5, batch_size=1024, virtual_batch_size=128)
joblib.dump(tabnet, "tabnet_model.pkl")

# ------------------- MLP -------------------
mlp = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train, y_train, epochs=20, batch_size=1024, verbose=1)
mlp.save("mlp_model.h5")

print("✅ All models trained and saved.")
