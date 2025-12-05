import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("cvd_prediction.csv")

# Drop irrelevant features
drop_features = ['region', 'income_level', 'air_pollution_exposure', 'participated_in_free_screening', 'medication_usage']
data = data.drop(columns=drop_features)

# Separate features and target
X = data.drop("heart_attack", axis=1)
y = data["heart_attack"]

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit and transform
X_transformed = preprocessor.fit_transform(X)

# Save preprocessing objects
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(feature_names := X.columns.tolist(), "feature_names.pkl")
joblib.dump(numeric_features, "numeric_features.pkl")

print("✅ Preprocessing completed and objects saved.")
