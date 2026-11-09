
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("chainsight_supply_chain.csv")

# 2. Create inventory risk target
df["inventory_risk_target"] = (
    df["actual_delivery_days"] > df["inventory_days"]
).astype(int)

# 3. Select input features
features = [
    "supplier",
    "product",
    "promised_delivery_days",
    "inventory_days",
    "demand_units",
    "quantity",
    "unit_cost",
    "disruption_type"
]

X = df[features]
y = df["inventory_risk_target"]

# 4. Identify categorical and numerical columns
categorical_features = [
    "supplier",
    "product",
    "disruption_type"
]

numerical_features = [
    "promised_delivery_days",
    "inventory_days",
    "demand_units",
    "quantity",
    "unit_cost"
]

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# 6. Build Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train model
model.fit(X_train, y_train)

# 9. Evaluate model
predictions = model.predict(X_test)

print("MODEL EVALUATION")
print("----------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 10. Save model
joblib.dump(model, "chainsight_risk_model.pkl")

print("\nModel saved successfully!")
print("File: chainsight_risk_model.pkl")
