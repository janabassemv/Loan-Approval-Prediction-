import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from collections import Counter

# Load dataset
path = "loan_approval_dataset.csv"   
df = pd.read_csv(path)

# Clean column names (remove accidental spaces in headers)
df.columns = df.columns.str.strip()

# Clean target column values (remove spaces like " Approved")
df["loan_status"] = df["loan_status"].str.strip()

# Encode target: Approved = 1, Rejected = 0
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

print("Unique target values after cleaning:", df["loan_status"].unique())

# Separate features (X) and target (y)
X = df.drop(columns=["loan_status", "loan_id"])
y = df["loan_status"]

# Encode categorical variables
categorical_cols = X.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Handle missing values
X = X.fillna(X.median(numeric_only=True))  # numeric columns
for col in categorical_cols:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mode()[0])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalanced data using scale_pos_weight
counts = Counter(y_train)
scale = counts[0] / counts[1] if counts[1] != 0 else 1

# Train XGBoost classifier
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
