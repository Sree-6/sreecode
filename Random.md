import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load or simulate transaction dataset
data = {
    'amount': [150.75, 9500.00, 35.20, 420.50, 5999.99, 12.00, 7800.00],
    'user_id': [101, 101, 102, 103, 103, 104, 104],
    'hour': [10, 10, 14, 18, 1, 9, 23],
    'transaction_type': ['online', 'in_store', 'in_store', 'in_store', 'online', 'in_store', 'online'],
    'is_fraud': [0, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Step 2: Encode categorical features
df = pd.get_dummies(df, columns=['transaction_type'], drop_first=True)

# Step 3: Prepare features and labels
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Step 4: Scale and balance dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Step 6: Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save model and scaler
joblib.dump(model, 'rf_fraud_model.pkl')
joblib.dump(scaler, 'fraud_scaler.pkl')

# Step 8: Real-time transaction guard function
def guard_transaction(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    for col in ['transaction_type_online']:
        if col not in input_df.columns:
            input_df[col] = 0
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return "Fraudulent" if prediction == 1 else "Legitimate"

# Simulate real-time prediction
sample_txn = {'amount': 3000, 'user_id': 105, 'hour': 3, 'transaction_type': 'online'}
result = guard_transaction(sample_txn)
print(f"Real-time Transaction Check: {result}")
