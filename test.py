import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample dataset (synthetic for demo)
data = {
    'weather': [1, 0, 2, 1, 2, 0],     # 0: Clear, 1: Rain, 2: Fog
    'traffic_volume': [100, 300, 500, 150, 450, 200],  # Cars per hour
    'hour': [8, 18, 23, 14, 7, 9],     # Time of day
    'accident': [0, 1, 1, 0, 1, 0]     # 0: No accident, 1: Accident
}

# Load into DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['weather', 'traffic_volume', 'hour']]
y = df['accident']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
example = pd.DataFrame([[1, 400, 17]], columns=['weather', 'traffic_volume', 'hour'])
prediction = model.predict(example)
print(f"Predicted accident risk: {'Yes' if prediction[0] == 1 else 'No'}")
