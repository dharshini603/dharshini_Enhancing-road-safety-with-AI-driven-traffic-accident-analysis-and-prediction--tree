import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_data():
    # Simulated traffic data
    data = {
        'weather': [0, 1, 2, 0, 1, 2, 0, 1],  # 0: Clear, 1: Rain, 2: Fog
        'traffic_volume': [200, 450, 500, 180, 400, 470, 220, 430],
        'hour': [8, 17, 23, 13, 7, 20, 10, 18],
        'accident': [0, 1, 1, 0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

def predict_risk(model, weather, traffic_volume, hour):
    sample = pd.DataFrame([[weather, traffic_volume, hour]],
                          columns=['weather', 'traffic_volume', 'hour'])
    prediction = model.predict(sample)[0]
    print(f"Accident risk prediction: {'High (Accident Likely)' if prediction == 1 else 'Low (No Accident)'}")

def main():
    df = load_data()
    X = df[['weather', 'traffic_volume', 'hour']]
    y = df['accident']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Example prediction
    print("\nTesting with sample input (weather=1, volume=420, hour=17):")
    predict_risk(model, weather=1, traffic_volume=420, hour=17)

if _name_ == "_main_":
    main()
