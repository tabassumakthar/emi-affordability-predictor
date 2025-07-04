from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from data_preprocessing import preprocess_data

def train_and_save_model(data_path):
    df, encoders = preprocess_data(data_path)

    X = df.drop(columns=["Affordable", "EMI", "EMI_Ratio"])
    y = df["Affordable"]

    # Save feature order
    joblib.dump(X.columns.tolist(), "feature_order.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and encoders
    joblib.dump(model, "emi_model.pkl")
    for col, enc in encoders.items():
        joblib.dump(enc, f"{col}_encoder.pkl")
    print("Model and encoders saved.")

if __name__ == "__main__":
    train_and_save_model("train_u6lujuX_CVtuZ9i.csv")  # your dataset path here
