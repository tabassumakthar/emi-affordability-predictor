# 🧮 EMI Affordability Predictor

This project predicts whether a person can comfortably afford a loan's EMI (Equated Monthly Installment) based on their income, expenses, and loan details using a trained Machine Learning model.

---

## 📌 Features

- Takes user input: income, co-applicant income, rent, loan amount, etc.
- Calculates EMI and EMI-to-Income ratio
- Predicts affordability using a Random Forest model
- Recommends a **safe EMI range** based on the user's available income

---

## 📁 Project Structure

EMI-Affordability-Predictor/
├── emi_model.pkl # Trained Random Forest model
├── feature_order.pkl # Order of features used for prediction
├── Gender_encoder.pkl # LabelEncoder for categorical columns
├── Married_encoder.pkl
├── Dependents_encoder.pkl
├── Education_encoder.pkl
├── Self_Employed_encoder.pkl
├── Property_Area_encoder.pkl
├── EMI_Predictor.ipynb # Jupyter Notebook (optional)
├── README.md # You're reading this!


---

## ⚙️ How to Use

1. Clone this repository or download the files.
2. Load the model and encoders using Python and `joblib`.
3. Pass user input (after encoding) into the model.
4. Get EMI prediction and personalized affordability suggestion.

---

## 📊 Model Info

- **Algorithm:** Random Forest Classifier  
- **Target:** Binary label (`1` = Can afford, `0` = Cannot afford)  
- **Dataset:** [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

---

## 💡 Sample Output

Your EMI: ₹7500.00
Safe EMI Range: ₹8800.00
❌ You CANNOT afford this EMI comfortably.
