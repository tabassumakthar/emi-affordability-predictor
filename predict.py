import pandas as pd
import joblib

def get_user_input_and_predict():
    print("Enter your details:")
    gender = input("Gender (Male/Female): ").capitalize()
    married = input("Married (Yes/No): ").capitalize()
    dependents = input("Dependents (0/1/2/3+): ")
    education = input("Education (Graduate/Not Graduate): ")
    education = "Graduate" if "grad" in education.lower() else "Not Graduate"
    self_employed = input("Self Employed (Yes/No): ").capitalize()
    applicant_income = float(input("Applicant Monthly Income: "))
    coapplicant_income = float(input("Coapplicant Monthly Income: "))
    loan_amount = float(input("Desired Loan Amount (in 1000s): "))
    loan_term = float(input("Loan Term (in months): "))
    credit_history = float(input("Credit History (0.0 or 1.0): "))
    property_area = input("Property Area (Urban/Semiurban/Rural): ").capitalize()
    rent = float(input("Monthly Rent: "))
    other_expenses = float(input("Other Monthly Expenses: "))

    model = joblib.load("emi_model.pkl")
    feature_order = joblib.load("feature_order.pkl")

    cols_to_encode = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in cols_to_encode}

    def safe_encode(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return -1

    data = {
        "Gender": safe_encode(encoders["Gender"], gender),
        "Married": safe_encode(encoders["Married"], married),
        "Dependents": safe_encode(encoders["Dependents"], dependents),
        "Education": safe_encode(encoders["Education"], education),
        "Self_Employed": safe_encode(encoders["Self_Employed"], self_employed),
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": safe_encode(encoders["Property_Area"], property_area)
    }

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=feature_order, fill_value=0)

    prediction = model.predict(input_df)[0]

    total_income = applicant_income + coapplicant_income
    emi = (loan_amount * 1000) / loan_term
    total_expenses = rent + other_expenses
    available = total_income - total_expenses
    safe_emi = available * 0.4

    print(f"\nEMI for this loan: ₹{emi:.2f}")
    print(f"Safe EMI range based on your income/expenses: ₹{safe_emi:.2f}")
    if prediction == 1:
        print("✅ You CAN afford this EMI.")
    else:
        print("❌ You CANNOT afford this EMI comfortably.")

if __name__ == "__main__":
    get_user_input_and_predict()
