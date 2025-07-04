import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(columns=["Loan_ID"], inplace=True)

    # Fill missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'LoanAmount', 'Loan_Amount_Term']:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Label encoding for categorical columns
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Feature engineering
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = (df["LoanAmount"] * 1000) / df["Loan_Amount_Term"]
    df["EMI_Ratio"] = df["EMI"] / df["Total_Income"]
    df["Affordable"] = df["EMI_Ratio"].apply(lambda x: 1 if x <= 0.5 else 0)

    return df, encoders
