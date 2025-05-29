import pandas as pd
import dill
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv("matedata.csv")

# Handle missing values
df["Interests"] = df["Interests"].fillna("")

# Fit the OneHotEncoder on categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[["Occupation", "Sleep Schedule", "Social Activity Level"]])

# Fit TfidfVectorizer on Interests
vectorizer = TfidfVectorizer()
vectorizer.fit(df["Interests"])

# Fit MinMaxScaler on Budget Range
budget_scaler = MinMaxScaler()
budget_scaler.fit(df[["Budget Range"]])

# Store all components in a dictionary
model_data = {
    "encoder": encoder,
    "vectorizer": vectorizer,
    "budget_scaler": budget_scaler
}

# Save the model using dill
with open("roommatenew.pkl", "wb") as f:
    dill.dump(model_data, f)

print("✅ roommatenew.pkl created successfully!")
