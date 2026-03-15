import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Load your cleaned dataset
df = pd.read_csv("cleaned_travel_dataset_no_unnamed.csv")

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Define encoders
encoders = {
    "le_nationality": LabelEncoder(),
    "le_gender": LabelEncoder(),
    "le_crowd": LabelEncoder(),
    "le_age_group": LabelEncoder(),
    "le_age_destination": LabelEncoder()
}

# Fit encoders
encoders["le_nationality"].fit(df["Traveler nationality"].dropna())
encoders["le_gender"].fit(df["gender"].dropna())
encoders["le_crowd"].fit(df["Crowd Level"].dropna())
encoders["le_age_group"].fit(df["Age Group"].dropna())
encoders["le_age_destination"].fit(df["Destination"].dropna())

# Save each encoder
for name, encoder in encoders.items():
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(encoder, f)

print("✅ All LabelEncoders have been fitted and saved successfully!")
