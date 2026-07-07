import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Features (input)
X = data.drop("label", axis=1)

# Target (output)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")