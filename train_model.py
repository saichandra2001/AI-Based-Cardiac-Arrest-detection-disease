import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset (change encoding if needed)
heart_data = pd.read_csv(r'C:\Users\GROVEER\Heart_Disease_App\heart_disease_data.csv', encoding='latin1')

# Features and target
X = heart_data.drop('target', axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Save model
with open(r'C:\Users\GROVEER\Heart_Disease_App\model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved successfully!")
