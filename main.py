# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic dataset
np.random.seed(42)
data_size = 1000

# Simulated features
rainfall = np.random.uniform(50, 500, data_size)           # mm
river_level = np.random.uniform(2, 10, data_size)          # meters
soil_moisture = np.random.uniform(10, 50, data_size)       # %

# Simulated flood occurrence: 1 = flood, 0 = no flood
flood_risk = (rainfall > 300) & (river_level > 6) & (soil_moisture > 30)
flood = flood_risk.astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Rainfall': rainfall,
    'RiverLevel': river_level,
    'SoilMoisture': soil_moisture,
    'Flood': flood
})

# Prepare data
X = df[['Rainfall', 'RiverLevel', 'SoilMoisture']]
y = df['Flood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predicting a sample case
sample = pd.DataFrame({'Rainfall': [350], 'RiverLevel': [7.5], 'SoilMoisture': [40]})
prediction = model.predict(sample)
print("Flood Predicted:" if prediction[0] == 1 else "No Flood Predicted")