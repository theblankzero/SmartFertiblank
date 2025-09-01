import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Load the dataset
df = pd.read_csv('f2.csv')

# Prepare the data
X = df[['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = df['Fertilizer']

# Encode categorical variables
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Create mappings for soil and crop types
soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 
              'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat', 'coffee', 
              'kidneybeans', 'orange', 'pomegranate', 'rice', 'watermelon']

# Fit encoders
soil_encoder.fit(soil_types)
crop_encoder.fit(crop_types)

# Transform categorical columns
X_encoded = X.copy()
X_encoded['Soil_Type'] = soil_encoder.transform(X['Soil_Type'])
X_encoded['Crop_Type'] = crop_encoder.transform(X['Crop_Type'])

# Encode target variable
y_encoded = fertilizer_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(fertilizer_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training ANN model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=fertilizer_encoder.classes_))

# Save the model and encoders
model.save('ann_fertilizer_model.h5')

# Save encoders and scaler
with open('ann_encoders.pkl', 'wb') as f:
    pickle.dump({
        'soil_encoder': soil_encoder,
        'crop_encoder': crop_encoder,
        'fertilizer_encoder': fertilizer_encoder,
        'scaler': scaler
    }, f)

print("ANN model and encoders saved successfully!")
print("Files created:")
print("- ann_fertilizer_model.h5")
print("- ann_encoders.pkl")