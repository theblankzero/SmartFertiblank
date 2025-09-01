import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

class ANNFertilizerPredictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.load_model()
    
    def load_model(self):
        """Load the trained ANN model and encoders"""
        try:
            # Load the Keras model
            self.model = keras.models.load_model('ann_fertilizer_model.h5')
            
            # Load encoders and scaler
            with open('ann_encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            
            print("ANN model loaded successfully!")
        except Exception as e:
            print(f"Error loading ANN model: {e}")
            self.model = None
            self.encoders = None
    
    def predict(self, temperature, humidity, moisture, soil_type, crop_type, 
                nitrogen, potassium, phosphorous):
        """
        Predict fertilizer recommendation using ANN
        
        Parameters:
        - temperature: Temperature in Celsius
        - humidity: Humidity percentage
        - moisture: Soil moisture percentage
        - soil_type: Soil type index (0-4)
        - crop_type: Crop type index (0-16)
        - nitrogen: Nitrogen content
        - potassium: Potassium content
        - phosphorous: Phosphorous content
        
        Returns:
        - Recommended fertilizer name
        """
        if self.model is None or self.encoders is None:
            return "Model not loaded. Please train the model first."
        
        try:
            # Prepare input data
            input_data = np.array([[
                temperature, humidity, moisture, soil_type, crop_type,
                nitrogen, potassium, phosphorous
            ]])
            
            # Scale the input data
            input_scaled = self.encoders['scaler'].transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            # Get fertilizer name
            fertilizer_name = self.encoders['fertilizer_encoder'].inverse_transform([predicted_class])[0]
            
            # Get prediction confidence
            confidence = np.max(prediction) * 100
            
            return {
                'fertilizer': fertilizer_name,
                'confidence': f"{confidence:.2f}%"
            }
            
        except Exception as e:
            return f"Error making prediction: {e}"
    
    def get_soil_types(self):
        """Return available soil types"""
        return ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
    
    def get_crop_types(self):
        """Return available crop types"""
        return ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 
                'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat', 'coffee', 
                'kidneybeans', 'orange', 'pomegranate', 'rice', 'watermelon']