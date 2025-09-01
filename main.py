from flask import Flask, request, render_template
import pickle
import os
from ann_predictor import ANNFertilizerPredictor

app = Flask(__name__)

#importing pickle files
model = pickle.load(open('classifier.pkl','rb'))
ferti = pickle.load(open('fertilizer.pkl','rb'))

# Initialize ANN predictor
ann_predictor = ANNFertilizerPredictor()
@app.route('/')
def home():
    return render_template('plantindex.html')

@ app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@ app.route('/ANN')
def ANN():
    return render_template('ANN.html')
@ app.route('/Detail')
def Detail():
    return render_template('Detail.html')



@app.route('/predict',methods=['POST'])
def predict():
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

# Convert values to integers
    input = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
    res = ferti.classes_[model.predict([input])]
    return render_template('Model1.html', x=res)
if __name__ == "__main__":
@app.route('/predict_ann', methods=['POST'])
def predict_ann():
    try:
        temp = request.form.get('temp')
        humi = request.form.get('humid')
        mois = request.form.get('mois')
        soil = request.form.get('soil')
        crop = request.form.get('crop')
        nitro = request.form.get('nitro')
        pota = request.form.get('pota')
        phosp = request.form.get('phos')
        
        # Validate inputs
        if None in (temp, humi, mois, soil, crop, nitro, pota, phosp):
            return render_template('ANN.html', 
                                 error='All fields are required. Please fill in all values.')
        
        # Convert to appropriate types
        try:
            temp = float(temp)
            humi = float(humi)
            mois = float(mois)
            soil = int(soil)
            crop = int(crop)
            nitro = float(nitro)
            pota = float(pota)
            phosp = float(phosp)
        except ValueError:
            return render_template('ANN.html', 
                                 error='Invalid input. Please provide numeric values.')
        
        # Make prediction using ANN
        result = ann_predictor.predict(temp, humi, mois, soil, crop, nitro, pota, phosp)
        
        if isinstance(result, dict):
            return render_template('ANN.html', 
                                 prediction=result['fertilizer'],
                                 confidence=result['confidence'])
        else:
            return render_template('ANN.html', error=str(result))
            
    except Exception as e:
        return render_template('ANN.html', error=f'An error occurred: {str(e)}')

    app.run(debug=True)