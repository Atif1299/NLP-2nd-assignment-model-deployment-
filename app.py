"""
Flask web application for Car Evaluation Prediction
Uses the trained NLP-style pipeline for predictions
"""
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
with open('car_nlp_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Class labels mapping
CLASS_LABELS = {0: 'Acceptable', 1: 'Good', 2: 'Unacceptable', 3: 'Very Good'}

# Options for each feature
OPTIONS = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

@app.route('/')
def home():
    return render_template('index.html', options=OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'buying': [request.form['buying']],
            'maint': [request.form['maint']],
            'doors': [request.form['doors']],
            'persons': [request.form['persons']],
            'lug_boot': [request.form['lug_boot']],
            'safety': [request.form['safety']]
        }
        
        # Create DataFrame
        input_df = pd.DataFrame(data)
        
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        result = CLASS_LABELS[prediction]
        
        return render_template('index.html', 
                             options=OPTIONS, 
                             prediction=result,
                             form_data=request.form)
    except Exception as e:
        return render_template('index.html', 
                             options=OPTIONS, 
                             error=str(e))

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

