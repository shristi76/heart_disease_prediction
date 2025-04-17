from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the model
model = pickle.load(open('heart_model.pkl', 'rb'))

# Create app
app = Flask(__name__)

# Home page with form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input from form and convert to float
            data = [float(x) for x in request.form.values()]
            
            # Define column names (must match training data)
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            input_df = pd.DataFrame([data], columns=columns)

            prediction = model.predict(input_df)

            result = "Person has heart disease" if prediction[0] == 1 else "Healthy heart"
            return render_template('index.html', prediction_text=result)
        except Exception as e:
            return f"Error: {e}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
