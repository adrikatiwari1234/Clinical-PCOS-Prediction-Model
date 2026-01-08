from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained Random Forest model
model = pickle.load(open('pcos_random_forest.sav', 'rb'))

# Load model feature names
model_columns = pickle.load(open('model_columns.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form in correct order
        input_features = []
        for col in model_columns:
            value = float(request.form[col])
            input_features.append(value)

        # Convert to numpy array
        final_input = np.array([input_features])

        # Prediction
        prediction = model.predict(final_input)[0]

        if prediction == 1:
            result = "PCOS Detected"
        else:
            result = "No PCOS Detected"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text="Error: Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
