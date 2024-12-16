from flask import Flask, render_template, request
import pandas as pd
import joblib
import random 
app = Flask(__name__)

# Load your model
model = joblib.load('models/rbf_lp_model.pkl')  # Ensure your model file is in the same directory

@app.route('/')
def index():
    return render_template('index.html', random=random)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'csvFile' in request.files:  # CSV Upload
            file = request.files['csvFile']
            data = pd.read_csv(file)
            predictions = model.predict(data.values).tolist()
            result = [{"features": row,  "prediction": f'<span style="color: {"red" if pred == 1 else "green"}">'
                          f'{"Asthma likely" if pred == 1 else "Asthma unlikely"}</span>'}
                      for row, pred in zip(data.values, predictions)]
            return render_template('index.html', results=result, random=random)

    except Exception as e:
        return render_template('index.html', error_text=f"An error occurred: {e}", random=random)

if __name__ == '__main__':
    app.run(debug=True)
