
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_dir, 'diabetes-prediction-rfc-model.pkl')
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        prediction_val = int(my_prediction[0])
        
        inputs = {
            'pregnancies': preg,
            'glucose': glucose,
            'bloodpressure': bp,
            'skinthickness': st,
            'insulin': insulin,
            'bmi': bmi,
            'dpf': dpf,
            'age': age
        }
        
        return render_template('result.html', prediction=prediction_val, inputs=inputs)

if __name__ == '__main__':
	app.run(debug=True)