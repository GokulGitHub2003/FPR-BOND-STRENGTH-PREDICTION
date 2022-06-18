import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
final_model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = final_model.predict(final_features)
    return render_template('index1.html', prediction_text='THE PREDICTED VALUE IS {}'.format(float(prediction)))

if __name__ == "__main__":
    app.run(debug=True)