# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:17:19 2020

@author: Krishna
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
my_model = pickle.load(open('my_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('my_index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    my_features=[x for x in request.form.values()]
    my_final_features = [np.array(my_features)]
    prediction = my_model.predict(my_final_features)

    return render_template('my_index.html', prediction_text='Vehicle Prices should be $ {}'.format(prediction))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = my_model.predict([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    