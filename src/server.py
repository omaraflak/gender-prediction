"""
Name2Gender API
"""
import os
import json
import numpy as np

import flask
from flask import Flask, request, jsonify

from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

def get_output(i, n):
    tmp = np.zeros(n)
    tmp[i] = 1
    return tmp

def read_model(network_path):
    if not os.path.exists(network_path):
        raise ValueError('Path not found : {}'.format(network_path))
    dat = json.loads(open(os.path.join(network_path, 'data.json')).read())
    mod = load_model(os.path.join(network_path, 'rnn.model'))
    return mod, dat

@app.route('/')
def hello_world():
    return "Welcome to GenderAPI. Please request on /predict using 'Content-Type: application/json' header and a json array of names in the body."

@app.route('/predict', methods=['POST'])
def predict():
    names = request.get_json()
    with graph.as_default():
        # format input
        names = [s.lower() for s in names]
        names = [list(i)+['END']*(max_len-len(i)) for i in names]
        names = [[get_output(char_index[j], vocab_len) for j in i] for i in names]
        names = np.asarray(names)

        # predict gender
        out = model.predict(names)
        labels = [('M' if p[0] > p[1] else 'F') for p in out]

        # return result
        return flask.jsonify(labels)

# Force CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Keras
graph = tf.get_default_graph()
model, data = read_model('model')
max_len = data['max_len']
vocab_len = data['vocab_len']
char_index = data['char_index']

# Run Flask
app.run(host='0.0.0.0', port=4000, debug=False)
