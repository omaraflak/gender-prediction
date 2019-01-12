from flask import Flask, request, jsonify
from gender_api import GenderAPI

# Init Flask and GenderAPI
app = Flask(__name__)
api = GenderAPI()

@app.route('/')
def hello_world():
    return "Welcome to GenderAPI. Please request on /predict using \
    'Content-Type: application/json' header and a json array of names in the body."

@app.route('/predict', methods=['POST'])
def predict():
    names = request.get_json()
    labels = api.predict(names)
    return jsonify(labels)

# Run Flask
app.run(host='0.0.0.0', port=4000, debug=False)
