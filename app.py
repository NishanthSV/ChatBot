from flask import Flask, render_template, request, jsonify

import sys
sys.path.append('./chatbot/')
from inference import inference

app = Flask(__name__)

app.static_folder = 'static'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['inputText']
    prediction = ml_inference(input_text)
    return jsonify({'response': prediction})

def ml_inference(input_text):
    return inference(input_text)

if __name__ == '__main__':
    app.run(debug=True)
