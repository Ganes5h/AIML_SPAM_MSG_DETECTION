# app.py
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    transformed_message = vectorizer.transform([message])
    prediction = model.predict(transformed_message)
    return jsonify({'prediction': prediction[0]})


@app.route('/developers')
def developers():
    return render_template('developers.html')


if __name__ == '__main__':
    app.run(debug=True)
