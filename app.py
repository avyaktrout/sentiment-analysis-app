from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('result.html', sentiment=sentiment, text=user_input)

if __name__ == '__main__':
    app.run(debug=True)