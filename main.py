from flask import Flask, jsonify
from flask import request
from src.classifier import Aspect_Classifier
import os

app = Flask(__name__)
pipeline = Aspect_Classifier(
    model_path=os.path.join(*[os.getcwd(), "weights", "tiny_bert.pt"]),
    token_path=os.path.join(*[os.getcwd(), "weights", "tiny_bert_tokenizer"])
)
sentiment_id = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


@app.route('/predict', methods=['POST'])
def predict():
    """
    Request
    -------
    >>> import requests
    >>> files = {
    ...    "text": "my laptop is working fine but the mouse is a little noisy."
    ...    "phrase": "mouse"
    ... }
    >>> requests.post("http://localhost:5000/predict", files=files).json()

    {'label': 2}

    """
    if request.method == 'POST':
        text = request.files["text"].read().decode('UTF-8')
        phrase = request.files["phrase"].read().decode('UTF-8')
        output = pipeline(str(text), str(phrase))
        sentiment = sentiment_id[output["label"]]
        probablities = round(output[sentiment], 2)
        return jsonify({
            "label": output["label"],
            "sentiment": sentiment,
            "probablities": probablities
        })


if __name__ == '__main__':

    app.run()
