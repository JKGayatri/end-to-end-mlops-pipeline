# app.py

from flask import Flask, request, jsonify
from src.predict import predict
from src.logger import get_logger

app = Flask(__name__)
logger = get_logger()


@app.route("/")
def home():
    return "House Price Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        data = request.get_json()
        
        logger.info(f"Input Data: {data}")

        prediction = predict(data)

        if prediction < 10000 or prediction > 1000000:
            print("⚠️ Warning: Prediction looks suspicious")

        response = {
            "predicted_price": prediction
        }

        logger.info(f"Prediction: {response}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)