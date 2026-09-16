from flask import Flask, request, jsonify
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))
from model import load_model, predict_direction, FEATURE_COLS

app = Flask(__name__)
model = load_model(str(ROOT / "model" / "model.pkl"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        missing = [f for f in FEATURE_COLS if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        prediction = predict_direction(model, data)
        return jsonify({"prediction": prediction, "meaning": "1=up, 0=down"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/plot", methods=["GET"])
def plot():
    from flask import send_file
    return send_file(str(ROOT / "reports" / "images" / "confusion_matrix.png"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)