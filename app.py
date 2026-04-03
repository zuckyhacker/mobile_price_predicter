from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and files
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
le = joblib.load("brand_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        brand = request.form["brand"]
        ram = int(request.form["ram"])
        battery = int(request.form["battery"])
        memory = int(request.form["memory"])
        camera = int(request.form["camera"])
        weight = int(request.form["weight"])

        brand_encoded = le.transform([brand])[0]

        input_data = pd.DataFrame([[ram, battery, memory, camera, weight, brand_encoded]],
                                  columns=features)

        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction_text=f"Estimated Price: ₹{int(prediction)}")

    except Exception as e:
        return str(e)

# IMPORTANT FOR RAILWAY
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
