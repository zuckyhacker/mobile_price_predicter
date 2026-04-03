from flask import Flask, request, render_template
import joblib
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and files
model = joblib.load("model.pkl")
features = pickle.load(open("features.pkl", "rb"))
le = pickle.load(open("brand_encoder.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            brand = request.form["brand"]
            ram = int(request.form["ram"])
            battery = int(request.form["battery"])
            memory = int(request.form["memory"])
            camera = int(request.form["camera"])
            weight = int(request.form["weight"])

            brand_encoded = le.transform([brand])[0]

            input_data = pd.DataFrame(
                [[ram, battery, memory, camera, weight, brand_encoded]],
                columns=features
            )

            pred = model.predict(input_data)
            prediction = int(pred[0])

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


# IMPORTANT: Only this block should run locally
if __name__ == "__main__":
    app.run(debug=True)
