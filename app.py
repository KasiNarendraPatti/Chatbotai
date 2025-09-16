from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
import os, random

# ML imports
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- Initialize Gemini client ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

client = genai.Client(api_key=api_key)

# --- Load plant disease model ---
MODEL_PATH = "plant_disease_model.h5"
plant_model = None

# PlantVillage dataset (38 classes)
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

if os.path.exists(MODEL_PATH):
    plant_model = load_model(MODEL_PATH)
    print("✅ Plant disease model loaded successfully!")
else:
    print("⚠️ Plant disease model not found. Running in placeholder mode.")

# --- Default page ---
@app.route("/")
def home():
    return render_template("index.html")

# --- Chat endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"reply": "⚠️ Please enter a message."})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"You are an AI assistant for farmers. Answer simply: {user_message}",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    return jsonify({"reply": response.text})

# --- Crop recommendation ---
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    ph = data.get("ph", 6.5)
    nitrogen = data.get("nitrogen", 120)
    phosphorus = data.get("phosphorus", 80)
    potassium = data.get("potassium", 70)
    rainfall = data.get("rainfall", 200)
    temperature = data.get("temperature", 28)

    crops_data = {
        "Rice": {"ph": (5.5, 7.0), "N": (80, 150), "P": (50, 100), "K": (40, 90),
                 "rainfall": (150, 300), "temp": (20, 35)},
        "Wheat": {"ph": (6.0, 7.5), "N": (90, 140), "P": (50, 100), "K": (40, 80),
                  "rainfall": (100, 200), "temp": (10, 25)},
        "Maize": {"ph": (5.5, 7.5), "N": (100, 160), "P": (40, 90), "K": (30, 70),
                  "rainfall": (150, 250), "temp": (18, 32)},
        "Cotton": {"ph": (6.0, 8.0), "N": (80, 120), "P": (40, 80), "K": (30, 60),
                   "rainfall": (50, 150), "temp": (20, 35)},
        "Sugarcane": {"ph": (6.0, 7.5), "N": (90, 150), "P": (50, 100), "K": (40, 90),
                      "rainfall": (150, 300), "temp": (20, 35)},
    }

    def score_crop(crop):
        params = crops_data[crop]
        score = 0
        score += 1 if params["ph"][0] <= ph <= params["ph"][1] else 0
        score += 1 if params["N"][0] <= nitrogen <= params["N"][1] else 0
        score += 1 if params["P"][0] <= phosphorus <= params["P"][1] else 0
        score += 1 if params["K"][0] <= potassium <= params["K"][1] else 0
        score += 1 if params["rainfall"][0] <= rainfall <= params["rainfall"][1] else 0
        score += 1 if params["temp"][0] <= temperature <= params["temp"][1] else 0
        return score

    crop_scores = {crop: score_crop(crop) for crop in crops_data}
    recommended_crops = sorted(crop_scores, key=lambda x: crop_scores[x], reverse=True)[:3]

    return jsonify({
        "crops": recommended_crops,
        "yield_estimate": round(random.uniform(2.5, 5.0), 2),
        "profit_estimate": round(random.uniform(400, 1200), 2),
        "sustainability_score": random.choice(["High", "Medium", "Low"])
    })

# --- Plant health detection ---
@app.route("/plant-health", methods=["POST"])
def plant_health():
    file = request.files.get("image")
    if not file:
        return jsonify({"status": "Error", "advice": "No image uploaded."})

    if plant_model:
        # Preprocess image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))  # Hugging Face model uses 224x224
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = plant_model.predict(x)
        idx = np.argmax(preds)
        status = class_labels[idx]

        return jsonify({
            "status": status,
            "advice": f"Detected: {status}. Please follow recommended treatment or consult a local expert."
        })
    else:
        return jsonify({
            "status": "Healthy",
            "advice": "Model not loaded, returning placeholder result."
        })

# --- Market trends ---
@app.route("/market-trends", methods=["GET"])
def market_trends():
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    prices = [200, 250, 230, 280, 300, 320]
    return jsonify({"labels": labels, "prices": prices})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
