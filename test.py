from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
import os, random

app = Flask(__name__)

# --- Initialize Gemini client with explicit API key ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

client = genai.Client(api_key=api_key)

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

    # Dummy logic for now
    if ph < 5.5:
        crops = ["Groundnut", "Potato"]
    elif 6 <= ph <= 7.5:
        crops = ["Rice", "Wheat", "Maize"]
    else:
        crops = ["Cotton", "Sugarcane"]

    return jsonify({
        "crops": crops,
        "yield_estimate": round(random.uniform(2.5, 5.0), 2),
        "profit_estimate": round(random.uniform(400, 1200), 2),
        "sustainability_score": random.choice(["High", "Medium", "Low"])
    })

# --- Plant health ---
@app.route("/plant-health", methods=["POST"])
def plant_health():
    file = request.files.get("image")
    # Placeholder: in real use, pass to a vision model
    return jsonify({
        "status": "Healthy",
        "advice": "No major issues detected. Keep monitoring soil moisture."
    })

# --- Market trends ---
@app.route("/market-trends", methods=["GET"])
def market_trends():
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    prices = [200, 250, 230, 280, 300, 320]  # Example prices
    return jsonify({"labels": labels, "prices": prices})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
