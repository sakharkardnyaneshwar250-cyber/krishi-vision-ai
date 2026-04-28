from flask import Flask, render_template, request
import numpy as np
import os
import requests
import razorpay
"Authorization": "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"
app = Flask(__name__)

client = razorpay.Client(auth=("rzp_test_SifWo24DcJ7yuT", "iqd4FlZKPXFMutoA19Rhgitd"))

# ---------- PRODUCTS ----------
products = [
    {"id": 1, "name": "Bavistin", "type": "Fungicide", "price": 350, "image": "bavistin.jpg"},
    {"id": 2, "name": "Mancozeb", "type": "Fungicide", "price": 200, "image": "mancozeb.jpg"},
    {"id": 3, "name": "Imidacloprid", "type": "Insecticide", "price": 250, "image": "imidacloprid.jpg"},
    {"id": 4, "name": "Chlorothalonil", "type": "Fungicide", "price": 300, "image": "chlorothalonil.jpg"},
    {"id": 5, "name": "Dimethoate", "type": "Insecticide", "price": 180, "image": "dimethoate.jpg"}
]

# ---------- CONFIG ----------
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- PRODUCT MAPPING ----------
product_map = {
    "aphids": "Imidacloprid",
    "brown_spot": "Mancozeb",
    "septoria": "Chlorothalonil",
    "leaf_blight": "Chlorothalonil",
    "stem_fly": "Dimethoate",
    "mildew": "Bavistin"
}

# ---------- SAFE VOICE ----------
def speak(text):
    print("Voice:", text)

# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')

        if not file:
            return "No file uploaded"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 🔥 HuggingFace API
        API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
        
        headers = {
            "Authorization": "Bearer YOUR_API_KEY"
        }

        with open(filepath, "rb") as f:
            response = requests.post(API_URL, headers=headers, data=f.read())

        result = response.json()

        # Safe handling
        if isinstance(result, list):
            disease = result[0]['label']
            confidence = round(result[0]['score'] * 100, 2)
        else:
            disease = "Unknown"
            confidence = 0

        return render_template(
            "output.html",
            prediction=disease,
            confidence=confidence,
            treatment="Consult expert",
            dose="-",
            image_path=filepath,
            product=None
        )

    except Exception as e:
        return str(e)
        # 🔥 STABLE DEMO (NO AI → NO CRASH)
        disease = "Leaf Spot"
        confidence = 92
        treatment = "Spray fungicide"
        dose = "2 ml per litre"

        # PRODUCT FIND
        recommended_product = None
        if disease in product_map:
            for p in products:
                if p["name"] == product_map[disease]:
                    recommended_product = p
                    break

        return render_template(
            "output.html",
            prediction=disease,
            confidence=confidence,
            treatment=treatment,
            dose=dose,
            image_path=filepath,
            product=recommended_product
        )

    except Exception as e:
        return str(e)


@app.route('/cart')
def cart():
    return render_template("cart.html")


@app.route('/search')
def search():
    query = request.args.get('q')
    results = [p for p in products if query and query.lower() in p["name"].lower()]
    return render_template("search.html", results=results)


@app.route('/pay/<int:amount>')
def pay(amount):
    order = client.order.create({
        "amount": amount * 100,
        "currency": "INR",
        "payment_capture": 1
    })
    return render_template("pay.html", order=order)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
