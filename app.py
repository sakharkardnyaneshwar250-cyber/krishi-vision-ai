from flask import Flask, render_template, request
import numpy as np
import os
import razorpay

app = Flask(__name__)

client = razorpay.Client(auth=("rzp_test_SifWo24DcJ7yuT", "iqd4FlZKPXFMutoA19Rhgitd"))

# ---------- PRODUCTS ----------
products = [
    {
        "id": 1,
        "name": "Bavistin",
        "type": "Fungicide",
        "price": 350,
        "image": "bavistin.jpg"
    },
    {
        "id": 2,
        "name": "Mancozeb",
        "type": "Fungicide",
        "price": 200,
        "image": "mancozeb.jpg"
    },
    {
        "id": 3,
        "name": "Imidacloprid",
        "type": "Insecticide",
        "price": 250,
        "image": "imidacloprid.jpg"
    },
    {
        "id": 4,
        "name": "Chlorothalonil",
        "type": "Fungicide",
        "price": 300,
        "image": "chlorothalonil.jpg"
    },
    {
        "id": 5,
        "name": "Dimethoate",
        "type": "Insecticide",
        "price": 180,
        "image": "dimethoate.jpg"
    }
]

# ---------- CONFIG ----------
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- LOAD MODELS ----------
cotton_model = load_model("cotton_model.h5")
soyabean_model = load_model("soyabean_model.h5")
wheat_model = load_model("wheat_model.h5")

# ---------- CLASS LABELS ----------
cotton_classes = [
    'aphids','army_worm','bacterial_blight','curl',
    'fussarium_wilt','healthy','herbicide_growth_damage',
    'leaf_hopper_jassids','leaf_redding','leaf_variegation',
    'powdery_mildew','target_spot'
]

soyabean_classes = [
    'bacterial_blight','brown_spot','caterpillar',
    'crestamento','diabrotica_speciosa','ferrugen',
    'healthy','mosaic_virus','powdery_mildew',
    'septoria','southern_blight','sudden_death_syndrome',
    'yellow_mosaic'
]

wheat_classes = [
    'aphid',
    'black_rust',
    'blast',
    'brown_rust',
    'common_root_rot',
    'fusarium_head_blight',
    'healthy',
    'leaf_blight',
    'mildew',
    'mite',
    'septoria',
    'smut',
    'stem_fly',
    'tan_spot',
    'yellow_rust'
]

# ---------- SOLUTIONS ----------
solutions = {
    "aphids": ("Imidacloprid", "0.3 ml per liter"),
    "army_worm": ("Spinosad", "0.5 ml per liter"),
    "bacterial_blight": ("Copper oxychloride", "2.5 g per liter"),
    "curl": ("Neem oil", "5 ml per liter"),
    "fussarium_wilt": ("Carbendazim", "1 g per liter"),
    "leaf_hopper_jassids": ("Thiamethoxam", "0.25 g per liter"),
    "leaf_redding": ("NPK spray", "As per label"),
    "leaf_variegation": ("Micronutrient spray", "As per label"),
    "powdery_mildew": ("Sulfur fungicide", "2 g per liter"),
    "target_spot": ("Mancozeb", "2.5 g per liter"),
    "herbicide_growth_damage": ("Flush with water", "No chemical"),

    "brown_spot": ("Mancozeb", "2 g per liter"),
    "caterpillar": ("Chlorantraniliprole", "0.4 ml per liter"),
    "crestamento": ("Balanced fertilizer", "As per label"),
    "diabrotica_speciosa": ("Imidacloprid", "0.3 ml per liter"),
    "ferrugen": ("Triazole fungicide", "1 ml per liter"),
    "mosaic_virus": ("Control whiteflies", "Neem oil spray"),
    "septoria": ("Chlorothalonil", "2 g per liter"),
    "southern_blight": ("Carbendazim", "1 g per liter"),
    "sudden_death_syndrome": ("Improve drainage", "No chemical"),
    "yellow_mosaic": ("Imidacloprid", "0.3 ml per liter"),

    "aphid": ("Imidacloprid (इमिडाक्लोप्रिड)", "0.3 ml per liter (0.3 मि.ली. प्रति लिटर)"),
    
    "black_rust": ("Propiconazole (प्रोपिकोनाझोल)", "1 ml per liter (1 मि.ली. प्रति लिटर)"),
    
    "blast": ("Tricyclazole (ट्रायसायक्लाझोल)", "0.6 g per liter (0.6 ग्रॅम प्रति लिटर)"),
    
    "brown_rust": ("Mancozeb (मॅन्कोजेब)", "2 g per liter (2 ग्रॅम प्रति लिटर)"),
    
    "common_root_rot": ("Carbendazim (कार्बेन्डाझिम)", "1 g per liter (1 ग्रॅम प्रति लिटर)"),
    
    "fusarium_head_blight": ("Tebuconazole (टेब्युकोनाझोल)", "1 ml per liter (1 मि.ली. प्रति लिटर)"),
    
    "leaf_blight": ("Chlorothalonil (क्लोरोथॅलोनील)", "2 g per liter (2 ग्रॅम प्रति लिटर)"),
    
    "mildew": ("Sulfur fungicide (गंधक आधारित बुरशीनाशक)", "2 g per liter (2 ग्रॅम प्रति लिटर)"),
    
    "mite": ("Abamectin (अॅबामेक्टिन)", "0.5 ml per liter (0.5 मि.ली. प्रति लिटर)"),
    
    "septoria": ("Mancozeb (मॅन्कोजेब)", "2 g per liter (2 ग्रॅम प्रति लिटर)"),
    
    "smut": ("Carboxin (कार्बॉक्सिन)", "2 g per kg seed (2 ग्रॅम प्रति किलो बियाणे)"),
    
    "stem_fly": ("Dimethoate (डायमेथोएट)", "1 ml per liter (1 मि.ली. प्रति लिटर)"),
    
    "tan_spot": ("Propiconazole (प्रोपिकोनाझोल)", "1 ml per liter (1 मि.ली. प्रति लिटर)"),
    
    "yellow_rust": ("Hexaconazole (हेक्साकोनाझोल)", "1 ml per liter (1 मि.ली. प्रति लिटर)"),

    "healthy": ("No treatment needed (उपचार आवश्यक नाही)", "Maintain proper care (योग्य देखभाल करा)")
}

# ---------- PRODUCT MAPPING ----------
product_map = {
    "aphids": "Imidacloprid",
    "brown_spot": "Mancozeb",
    "septoria": "Chlorothalonil",
    "leaf_blight": "Chlorothalonil",
    "stem_fly": "Dimethoate",
    "mildew": "Bavistin"
}

# ---------- MARATHI ----------
marathi_names = {
    "caterpillar": "अळी",
    "aphids": "मावा",
    "army_worm": "लष्करी अळी",
    "bacterial_blight": "जैविक डाग रोग",
    "brown_spot": "तपकिरी डाग",
    "powdery_mildew": "भुरी रोग",
    "yellow_mosaic": "पिवळा मोझॅक",
    "mosaic_virus": "मोझॅक विषाणू",
    "septoria": "सेप्टोरिया",
    "healthy": "निरोगी",

    "Chlorantraniliprole": "क्लोरॅन्ट्रानिलिप्रोल",
    "Imidacloprid": "इमिडाक्लोप्रिड",
    "Mancozeb": "मॅन्कोझेब",
    "Carbendazim": "कार्बेन्डाझिम",
    "Neem oil": "कडुनिंब तेल",
    "Sulfur fungicide": "गंधक फवारणी",

    "black_rust": "काळा गंज रोग",
    "blast": "ब्लास्ट रोग",
    "brown_rust": "तपकिरी गंज रोग",
    "common_root_rot": "मुळ कुज रोग",
    "fusarium_head_blight": "फ्युजेरियम हेड ब्लाइट",
    "healthy": "निरोगी",
    "leaf_blight": "पान करपा रोग",
    "mildew": "पांढरी बुरशी",
    "mite": "कोळी कीड",
    "smut": "स्मट रोग",
    "stem_fly": "खोड माशी",
    "tan_spot": "टॅन स्पॉट रोग",
    "yellow_rust": "पिवळा गंज रोग"
}

# ---------- SAFE VOICE ----------
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    def speak(text):
        engine.say(text)
        engine.runAndWait()
except:
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
        crop = request.form.get('crop')

        if not file:
            return "No file uploaded"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 🔥 STABLE DEMO OUTPUT (no ML, no crash)
        disease = "Leaf Spot"
        confidence = 92
        treatment = "Spray fungicide"
        dose = "2 ml per litre"

        # PRODUCT FIND (same logic)
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

    # Model selection
    if crop == "cotton":
        model = cotton_model
        classes = cotton_classes
    elif crop == "soyabean":
        model = soyabean_model
        classes = soyabean_classes
    elif crop == "wheat":
        model = wheat_model
        classes = wheat_classes
    else:
        return "Invalid crop"

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    disease = classes[class_index]
    confidence = round(100 * np.max(prediction), 2)

    treatment, dose = solutions.get(disease, ("Consult expert", "Not available"))

    # 🔥 PRODUCT FIND
    recommended_product = None
    if disease in product_map:
        for p in products:
            if p["name"] == product_map[disease]:
                recommended_product = p
                break

    # 🔥 RESULT LOGIC
    if confidence < 70:
        disease_final = "Not sure (कृपया स्पष्ट फोटो द्या)"
        treatment_final = "Please upload clear image"
        dose = "-"
    else:
        disease_mr = marathi_names.get(disease, disease)
        disease_final = f"{disease_mr} ({disease})"
        treatment_final = treatment

    # 🔊 Voice
    if confidence < 70:
        speak("रोग ओळखता आला नाही. कृपया स्पष्ट फोटो द्या")
    else:
        speak(f"रोग आहे {disease}")

    # 🔥 FINAL RETURN (IMPORTANT)
    return render_template(
        "output.html",
        prediction=disease_final,
        confidence=confidence,
        treatment=treatment_final,
        dose=dose,
        image_path=filepath,
        product=recommended_product
    )
@app.route('/cart')
def cart():
    return render_template("cart.html")

@app.route('/search')
def search():
    query = request.args.get('q')

    results = [p for p in products if query.lower() in p["name"].lower()]

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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
