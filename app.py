from flask import Flask, render_template, request
import numpy as np
import os
import razorpay
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

client = razorpay.Client(auth=("rzp_test_SifWo24DcJ7yuT", "iqd4FlZKPXFMutoA19Rhgitd"))

# ---------- CONFIG ----------
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- LOAD MODELS (FIXED) ----------
cotton_model = tflite.Interpreter(model_path="cotton_model.tflite")
cotton_model.allocate_tensors()

soyabean_model = tflite.Interpreter(model_path="soyabean_model.tflite")
soyabean_model.allocate_tensors()

wheat_model = tflite.Interpreter(model_path="wheat_model.tflite")
wheat_model.allocate_tensors()

# ---------- HELPER ----------
def predict_tflite(model, img_array):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    model.invoke()

    return model.get_tensor(output_details[0]['index'])

# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    crop = request.form['crop']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # ✅ IMAGE FIX
    img = Image.open(filepath).resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

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

    # ✅ PREDICTION FIX
    prediction = predict_tflite(model, img_array)

    class_index = np.argmax(prediction)
    disease = classes[class_index]
    confidence = round(100 * float(np.max(prediction)), 2)

    treatment, dose = solutions.get(disease, ("Consult expert", "Not available"))

    # PRODUCT FIND (same)
    recommended_product = None
    if disease in product_map:
        for p in products:
            if p["name"] == product_map[disease]:
                recommended_product = p
                break

    # RESULT LOGIC (same)
    if confidence < 70:
        disease_final = "Not sure (कृपया स्पष्ट फोटो द्या)"
        treatment_final = "Please upload clear image"
        dose = "-"
    else:
        disease_mr = marathi_names.get(disease, disease)
        disease_final = f"{disease_mr} ({disease})"
        treatment_final = treatment

    # FINAL RETURN
    return render_template(
        "output.html",
        prediction=disease_final,
        confidence=confidence,
        treatment=treatment_final,
        dose=dose,
        image_path=filepath,
        product=recommended_product
    )
