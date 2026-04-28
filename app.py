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
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def preprocess(filepath):
    img = Image.open(filepath).resize((150,150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        crop = request.form['crop']

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img_array = preprocess(filepath)

        # Model select
        if crop == "cotton":
            interpreter = cotton_model
            classes = ['aphids','army_worm','bacterial_blight','curl','fussarium_wilt','healthy','herbicide_growth_damage','leaf_hopper_jassids','leaf_redding','leaf_variegation','powdery_mildew','target_spot']
        elif crop == "soyabean":
            interpreter = soyabean_model
            classes = ['bacterial_blight','brown_spot','caterpillar','healthy','mosaic_virus','powdery_mildew']
        elif crop == "wheat":
            interpreter = wheat_model
            classes = ['brown_rust','healthy','yellow_rust']
        else:
            return "Invalid crop"

        # Prediction
        prediction = predict_tflite(interpreter, img_array)
        class_index = np.argmax(prediction)
        disease = classes[class_index]
        confidence = round(100 * float(np.max(prediction)), 2)

        return render_template(
            "output.html",
            prediction=disease,
            confidence=confidence,
            treatment="Spray fungicide",
            dose="2 ml per litre",
            image_path=filepath,
            product=None
        )

    except Exception as e:
        return str(e)

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
