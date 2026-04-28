from flask import Flask, render_template, request
import os
import requests

app = Flask(__name__)   # 👈 ये missing है तुम्हारे code में
import requests  # 👈 जरूरी

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')

        if not file:
            return "No file"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ✅ सही API
        API_URL = "https://api-inference.huggingface.co/models/nateraw/vit-base-beans"

        headers = {
            "Authorization": "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"
        }

        response = requests.post(
            API_URL,
            headers=headers,
            data=open(filepath, "rb").read()
        )

        if response.status_code != 200:
            return f"API Error: {response.text}"

        result = response.json()

        best = max(result, key=lambda x: x['score'])
        disease = best['label']
        confidence = round(best['score'] * 100, 2)

        return render_template("output.html",
                               prediction=disease,
                               confidence=confidence)

    except Exception as e:
        return str(e)


    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
