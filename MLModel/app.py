import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 📌 **Initialize Flask App**
app = Flask(__name__)

# 📂 **Load Trained Model**
MODEL_PATH ="skin_cancer_mobilenetv2_optimized.h5"
model = load_model(MODEL_PATH)

# 📌 **Define Image Preprocessing Function**
def preprocess_image(image_path, target_size=(128, 128)):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# 📌 **Define API Route for Prediction**

@app.route('/',methods=['GET'])
def get_data():
    return jsonify({"success":"true"})



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 **Check if image file is provided**
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)  # Save image temporarily

        # 🔹 **Preprocess Image**
        image = preprocess_image(image_path)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # 🔹 **Extract Additional Metadata**
        age = float(request.form.get("age", 50))  # Default 50 if not provided
        sex = float(request.form.get("sex", 0))  # 0: Male, 1: Female
        loc = float(request.form.get("loc", 0))  # Default 0 if not provided

        # 📌 **Prepare Inputs**
        age = np.array([[age]])
        sex = np.array([[sex]])
        loc = np.array([[loc]])

        # 🔥 **Make Prediction**
        prediction = model.predict([image, age, sex, loc])
        probability = float(prediction[0][0])  # Convert to Python float
        confidence_score = round(probability * 100, 2)  # Convert to percentage

        # ✅ **Return Result**
        return jsonify({
            "prediction": probability,
            "confidence_score": f"{confidence_score}%",
            "diagnosis": "Malignant" if probability > 0.5 else "Benign"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 📌 **Run Flask Server**
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



# import os
# import numpy as np
# import joblib  # For loading the model
# from flask import Flask, request, jsonify
# from PIL import Image
# import io

# # 📌 **Initialize Flask App**
# app = Flask(__name__)

# # 📂 **Load Trained Scikit-learn Model**
# MODEL_PATH = "skin_cancer_mobilenetv2_optimized.h5"
# model = joblib.load(MODEL_PATH)

# # 📌 **Define Image Preprocessing Function**
# def preprocess_image(image):
#     img = Image.open(io.BytesIO(image)).convert("RGB")
#     img = img.resize((128, 128))  # Resize to match training size
#     img_array = np.array(img).flatten().reshape(1, -1)  # Convert to 1D array
#     return img_array

# # 📌 **Define API Route for Prediction**
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # 🔹 **Check if image file is provided**
#         if 'image' not in request.files:
#             return jsonify({"error": "No image file provided"}), 400

#         image_file = request.files['image'].read()
#         image = preprocess_image(image_file)

#         # 🔹 **Extract Additional Metadata**
#         age = float(request.form.get("age", 50))
#         sex = float(request.form.get("sex", 0))
#         loc = float(request.form.get("loc", 0))

#         # 📌 **Prepare Inputs**
#         features = np.hstack([image, [[age, sex, loc]]])  # Combine image + metadata

#         # 🔥 **Make Prediction**
#         probability = model.predict_proba(features)[0][1]  # Get probability for class 1
#         confidence_score = round(probability * 100, 2)

#         # ✅ **Return Result**
#         return jsonify({
#             "prediction": probability,
#             "confidence_score": f"{confidence_score}%",
#             "diagnosis": "Malignant" if probability > 0.5 else "Benign"
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # 📌 **Run Flask Server**
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

