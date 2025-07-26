import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
app.secret_key = "your_secret_key"  # Replace with your secret key

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the pre-trained models once at startup
# Make sure these files exist in the 'models' folder
cnn_model = load_model(os.path.join("models", "cnn_model.keras"))
densenet_model = load_model(os.path.join("models", "densenet_model.keras"))
efficientnet_model = load_model(os.path.join("models", "efficientnet_model.keras"))
# Categories of skin conditions
categories = ['acne', 'dark spots', 'normal skin', 'puffy eyes', 'wrinkles']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# Function to predict the skin condition
def predict_skin_condition(model_type, img_path):
    # Select the model
    if model_type == 'cnn':
        model = cnn_model
    elif model_type == 'densenet':
        model = densenet_model
    elif model_type == 'efficientnet':
        model = efficientnet_model
    else:
        raise ValueError("Invalid model type. Choose 'cnn', 'densenet', or 'efficientnet'.")

    # Preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0

    # Perform prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return categories[class_index]

# Treatment suggestions
def suggest_treatment(condition):
    treatments = {
        'acne': [
            "Rửa mặt với sữa rửa mặt chứa acid salicylic 2 lần/ngày.",
            "Bôi kem chứa benzoyl peroxide lên vùng da bị mụn.",
            "Dưỡng ẩm với kem không gây bít tắc lỗ chân lông.",
            "Luôn dùng kem chống nắng (SPF 30+) vào ban ngày."
        ],
        'dark spots': [
            "Sử dụng sữa rửa mặt tẩy tế bào chết dịu nhẹ chứa glycolic acid.",
            "Thoa serum Vitamin C mỗi ngày để làm sáng da.",
            "Bảo vệ da với kem chống nắng phổ rộng.",
            "Có thể cân nhắc các liệu trình chuyên sâu như peel hóa học."
        ],
        'normal skin': [
            "Rửa mặt với sữa rửa mặt dịu nhẹ 2 lần/ngày.",
            "Dưỡng ẩm với kem chứa hyaluronic acid.",
            "Thoa kem chống nắng SPF 30+ vào buổi sáng.",
            "Duy trì chế độ ăn uống lành mạnh và uống đủ nước."
        ],
        'puffy eyes': [
            "Chườm lạnh vùng mắt vài phút vào buổi sáng.",
            "Dùng kem mắt chứa caffeine hoặc retinol.",
            "Hạn chế ăn mặn và uống đủ nước mỗi ngày.",
            "Ngủ với đầu cao hơn một chút."
        ],
        'wrinkles': [
            "Rửa mặt với sữa rửa mặt dịu nhẹ, dưỡng ẩm.",
            "Thoa serum chứa retinol vào buổi tối.",
            "Dùng kem dưỡng chứa peptide giúp tăng đàn hồi da.",
            "Luôn dùng kem chống nắng để ngăn lão hóa."
        ],
    }
    return treatments.get(condition, ["Hãy tham khảo ý kiến bác sĩ da liễu để được tư vấn phù hợp."])

def condition_to_vietnamese(condition):
    mapping = {
        'acne': 'Da mụn',
        'dark spots': 'Thâm nám',
        'normal skin': 'Da bình thường',
        'puffy eyes': 'Có quầng thâm và bọng mắt',
        'wrinkles': 'Da có nếp nhăn',
    }
    return mapping.get(condition, condition)

# Landing page (index.html)
@app.route('/')
def index():
    return render_template("index.html")

# Upload page (upload.html)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part in the request.")
            print("DEBUG: No file part in request")
            return redirect(request.url)

        file = request.files['file']

        if file.filename == "":
            flash("No file selected.")
            print("DEBUG: No file selected")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            print(f"DEBUG: File saved at {file_path}")

            try:
                condition = predict_skin_condition('densenet', file_path)
                treatment = suggest_treatment(condition)
                condition_vn = condition_to_vietnamese(condition)
                print(f"DEBUG: Predicted condition - {condition}")
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                print(f"DEBUG: Error processing image - {str(e)}")
                return redirect(request.url)

            return render_template("results.html",
                                   image_path=url_for('static', filename=f"uploads/{filename}"),
                                   condition=condition_vn,
                                   treatment=treatment)
        else:
            flash("Invalid file type. Please upload an image file.")
            print("DEBUG: Invalid file type")
            return redirect(request.url)

    return render_template("upload.html")

if __name__ == '__main__':
    # Turn on debugging for development
    app.run(debug=True)
