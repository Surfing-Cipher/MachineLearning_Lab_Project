import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import base64

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'face_mask_detector_final.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# --- LOAD ASSETS ---
print(f"Loading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

print(f"Loading haar cascade from {CASCADE_PATH}...")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("Error loading cascade classifier! Make sure the XML file is in the correct path.")

# --- PREPROCESSING & INFERENCE ---
def predict_mask(face_img):
    """
    Predicts whether a face has a mask or not.
    Args:
        face_img: Numpy array of the face (RGB)
    Returns:
        label (str), color (tuple)
    """
    # Resize to 224x224 matches MobileNetV2 input
    face = cv2.resize(face_img, (224, 224))
    face = img_to_array(face)
    # Preprocessing typically involves scaling to [-1, 1] for MobileNetV2 or [0, 1] generically.
    # We will assume standard [0, 1] or MobileNet specific. 
    # MobileNetV2 usually expects [-1, 1], but many custom trainings use [0, 1] (/255.0).
    # Since the user didn't specify preprocessing details of THEIR training, we'll try standard /255.0
    # If the user trained with `tf.keras.applications.mobilenet_v2.preprocess_input`, it might need to be different.
    # Given "beginner friendly" requests usually imply simple rescaling:
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    # Predict
    # Output is sigmoid (0 to 1). < 0.5 usually class 0, > 0.5 usually class 1.
    # We need to know which class is which. Typically:
    # 0: Mask, 1: No Mask   OR   0: No Mask, 1: Mask.
    # We will try a convention. If it's inverted, we can flip it.
    # Common alphabetical order: Mask (0), No Mask (1)? Or 'with_mask', 'without_mask'.
    # Assumption for this logical flow: 0=Mask, 1=NoMask is common, but let's assume
    # the user said "binary classifier" and output sigmoid. 
    # Let's assume Low Prob = Mask (0), High Prob = No Mask (1).
    # We'll label based on threshold.
    
    pred = model.predict(face, verbose=0)[0][0]
    
    # NOTE: You may need to swap these depending on your class indices!
    # If your folder structure was 'with_mask' and 'without_mask', 'with_mask' comes first alphabetically? 
    # actually 'with_mask' vs 'without_mask': 'with_mask' is 0, 'without_mask' is 1.
    if pred < 0.5:
        label = "With Mask"
        color = (0, 255, 0) # Green
    else:
        label = "Without Mask"
        color = (0, 0, 255) # Red
        
    label = "{}: {:.2f}%".format(label, (max(pred, 1-pred)) * 100)
    return label, color

def detect_and_predict(frame):
    """
    Detects faces in a frame and draws bounding boxes with predictions.
    Args:
        frame: BGR numpy array (OpenCV standard)
    Returns:
        frame: Annotated BGR numpy array
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Convert frame to RGB for model (OpenCV is BGR, Keras expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame_rgb[y:y+h, x:x+w]
        
        if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
            continue

        # Predict
        label, color = predict_mask(face_roi)

        # Draw box and label on original BGR frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
        cv2.putText(frame, label, (x+6, y-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    return frame

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0) # Open default camera
    if not camera.isOpened():
        print("Cannot open camera")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Run detection and prediction
        frame = detect_and_predict(frame)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Read image into memory
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            in_memory_file.seek(0)
            
            # Convert to numpy array
            file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process image
            img_processed = detect_and_predict(img)
            
            # Encode back to PNG for display
            ret, buffer = cv2.imencode('.png', img_processed)
            img_str = base64.b64encode(buffer).decode('utf-8')
            img_base64 = f"data:image/png;base64,{img_str}"
            
            return render_template('result.html', image_data=img_base64)
            
    return render_template('upload.html')

if __name__ == '__main__':
    # 'debug=True' enables auto-reload on code changes
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
