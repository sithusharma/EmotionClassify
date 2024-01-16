import cv2
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model('emotion.h5')
model.load_weights('64pWeightsEmotion.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    # Check if the image is not grayscale (has more than 1 channel)
    if len(frame.shape) > 2 and frame.shape[2] == 3:
        # Convert the image to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # If it's already grayscale, use it as is
        gray_frame = frame

    # Resize, normalize, and expand dimensions as required by your model
    gray_frame = cv2.resize(gray_frame, (48, 48))  # Example size
    gray_frame = gray_frame / 255.0  # Normalizing
    gray_frame = np.expand_dims(gray_frame, axis=-1)  # Add channel dimension
    gray_frame = np.expand_dims(gray_frame, axis=0)   # Expand dimensions
    return gray_frame

def get_emotion_label(prediction):
    emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Suprised"]
    return emotions[np.argmax(prediction)]

def run_emotion_analysis():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
# Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop and preprocess the face for emotion detection
        face = gray[y:y+h, x:x+w]
        processed_face = preprocess_frame(face)
        
        # Predict the emotion
        emotion_prediction = model.predict(processed_face)
        emotion_label = get_emotion_label(emotion_prediction)

        # Display the emotion label
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


