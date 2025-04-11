import cv2
import numpy as np
from keras.models import load_model

try:
    # Load the Keras model
    model = load_model('C:/Users/aiman/OneDrive/Documents/assignment-kasatria/tasneem/emotion_detection_model.h5')
    
    # Check the model's input shape
    input_shape = model.input_shape
    print("Model Input Shape:", input_shape)
except ValueError as ve:
    print("ValueError loading model or retrieving input shape:", ve)
except Exception as e:
    print("Error loading model or retrieving input shape:", e)

# Load emotion labels from labels.txt
with open('labels.txt', 'r') as f:
    labels = [line.strip().split(' ')[1] for line in f.readlines()]

# Start video capture
cap = cv2.VideoCapture(0)  # Changed camera index to 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Preprocess the frame (resize, etc.)
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to expected input size
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 224, 224, 3)).astype(np.float32)  # Ensure 3 channels

    # Make predictions
    predictions = model.predict(reshaped_frame)
    emotion_index = np.argmax(predictions[0])
    emotion = labels[emotion_index]

    # Implement thresholding logic
    threshold = 0.5  # Set a new threshold for classification
    if predictions[0][emotion_index] < threshold:
        emotion = "uncertain"

    # Debugging output
    print("Output Probabilities:", predictions[0])  # Log the probabilities
    print("Predicted Emotion:", emotion)  # Log the selected emotion

    # Loop through detected faces
    for (x, y, width, height) in faces:
        # Draw a green rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the emotion label above the rectangle
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
