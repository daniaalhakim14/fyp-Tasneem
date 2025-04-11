import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to create the model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(open('labels.txt').readlines()), activation='softmax'))
    return model

# Function to train the model
def train_model():
    # Load emotion labels from labels.txt
    with open('labels.txt', 'r') as f:
        emotions = [line.strip().split(' ')[1] for line in f.readlines()]

    # Create the model
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Set up data generators
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # No validation split, as we will use separate folders

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/aiman/OneDrive/Documents/assignment-kasatria/tasneem/data/train',  # Training data path
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'  # Use 'categorical' if using one-hot encoding
    )

    validation_generator = train_datagen.flow_from_directory(
        'C:/Users/aiman/OneDrive/Documents/assignment-kasatria/tasneem/data/test',  # Testing data path
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'  # Use 'categorical' if using one-hot encoding
    )

    # Train the model
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

    # Save the model
    model.save('emotion_detection_model.h5')

# Call the train_model function
train_model()

# Load the Keras model
model = load_model('C:/Users/aiman/OneDrive/Documents/assignment-kasatria/tasneem/emotion_detection_model.h5')

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

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
