# violenece-detection-and-gender-classification-code
import cv2
import numpy as np
import joblib
from keras.models import load_model

# Load the trained SVM model for violence detection
clf_violence = joblib.load('svm_model_violence.pkl')

# Load the trained CNN model for gender classification
model_gender = load_model('cnn_model_gender.h5')

# Define the function to detect violence and classify gender
def detect_violence_and_gender(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Define the video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Loop through the video frames
    while(cap.isOpened()):
        # Read the current frame
        ret, frame = cap.read()

        # If the frame is not valid, break the loop
        if not ret:
            break

        # Resize the frame to 224x224 for input to the gender classification model
        frame_resized = cv2.resize(frame, (224, 224))

        # Normalize the pixel values to be between 0 and 1
        frame_normalized = frame_resized / 255.0

        # Use the gender classification model to predict the gender of the person in the frame
        gender_prediction = model_gender.predict(np.array([frame_normalized]))

        # Use the SVM model to predict whether the frame contains violence or not
        # If the person in the frame is female, skip violence detection
        if gender_prediction[0][0] >= 0.5:
            prediction = 0
        else:
            # Resize the frame to 224x224 for input to the violence detection model
            frame_resized = cv2.resize(frame, (224, 224))

            # Normalize the pixel values to be between 0 and 1
            frame_normalized = frame_resized / 255.0

            # Flatten the frame to a 1D array for input to the violence detection model
            frame_flattened = frame_normalized.flatten()

            # Use the SVM model to predict whether the frame contains violence or not
            prediction = clf_violence.predict([frame_flattened])[0]

        # Draw a red rectangle around the frame if it contains violence
        if prediction == 1:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)

        # Display the frame
        cv2.imshow('frame', frame)

        # Wait for 1 millisecond for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to detect violence and classify gender in a video file
detect_violence_and_gender('video path')
