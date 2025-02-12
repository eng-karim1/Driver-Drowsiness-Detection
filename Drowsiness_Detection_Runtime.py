import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the model
model = load_model("Driver_Drowsiness_Detection.h5")

# Define labels for eye state
labels = ["Closed", "Open"]

def preprocess_image(eye):
    # Resize and prepare eye image for model prediction
    eye = cv2.resize(eye, (32, 32))
    eye = np.expand_dims(eye, axis=0)
    eye = np.array(eye, dtype=np.float32)
    return eye

def predict_keras(image):
    # Predict eye state and return the label
    predictions = model.predict(image)
    prediction = np.argmax(predictions)
    return labels[prediction]

# Setup Mediapipe for face detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame color for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Default states for eye detection
    left_eye_status = "Unknown"
    right_eye_status = "Unknown"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Detect head tilt by calculating angle between eyes
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            eye_dx = right_eye.x - left_eye.x
            eye_dy = right_eye.y - left_eye.y
            angle = np.arctan2(eye_dy, eye_dx) * 180 / np.pi
            head_tilt = "Straight"
            if angle > 10:
                head_tilt = "Right"
            elif angle < -10:
                head_tilt = "Left"
            cv2.putText(frame, f"Head Tilt: {head_tilt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Detect yawning by checking lip distance
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            mouth_open = abs(upper_lip.y - lower_lip.y) > 0.05
            if mouth_open:
                cv2.putText(frame, "Yawning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Crop eye regions for model prediction
            h, w, _ = frame.shape
            left_eye_points = [face_landmarks.landmark[i] for i in [33, 133, 160, 159, 158, 144]]
            right_eye_points = [face_landmarks.landmark[i] for i in [362, 263, 387, 386, 385, 380]]

            left_eye_coords = [(int(pt.x * w), int(pt.y * h)) for pt in left_eye_points]
            right_eye_coords = [(int(pt.x * w), int(pt.y * h)) for pt in right_eye_points]

            left_eye_rect = cv2.boundingRect(np.array(left_eye_coords))
            right_eye_rect = cv2.boundingRect(np.array(right_eye_coords))

            left_eye_crop = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
            right_eye_crop = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]

            # Predict and update eye state if eye crop is available
            if left_eye_crop.size > 0:
                left_eye_status = predict_keras(preprocess_image(left_eye_crop))

            if right_eye_crop.size > 0:
                right_eye_status = predict_keras(preprocess_image(right_eye_crop))

    # Display eye state on frame
    cv2.putText(frame, f"Left Eye: {left_eye_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Eye: {right_eye_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close OpenCV windows
