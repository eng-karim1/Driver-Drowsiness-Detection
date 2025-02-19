import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO
import time

# إعداد البازر
BUZZER_PIN = 17  # منفذ GPIO المتصل بالبازر
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Load the model
model = load_model("D:\\Graduation Project\\DDDgpt\\projects\\DDD\\Driver_Drowsiness_Detection\\Driver_Drowsiness_Detection_Model.h5")

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

# إعداد الموقتات
last_eye_close_time = None
is_buzzer_on = False
buzzer_sleep_time = 0.3  # وقت التأخير بين الأصوات (طق طق طق)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame color for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Default states for driver drowsiness detection
    is_drowsy = False
    drowsiness_status = "Not Drowsy"

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
                is_drowsy = True
            elif angle < -10:
                head_tilt = "Left"
                is_drowsy = True
            cv2.putText(frame, f"Head Tilt: {head_tilt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Detect yawning by checking lip distance
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            mouth_open = abs(upper_lip.y - lower_lip.y) > 0.05
            if mouth_open:
                is_drowsy = True
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

            # Predict and update drowsiness state if eye crop is available
            if left_eye_crop.size > 0:
                left_eye_status = predict_keras(preprocess_image(left_eye_crop))
                if left_eye_status == "Closed":
                    if last_eye_close_time is None:
                        last_eye_close_time = time.time()
                    elif time.time() - last_eye_close_time > 2:  # إذا غلق السائق عينيه لمدة 2 ثانية
                        if not is_buzzer_on:
                            is_buzzer_on = True
                            GPIO.output(BUZZER_PIN, GPIO.HIGH)
                            time.sleep(buzzer_sleep_time)
                            GPIO.output(BUZZER_PIN, GPIO.LOW)
                else:
                    last_eye_close_time = None

            if right_eye_crop.size > 0:
                right_eye_status = predict_keras(preprocess_image(right_eye_crop))
                if right_eye_status == "Closed":
                    if last_eye_close_time is None:
                        last_eye_close_time = time.time()
                    elif time.time() - last_eye_close_time > 2:  # إذا غلق السائق عينيه لمدة 2 ثانية
                        if not is_buzzer_on:
                            is_buzzer_on = True
                            GPIO.output(BUZZER_PIN, GPIO.HIGH)
                            time.sleep(buzzer_sleep_time)
                            GPIO.output(BUZZER_PIN, GPIO.LOW)
                else:
                    last_eye_close_time = None

    # Update drowsiness status
    if is_drowsy:
        drowsiness_status = "Drowsy"
        if not is_buzzer_on:  # إذا كانت الحالة drowsy ولم يكن البازر شغال
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(buzzer_sleep_time)
            GPIO.output(BUZZER_PIN, GPIO.LOW)

    # Display drowsiness status on frame
    cv2.putText(frame, f"Drowsiness Status: {drowsiness_status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the camera
GPIO.cleanup()  # تنظيف منفذ GPIO بعد الانتهاء
