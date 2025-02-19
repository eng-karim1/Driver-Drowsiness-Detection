# **Driver Drowsiness Detection Using Deep Learning (Raspberry Pi Version)**  

## **Overview**  
This project detects driver drowsiness using deep learning and computer vision techniques. The system monitors three key indicators:
- **Eye closure detection** (to identify closed vs. open eyes)
- **Head tilt detection** (to determine if the driver is leaning excessively)
- **Yawning detection** (to recognize signs of fatigue through mouth movement)

The real-time detection system is implemented using **OpenCV**, **MediaPipe**, and a pre-trained **CNN model** and is optimized to run on **Raspberry Pi** with a **buzzer alert system**.

---

## **Technologies Used**  
- **Python**  
- **TensorFlow/Keras** (for training the deep learning model)  
- **OpenCV** (for image processing)  
- **MediaPipe** (for facial landmark detection)  
- **RPi.GPIO** (for controlling the buzzer)  
- **Matplotlib & Seaborn** (for visualization)  
- **Dlib** (for face alignment)  

---

## **Project Components**  
### **1. Model Training (`Drowsiness_Detection_Train_Model.ipynb`)**  
- Loads and preprocesses image data  
- Builds a **Convolutional Neural Network (CNN)** for classifying **eye state (open/closed)**  
- Trains the model on labeled eye images  
- Saves the trained model as `Driver_Drowsiness_Detection_Model.h5`  

### **2. Real-Time Detection for Raspberry Pi (`raspi_runtime.py`)**  
- Uses **MediaPipe** to detect **face landmarks**  
- Extracts **eye regions** and feeds them into the trained CNN model to classify **eye state**  
- Calculates **head tilt** using facial key points  
- Detects **yawning** based on mouth distance  
- **Triggers a buzzer alert on Raspberry Pi GPIO when drowsiness is detected**  
- Displays real-time results with alerts  

---

## **Setup & Installation on Raspberry Pi**  
### **1. Clone the Repository**  
```bash  
git clone https://github.com/eng-karim1/Driver-Drowsiness-Detection.git  
cd Driver-Drowsiness-Detection/raspberry_pi_version  
```

### **2. Install Dependencies**  
```bash  
pip install numpy opencv-python mediapipe tensorflow imutils seaborn matplotlib dlib RPi.GPIO  
```

### **3. Run Real-Time Detection on Raspberry Pi**  
```bash  
sudo python raspi_runtime.py  
```

---

## **Hardware Requirements**  
- **Raspberry Pi 4 (or compatible version)**  
- **Raspberry Pi Camera or USB Webcam**  
- **Buzzer connected to GPIO pin 17**  

---

## **Results & Performance**  
- The trained CNN model achieves **96% accuracy** in detecting open vs. closed eyes.  
- The system successfully detects drowsiness **in real-time** using a Raspberry Pi camera.  

---

## **Future Improvements**  
- Optimize model inference speed on Raspberry Pi.  
- Enhance detection under low-light conditions.  
- Improve buzzer alert logic for better responsiveness.  

---

### **GitHub Repository Link**  
👉 [GitHub Repository](https://github.com/eng-karim1/Driver-Drowsiness-Detection)  

---

