# **Driver Drowsiness Detection Using Deep Learning**  

## **Overview**  
This project aims to detect driver drowsiness using deep learning and computer vision techniques. The model monitors three key indicators of drowsiness:  
- **Eye closure detection** (to identify closed vs. open eyes)  
- **Head tilt detection** (to determine if the driver is leaning excessively)  
- **Yawning detection** (to recognize signs of fatigue through mouth movement)  

The real-time detection system is implemented using **OpenCV**, **MediaPipe**, and a pre-trained **CNN model**.

---

## **Technologies Used**  
- **Python**  
- **TensorFlow/Keras** (for training the deep learning model)  
- **OpenCV** (for image processing)  
- **MediaPipe** (for facial landmark detection)  
- **Matplotlib & Seaborn** (for visualization)  
- **Dlib** (for face alignment)  

---

## **Project Components**  
### **1. Model Training (`Drowsiness_Detection_Train_Model.ipynb`)**  
- Loads and preprocesses image data  
- Builds a **Convolutional Neural Network (CNN)** for classifying **eye state (open/closed)**  
- Trains the model on labeled eye images  
- Saves the trained model as `Driver_Drowsiness_Detection.h5`  

### **2. Real-Time Detection (`Drowsiness_Detection_Runtime.py`)**  
- Uses **MediaPipe** to detect **face landmarks**  
- Extracts **eye regions** and feeds them into the trained CNN model to classify **eye state**  
- Calculates **head tilt** using facial key points  
- Detects **yawning** based on mouth distance  
- Displays real-time results with alerts  

---

## **Setup & Installation**  
### **1. Clone the Repository**  
```bash
git clone https://github.com/eng-karim1/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

### **2. Install Dependencies**  
```bash
pip install numpy opencv-python mediapipe tensorflow imutils seaborn matplotlib dlib
```

### **3. Run Real-Time Detection**  
```bash
python Drowsiness_Detection_Runtime.py
```

---

## **Results & Performance**  
- The trained CNN model achieves **96% accuracy** in detecting open vs. closed eyes.  
- The system successfully detects drowsiness **in real-time** using a webcam.  

---

## **Future Improvements**  
- Enhance detection under different lighting conditions.  
- Integrate a voice alert system for increased safety.  
- Optimize the CNN model for faster real-time inference.

---

### **GitHub Repository Link (To Be Updated)**  
👉 [GitHub Repository](https://github.com/eng-karim1/Driver-Drowsiness-Detection)  

---
