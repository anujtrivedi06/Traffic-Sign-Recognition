
# 🚦 Traffic Sign Recognition using CNN

This project builds a Convolutional Neural Network (CNN) to classify traffic signs from the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news). The trained model achieves **96% test accuracy** and is deployed as a Streamlit web application to allow real-time image predictions.

---

## 📌 Project Overview

- 🧠 CNN-based multi-class image classification (43 classes)
- 🧹 Preprocessed and normalized image data
- 📈 Achieves **~96% accuracy** on the test set
- 📊 Model evaluation using confusion matrix and prediction visualization
- 🌐 Real-time deployment via Streamlit
- 🧪 Dataset: GTSRB (via Kaggle, preprocessed)

---

## 📁 Folder Structure

```
Traffic-Sign-Recognition/
├── app.py                  # Streamlit frontend
├── label_names.csv           # Class ID to label mapping
├── traffic_sign_model.keras  # Trained model
├── traffic-Recognition.ipynb # Training pipeline
├── README.md
├── Requirements.txt
├── labels.pickle
└── .gitignore
```

---

## 🚀 Getting Started

### 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/<your-username>/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition
```

2. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

### 💾 Dataset

This project uses the preprocessed dataset available on Kaggle:

**[valentynsichkar/traffic-signs-preprocessed](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed)**

> Download the dataset and place the `train.pickle`, `test.pickle`, and `label_names.csv` files in the appropriate folders before running the notebook.

---

## 🧠 Model Architecture

The CNN model consists of:

- 3 convolutional layers with ReLU activation
- MaxPooling and Dropout layers for regularization
- A dense layer with L2 regularization
- Final `softmax` layer with 43 output classes

```text
Input → Conv2D → ReLU → Dropout →
Conv2D → ReLU → MaxPooling → Dropout →
Conv2D → MaxPooling →
Flatten → Dense → Dropout → Softmax
```

---

## 📊 Evaluation

### ✅ Accuracy Metrics:
- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~96%
- **Test Accuracy:** **96.13%**

The model performs consistently across training, validation, and test sets, indicating good generalization.

### 🔍 Visual Analysis:
- Confusion matrix highlights strong per-class performance
- Sample predictions confirm model reliability on unseen images

---

## 🌐 Streamlit App

### To Run the App Locally:
```bash
cd app
streamlit run app.py
```

### Features:
- Upload any `.jpg`, `.jpeg`, or `.png` image
- Model resizes and classifies it
- Returns predicted traffic sign label in real-time

---

## 🖼️ Sample Screenshot

> <img width="185" height="335" alt="image" src="https://github.com/user-attachments/assets/f8d38eed-c26e-4166-9365-ddfa55e891bd" />


---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas / Matplotlib
- scikit-learn
- Streamlit

---

## 👤 Author

**Anuj Trivedi**  
_B.Tech Graduate | Machine Learning and DSA Enthusiast_  

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/anuj-trivedi-2a538827a/)

---

