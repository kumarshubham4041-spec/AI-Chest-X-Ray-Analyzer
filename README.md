# AI-Powered Pneumonia Diagnostic Web App 🩺

An end-to-end web application that utilizes a fine-tuned deep learning model to detect pneumonia from chest X-ray images in real-time. 

## 🚀 Project Overview
This project bridges the gap between machine learning and web development by wrapping a custom-trained PyTorch model within a Django web server. It provides a seamless, user-friendly interface for uploading medical images and receiving instant diagnostic predictions.

### Key Features:
* **Binary Classification:** Accurately classifies chest X-rays as either *Normal* or *Pneumonia*.
* **Custom AI Architecture:** Utilizes a ResNet18 convolutional neural network (CNN), fine-tuned specifically for medical image analysis.
* **Full-Stack Deployment:** Integrated the PyTorch model directly into a Django backend with automated tensor transformations and inference routing.
* **Dynamic UI:** Clean HTML/CSS frontend for image uploads and result rendering.

## 🛠️ Tech Stack
* **Backend:** Python, Django
* **Machine Learning:** PyTorch, Torchvision
* **Image Processing:** Pillow (PIL)
* **Frontend:** HTML5

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```Bash
   git clone [https://github.com/kumarshubham4041-spec/AI-Pneumonia-Diagnostic-App.git](https://github.com/kumarshubham4041-spec/AI-Pneumonia-Diagnostic-App.git)
   cd AI-Pneumonia-Diagnostic-App
2. **Create and activate a virtual environment:**

    ```Bash
    python -m venv venv
    venv\Scripts\activate
3. **Install dependencies:**
    ```Bash
    pip install django torch torchvision pillow

4. **Run the local development server:**
    ```Bash
    python manage.py runserver
5. **Open your browser and navigate to**
    http://127.0.0.1:8000/.
   
