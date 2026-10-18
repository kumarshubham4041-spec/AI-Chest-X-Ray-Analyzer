# AI Chest X-Ray Analyzer 🩺

An end-to-end HealthTech web application that utilizes a fine-tuned deep learning model to detect up to 14 different pulmonary conditions from chest X-ray images, featuring explainable AI (Grad-CAM) and LLM-generated clinical summaries.

## 🚀 Project Overview
This project bridges the gap between machine learning and medical web applications. By wrapping a custom PyTorch computer vision model and a local Large Language Model (LLM) within a Django web server, it provides a seamless, user-friendly interface for uploading medical images, visualizing AI decision-making, and receiving readable diagnostic reports.

### Key Features:
* **Multi-Condition Detection:** Analyzes chest X-rays across 14 distinct pathologies (e.g., Infiltration, Cardiomegaly, Effusion) alongside standard "Normal" classifications.
* **Explainable AI (Grad-CAM):** Automatically generates heatmaps over the uploaded X-ray to highlight the specific structural features driving the AI's prediction.
* **LLM Clinical Summaries:** Integrates a local Llama 3.2 model to dynamically translate raw probability scores into professional, easy-to-read clinical interpretations with appropriate medical guardrails.
* **Full-Stack Deployment:** Integrated the PyTorch CV model and Ollama API directly into a Django backend with a modern, polished HTML/CSS frontend dashboard.

## 🛠️ Tech Stack
* **Backend:** Python, Django
* **Computer Vision:** PyTorch, Torchvision (ResNet architecture)
* **Generative AI:** Ollama (Llama 3.2)
* **Image Processing:** Pillow (PIL), NumPy
* **Frontend:** HTML5, Custom CSS (Inter font)

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/kumarshubham4041-spec/ai-chest-xray-analyzer.git] (https://github.com/kumarshubham4041-spec/ai-chest-xray-analyzer.git)
   cd ai-chest-xray-analyzer
2.**Create and activate a virtual environment:**
  ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3.**Install dependencies:**
   ```bash
   pip install django torch torchvision pillow requests numpy tzdata
   Start the Local LLM Server:
  
```
4.**Ensure you have Ollama installed on your machine. Open a separate terminal and run:**
   ```Bash
   ollama run llama3.2
```
5.**Run the local Django development server:**

```Bash
   python manage.py runserver
```
6.**Access the Application:**
```bash
   Open your browser and navigate to http://127.0.0.1:8000.
