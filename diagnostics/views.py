import os
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
from django.http import FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from torchvision import models, transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from django.core.files.base import ContentFile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from .forms import XRayForm
from .models import XRay
from django.contrib import messages
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import XRaySerializer
from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser
from pytorch_grad_cam import GradCAM
import requests
import json

class MedicalAI:
    def __init__(self):
        self.device = torch.device('cpu') 
        
        # 1. Initialize ResNet-50 architecture
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 14) 
        
        # 2. Load the trained weights
        model_path = os.path.join(settings.BASE_DIR, 'diagnostics', 'multi_disease_resnet50.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 3. Setup Grad-CAM for ResNet-50
        target_layers = [self.model.layer4[-1]]
        self.cam = GradCAM(model=self.model, target_layers=target_layers)

        # 4. Disease classes
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
    def diagnose_and_explain(self, xray_record):
        img_path = xray_record.image.path
        
        # 1. Prepare image for standard prediction
        img_pil = Image.open(img_path).convert('RGB')
        image_rules = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = image_rules(img_pil).unsqueeze(0).to(self.device)
        
        # 2. Get Multi-Label Predictions & Confidences
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs[0])
            
        detected_diseases = []
        for i, prob in enumerate(probabilities):
            if prob.item() > 0.50:
                detected_diseases.append(f"{self.classes[i]} ({round(prob.item() * 100, 1)}%)")
                
        if not detected_diseases:
            prediction_text = "No Findings / Normal"
            confidence_score = round((1.0 - torch.max(probabilities).item()) * 100, 2)
        else:
            prediction_text = ", ".join(detected_diseases)
            confidence_score = round(torch.max(probabilities).item() * 100, 2)

        # ==========================================
        # 3. NEW LLM LOGIC GOES HERE
        # ==========================================
        all_scores = {self.classes[i]: round(prob.item() * 100, 1) for i, prob in enumerate(probabilities)}
        
        # DO THE MATH IN PYTHON: Filter for scores exactly between 15% and 49.9%
        secondary_scores = {k: v for k, v in all_scores.items() if 15.0 <= v < 50.0}
        
        prompt = f"""
        You are a clinical AI assistant generating a patient-facing summary.
        Primary Prediction: {prediction_text} (Confidence: {confidence_score}%)
        Secondary Observations (15%-49%): {secondary_scores}
        
        Write a 2 to 3 paragraph clinical summary following these strict rules:
        1. Start immediately with the analysis. Do not use greetings or mention any technical architecture names like "ResNet".
        2. Explain the primary prediction in clear, empathetic, plain English. If it is "No Findings" or "Normal", reassure the patient that no significant pulmonary abnormalities are present.
        3. Only discuss secondary conditions if they appear in the Secondary Observations list: {secondary_scores}. If that list is empty, state clearly that no secondary signals or minor abnormalities were detected. Never introduce or discuss conditions not present in that list.
        4. Conclude with a brief standard medical disclaimer advising the patient to consult a licensed healthcare professional for a complete clinical evaluation.
        
        Do not use markdown, asterisks, or bold text. Keep the tone professional, objective, and empathetic.
        """
        llm_report = "Clinical report generation currently unavailable."

        try:
            # Removed the 'options' block so it can think freely and write as much as it wants
            llm_response = requests.post('http://localhost:11434/api/generate', json={
                'model': 'llama3.2', 
                'prompt': prompt,
                'stream': False
            }, timeout=60)
            
            if llm_response.status_code == 200:
                llm_report = llm_response.json().get('response', 'Report generation failed.')
            else:
                llm_report = f"LLM server responded with status code: {llm_response.status_code}"
                
        except requests.exceptions.Timeout:
            llm_report = "The AI took too long to respond. Please try again."
        except Exception as e:
            llm_report = f"Error connecting to local LLM: {str(e)}"
   
        # ==========================================

        # 4. Generate Grad-CAM Heatmap
        rgb_img_np = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
        
        # 5. Save the new heatmap image to the database
        visualization_pil = Image.fromarray(visualization)
        buffer = io.BytesIO()
        visualization_pil.save(buffer, format='JPEG')
        
        heatmap_filename = f"heatmap_{xray_record.id}.jpg"
        xray_record.heatmap_image.save(heatmap_filename, ContentFile(buffer.getvalue()), save=False)
        
        # CHANGE HERE: Make sure it returns all 3 variables!
        return prediction_text, confidence_score, llm_report

ai_brain = MedicalAI()

def is_valid_xray(image_path):
    """
    OOD Gatekeeper: Verifies if the uploaded image matches the 
    characteristics of a medical chest X-ray.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Invalid image file."

        # 1. Robust Color Check using HSV Saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # The 'S' (Saturation) channel is at index 1.
        # 0 = completely grayscale, 255 = fully saturated color.
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        
        # Compressed grayscale JPEGs usually stay under a saturation of 10.
        # Normal color photos typically exceed 50.
        if mean_saturation > 15:
            return False, "Out-of-Distribution Error: The uploaded image contains too much color to be a medical X-ray."

        # 2. Brightness Check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness > 160:
            return False, "Out-of-Distribution Error: The image is too bright. Medical X-rays have dark backgrounds."

        # 3. Variance / Texture Check
        variance = np.var(gray)
        if variance < 600:
            return False, "Out-of-Distribution Error: The image lacks standard radiographic structural contrast."

        return True, "Valid"

    except Exception as e:
        return False, f"Image processing error: {str(e)}"

def upload_xray(request):
    if request.method == 'POST':
        form = XRayForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the file temporarily so the Gatekeeper can look at it
            xray_record = form.save()
            xray_record.save()
            
            # --- GATEKEEPER CHECK ---
            is_valid, error_message = is_valid_xray(xray_record.image.path)
            
            if not is_valid:
                xray_record.image.delete(save=False)
                # If OOD, delete the invalid image from the database and show an error
                xray_record.delete()
                messages.error(request, error_message)
                return render(request, 'diagnostics/upload.html', {'form': form})
            # ------------------------

            # If it passes the Gatekeeper, proceed to the PyTorch AI
            prediction, confidence, llm_report = ai_brain.diagnose_and_explain(xray_record)
            
            xray_record.prediction = prediction
            xray_record.confidence = confidence
            xray_record.save()
            
            # ---> ADDED llm_report TO THIS DICTIONARY <---
            context = {
                'xray': xray_record,
                'llm_report': llm_report
            }
            return render(request, 'diagnostics/result.html', context)
    else:
        form = XRayForm()
        
    return render(request, 'diagnostics/upload.html', {'form': form})

def download_pdf_report(request, pk):
    # 1. Get the specific X-ray from the database
    xray = XRay.objects.get(pk=pk)

    # 2. Create a temporary buffer to hold the PDF data
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    # 3. Draw the Text Layout
    p.setFont("Helvetica-Bold", 20)
    p.drawString(150, 750, "AI Clinical Diagnostic Report")

    p.setFont("Helvetica", 12)
    p.drawString(50, 700, f"Scan ID: {xray.id}")
    p.drawString(50, 680, f"Date Analyzed: {xray.uploaded_at.strftime('%Y-%m-%d %H:%M')}")

    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, 640, f"AI Prediction: {xray.prediction}")
    p.drawString(50, 620, f"Confidence Score: {xray.confidence}%")

    # 4. Draw the Images side-by-side
    p.setFont("Helvetica", 12)
    if xray.image:
        p.drawString(80, 570, "Original X-Ray")
        p.drawImage(xray.image.path, 50, 350, width=200, height=200)

    if xray.heatmap_image:
        p.drawString(350, 570, "Grad-CAM Heatmap")
        p.drawImage(xray.heatmap_image.path, 320, 350, width=200, height=200)

    # 5. Finalize and Save
    p.showPage()
    p.save()
    buffer.seek(0)

    # 6. Send the file to the user's browser as a download
    return FileResponse(buffer, as_attachment=True, filename=f"Diagnostic_Report_{xray.id}.pdf")

def analytics_dashboard(request):
    # Query the database for statistics
    total_scans = XRay.objects.count()
    pneumonia_count = XRay.objects.filter(prediction='Pneumonia').count()
    normal_count = XRay.objects.filter(prediction='Normal').count()
    
    # Grab the 5 most recent scans for a quick-view table
    recent_scans = XRay.objects.order_by('-uploaded_at')[:5]
    
    context = {
        'total_scans': total_scans,
        'pneumonia_count': pneumonia_count,
        'normal_count': normal_count,
        'recent_scans': recent_scans
    }
    return render(request, 'diagnostics/dashboard.html', context)


class AnalyzeXrayAPI(generics.CreateAPIView):
    serializer_class = XRaySerializer
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        
        if serializer.is_valid():
            xray_record = serializer.save()
            
            # 1. Run the Security Gatekeeper
            is_valid, error_message = is_valid_xray(xray_record.image.path)
            if not is_valid:
                xray_record.delete()
                return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)
                
            # 2. Run the PyTorch Model
            try:
                prediction, confidence = ai_brain.diagnose_and_explain(xray_record)
                xray_record.prediction = prediction
                xray_record.confidence = confidence
                xray_record.save()
                
                result_serializer = self.get_serializer(xray_record)
                return Response(result_serializer.data, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                xray_record.delete()
                return Response({'error': f"AI Processing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)