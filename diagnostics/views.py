import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from .forms import XRayForm
from .models import XRay

# 1. Rebuild the AI Architecture (Must match exactly what we built in Colab)
class MedicalAI:
    def __init__(self):
        # We use CPU here because inference on one image is very fast
        self.device = torch.device('cpu') 
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        
        # Load your trained weights from the .pth file
        model_path = os.path.join(settings.BASE_DIR, 'diagnostics', 'pneumonia_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Put the AI in 'Guessing Mode', not 'Learning Mode'

    def diagnose(self, image_path):
        # Apply the exact same image rules we used in Colab
        image_rules = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Open the image, apply rules, and pass it to the AI
        img = Image.open(image_path).convert('RGB')
        tensor = image_rules(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            
        classes = ['Normal', 'Pneumonia']
        return classes[predicted.item()]

# Load the brain into memory once when the server starts
ai_brain = MedicalAI()

# 2. Define What Happens When a User Visits the Page
def upload_xray(request):
    if request.method == 'POST':
        form = XRayForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the image to the database
            xray_record = form.save()
            
            # Pass the image to the AI for diagnosis
            diagnosis = ai_brain.diagnose(xray_record.image.path)
            
            # Save the AI's diagnosis back to the database
            xray_record.prediction = diagnosis
            xray_record.save()
            
            # Send the user to a results page
            return render(request, 'diagnostics/result.html', {'xray': xray_record})
    else:
        form = XRayForm()
        
    return render(request, 'diagnostics/upload.html', {'form': form})
