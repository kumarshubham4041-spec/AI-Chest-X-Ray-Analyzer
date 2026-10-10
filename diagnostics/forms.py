from django import forms
from .models import XRay

class XRayForm(forms.ModelForm):
    class Meta:
        model = XRay
        # We only want the user to upload an image. 
        # The AI will fill in the 'prediction' behind the scenes.
        fields = ['image']