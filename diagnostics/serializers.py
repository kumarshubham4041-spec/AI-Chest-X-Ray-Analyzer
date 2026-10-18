from rest_framework import serializers
from .models import XRay

class XRaySerializer(serializers.ModelSerializer):
    class Meta:
        model = XRay
        fields = ['id', 'image', 'heatmap_image', 'prediction', 'confidence', 'uploaded_at']
        read_only_fields = ['heatmap_image', 'prediction', 'confidence', 'uploaded_at']