from django.db import models

class XRay(models.Model):
    image = models.ImageField(upload_to='xrays/')
    # Two new fields for our XAI features
    heatmap_image = models.ImageField(upload_to='heatmaps/', blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    
    prediction = models.CharField(max_length=50, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"XRay {self.id} - {self.prediction}"
