from django.db import models

class XRay(models.Model):
    # This stores the actual uploaded image
    image = models.ImageField(upload_to='xrays/')
    # This will store the AI's final diagnosis (Normal or Pneumonia)
    prediction = models.CharField(max_length=100, blank=True)
    # This automatically records the exact date and time of the upload
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"X-Ray uploaded on {self.uploaded_at}"
