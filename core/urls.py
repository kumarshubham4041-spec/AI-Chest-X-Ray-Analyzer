from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from diagnostics import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.upload_xray, name='upload_xray'), 
    # Add this new line for the PDF download
    path('download/<int:pk>/', views.download_pdf_report, name='download_pdf_report'),
    path('dashboard/', views.analytics_dashboard, name='dashboard'),
    path('api/analyze/', views.AnalyzeXrayAPI.as_view(), name='api_analyze_xray'),
]

# This allows Django to display the uploaded X-ray images on the webpage
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)