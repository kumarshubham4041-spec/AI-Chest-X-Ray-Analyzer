from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from diagnostics import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # This makes our upload page the main homepage
    path('', views.upload_xray, name='upload_xray'), 
]

# This allows Django to display the uploaded X-ray images on the webpage
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)