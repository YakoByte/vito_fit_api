# urls to be added to the project's urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.ObjectDetectionApi.as_view()),
]