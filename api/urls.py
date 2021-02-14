from django.urls import include, path, re_path
from rest_framework import routers
from api import views

router = routers.DefaultRouter()

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.

urlpatterns = [
    path('', include(router.urls)),
    path('ocr/', views.get_model, name='models'),
    path('ocr/<str:model_id>', views.ocr_by_model, name='ocr')
]