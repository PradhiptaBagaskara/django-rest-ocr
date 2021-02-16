from django.urls import (include, path, re_path)
from rest_framework import routers
from api import views

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    re_path(r'^.*/$',views.handler404,name='error404'),
    path('models/', views.get_model, name='models'),
    path('ocr/<str:model_id>', views.ocr_by_model, name='ocr')
]