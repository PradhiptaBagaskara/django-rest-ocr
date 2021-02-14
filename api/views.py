import os
import glob
import re

from django.shortcuts import render
from rest_framework.response import Response
from django.conf import settings
from rest_framework.permissions import IsAuthenticated  
from django.http import Http404
from rest_framework import status
from rest_framework.decorators import (api_view, permission_classes)
from rest_framework.permissions import IsAuthenticated

from api.ocr import TFOCR


def get_list_models():
    models = glob.glob(os.path.join(settings.MODEL_DIR, "*"))
    data = []
    for path in models:
        model_name = re.sub(r"(.*?)/|(.*?)\\", "",path)
        data.append(model_name)
    return data



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_model(request, format=None):
    data = {
        'total_model': 0,
        'model_id': []
    }
    models = get_list_models()
    if len(models) > 0:
        data['total_model'] = len(models)
        data['model_id'] = models
    return Response(data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def ocr_by_model(request, model_id):
    models = get_list_models()
    if model_id not in models:
        return Response({'detail': 'Invalid Model ID!'}, status=status.HTTP_400_BAD_REQUEST)            

    if request.FILES:
        if 'file' in request.FILES:
            img = request.FILES['file']
            print(img)
            models = glob.glob(os.path.join(settings.MODEL_DIR, "*"))

            model_dir = os.path.join(settings.MODEL_DIR, model_id)
            ocr = TFOCR(img.read(), model_dir)
            filename = {'filename': str(img)}
            result = {**filename, **ocr.run_ocr()}
            
            return  Response(result)
        else:
            return Response({'detail':'invalid key!'}, status=status.HTTP_400_BAD_REQUEST)

    return Response({'detail': 'file is missing!'}, status=status.HTTP_400_BAD_REQUEST)

