import re
import os
import cv2
import pytesseract
import numpy as np
from io import BytesIO
from PIL import Image
from api.utils import detection_util


class TFOCR(object):
    def __init__(self, imageByte, model_dir):
        self.model_dir = model_dir
        self.image = imageByte
        self.unreadable = "unreadable"
        self.tessdata = "tessdata/"
        self.config=r'-c preserve_interword_spaces=1 -l lex --oem 1 --psm 6'

        self.data = {}
    
    def img_to_PIL(self):
        """
            method convert byte to PIL image
        """
        image = BytesIO(bytearray(self.image))
        img = Image.open(image)
        return img

    def clean_str(self, text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[\~\!\@\#\$\%\^\&\*\_\+\`\"\?\.\(\)\:]+', '', text)
        text = " ".join(text.split())
        return text

    @staticmethod
    def clean_date(text):
        """
        :param text: Unclean date format
        :return: Cleaned date format; dd/mm/yyyy
        """
        out_date = re.sub('[^\d\s\-]', ' ', text)
        out_date = re.sub('[/—-]', '-', out_date)
        out_date = re.sub('[ ]{2,}', ' ', out_date)
        out_date = re.sub(r'((?<=\D)|(?<=^))([0-9])(?=\D)', r'0\2', out_date)
        out_date = re.sub('-', '/', out_date)
        s = re.search(r'(([\d]+)/([\d]+)/([\d]+))', out_date)
        if s:
            out_date = s.group()
        out_date = re.sub(r'[\s]', '', out_date)
        return out_date

    def run_ocr(self):
        img = self.img_to_PIL()
        config = self.config +  r' --tessdata-dir "{}"'.format(re.sub(r"\\","/",self.tessdata))
        ocr = detection_util.ocr_label_to_dict(image= img, model_dir=self.model_dir, tess_config=config)
        return ocr

       



