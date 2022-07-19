__author__='soominlee@swu.ac.kr'

import numpy as np
import cv2
import os

'''사용자정의함수'''
import p1_MaskDilation_text2png as tp

import sys

print(sys.path[0])
'''
1. text2png()
2. DICOM header read
3. uploading mask.png file and dilation as spacing size: Round(Nmm/spacingSize)
4. airway, chestwall delete
5. dilation mask generation png
6. png2text()
'''

sourceDIR = "D:\\Pycharm\\Prediction\\Mask_org\\"
destDIR = "D:\\Pycharm\\Prediction\\Mask_png\\0mm\\"
dicomDIR = "D:\\Pycharm\\Prediction\\DICOM\\patients_217\\"
tp.txt2png(sourceDIR,destDIR,dicomDIR)