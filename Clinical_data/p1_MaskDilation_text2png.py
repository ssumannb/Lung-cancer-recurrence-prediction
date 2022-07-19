import numpy as np
import os
import cv2
from tqdm import tqdm


def txt2png(srcDir,dstDir,DICOMdir):

    txtList = os.listdir(srcDir)    ## load text file list

    for file in tqdm(txtList):
        maskInTxt = np.loadtxt(srcDir+'\\{}'.format(file))       ## load text file
        patient_num = file.split('.')[0]
        '''
        text file 구조
        x좌표     y좌표     slice number
        '''
        Z = list(map(int, maskInTxt[:,2])) # slice number
        slice_first = Z[0]
        slice_last = Z[maskInTxt.shape[0]-1]

        try:
            sliceNum = os.listdir('{}{}\\'.format(DICOMdir, patient_num))
            sliceNum = len(sliceNum)
        except:
            continue

        for idx_slice in range(sliceNum):
            empty_png = np.zeros((512,512))

            maskInTxt_slice = maskInTxt[maskInTxt[:,2]==idx_slice]
            if not maskInTxt_slice.any():
                pass
            else:
                X = list(map(int, maskInTxt_slice[:, 0]))
                Y = list(map(int, maskInTxt_slice[:, 1]))
                for px in range(len(maskInTxt_slice)):
                    empty_png[Y[px]][X[px]] = 1

            maskInPng = empty_png[:,:]*255

            dst_path = "{}{}\\".format(dstDir, patient_num)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)

            cv2.imwrite(dst_path+"{}_{}.png".format(patient_num, str(idx_slice+1).zfill(3)), maskInPng)


