import argparse
import cv2
import numpy as np
import os
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '.', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def alignMain(outputDir, inputDir):
    openface.helper.mkdirP(outputDir)
    size = 96
    imgs = list(iterImgs(inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }

    landmarkIndices = landmarkMap['outerEyesAndNose']

    dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
    align = openface.AlignDlib(dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(outputDir, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"
        rgb = imgObject.getRGB()
        if rgb is None:
            outRgb = None
        else:
            outRgb = align.align(size, rgb,
                                 landmarkIndices=landmarkIndices)
        if outRgb is not None:
            outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(imgName, outBgr)
            
def run_script():
    os.system("./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/")
        
#alignMain('./aligned-images/','./training-images/')
#run_script()
