import cv2
import os
import pickle
from operator import itemgetter
import numpy as np
import pandas as pd
import openface
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '.', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# modelDir = "./models"
# dlibModelDir = "./models/dlib"
# openfaceModelDir = "./model/openface"

size = 96
dlibFacePredictor = default = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
align = openface.AlignDlib(dlibFacePredictor)
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
net = openface.TorchNeuralNet(networkModel, imgDim=size)
classifierModel = "./generated-embeddings/classifier.pkl"

def getRepImg(imgPath, multiple=True):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception("Unable to find a face: {}".format(imgPath))
    reps = []
    for bb in bbs:        
        alignedFace = align.align(
            size,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps
      
def processImage(img, multiple=True):
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)
    print("\n=== {} ===".format(img))
    reps = getRepImg(img)
    if len(reps) > 1:
        print("List of faces in image from left to right")
    lst = []
    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if multiple:
            print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                     confidence))
            lst.append(person)
        else:
            print("Predict {} with {:.2f} confidence.".format(person, confidence))
    return lst    

def getRepVid(bgrImg):
    if bgrImg is None:
        raise Exception("Unable to load image/frame")
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)
    if bb is None:
        return None
    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                size,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))
    return reps
    
def processFrame(frame):
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer
    reps = getRepVid(frame)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        predictions = clf.predict_proba(rep).ravel()
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        confidences.append(predictions[maxI])        
    return (persons, confidences)
    
def predictFromVideo(videoPath):
    video_capture = cv2.VideoCapture(videoPath)
    time = 0
    confidenceList = []
    result = set()
    success = True
    while success:
        video_capture.set(cv2.CAP_PROP_POS_MSEC, time)
        time += 300
        success, frame = video_capture.read()
        if success:
            persons, confidences = processFrame(frame)
            #print "P: " + str(persons) + " C: " + str(confidences)
            try:
                # append with two floating point precision
                #confidenceList.append('%.2f' % confidences[0])
                i =0
                for person in persons:
                    if confidences[i] > 0.09:
                        result.add(person)
                        print person
                    i = i+1
            except:
                # If there is no face detected, confidences matrix will be empty.
                # We can simply ignore it.
                pass
            #for i, c in enumerate(confidences):
                #if c <= 0.1:  # 0.5 is kept as threshold for known face.
                    #persons[i] = "_unknown"
    
                    # Print the person name and conf value on the frame
            #cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
            #            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #cv2.imshow('', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
    # When everything is done, release the capture
    video_capture.release()
    #cv2.destroyAllWindows()
    print result
    return result
    
def identify_images(path):
    result = {}
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for image_path in image_paths:
        lst = processImage(image_path)
        result[image_path] = lst
    #print result
    return result  
    
        
#processImage("/home/ishan/Desktop/Project/newApp/test3.JPG")
#identify_images("/media/akash/Storage/Projects/FinalYear/CROP")
#processVideo("/home/akash/Videos/nosound270480.mkv")
