import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import sys
import os

def intializePredectionModel():
    model = load_model('/home/ce/Downloads/all/payslip/Sudoku_Game/Sudoku_tensorflow/models/sudoku_model.h5')
    return model

def apply_threshold(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray,(5,5),1)
    image_Threshold = cv2.adaptiveThreshold(image_blur, 255, 1, 1, 11, 2)
    return image_Threshold

def reorder_contours(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def draw_contours(img):
    image_Threshold = apply_threshold(img)
    image_contours = img.copy()
    contours, hierarchy = cv2.findContours(image_Threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 3)
    return contours

def largest_contours(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def split_boxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def getPredection(boxes,model):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        ## GET PREDICTION
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        ## SAVE TO RESULT
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

def display_numbers(img,numbers,color = (0,0,255)):
    w = int(img.shape[1]/9)
    h = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*w+int(w/2)-10, int((y+0.8)*h)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

def draw_grid(img):
    w = int(img.shape[1]/9)
    h = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,h*i)
        pt2 = (img.shape[1],h*i)
        pt3 = (w * i, 0)
        pt4 = (w*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='',
                        help='path to image file')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='path to the output directory')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        print('==> Creating the {} directory...'.format(args.output_dir))
        os.makedirs(args.output_dir)
    else:
        print('==> Skipping create the {} directory...'.format(args.output_dir))
    return args