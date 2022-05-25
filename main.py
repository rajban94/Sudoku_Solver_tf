##### Importing Libraries #####

print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from utils import *
from solver import *
import argparse
import sys

##### Initialization of model #####

model = intializePredectionModel() 


##### image path and image size declare #####

args = get_args()
if args.image:
    if not os.path.isfile(args.image):
        print("[!] ==> Input image file {} doesn't exist".format(args.image))
        sys.exit(1)

result_path = args.output_dir
image_path = args.image
image_height = 450
image_width = 450
image_blank = np.zeros((image_height, image_width, 3), np.uint8)

##### Convert to Gray Scale and apply Threshold #####

image = cv2.imread(image_path)
image = cv2.resize(image,(image_width,image_height))
image_Threshold = apply_threshold(image)

##### find biggest contour #####

contours = draw_contours(image)
biggest,max_area = largest_contours(contours)

if biggest.size != 0:
    biggest = reorder_contours(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0],[image_width, 0], [0, image_height],[image_width, image_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(image, matrix, (image_width, image_height))
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

##### split image and find each digit #####
    
    boxes = split_boxes(imgWarpColored)
    numbers = getPredection(boxes, model)
    # print(np.matrix(numbers))
    imgDetectedDigits = image_blank.copy()
    imgDetectedDigits = display_numbers(imgDetectedDigits, numbers, color=(0, 0, 255))
    # cv2.imwrite('ques.jpg',imgDetectedDigits)
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    
##### Solve the Sudoku #####

    imgSolvedDigits = image_blank.copy()
    board = np.array_split(numbers,9)
    
    try:
        solve(board)
    except:
        pass
    print(np.matrix(board))
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= display_numbers(imgSolvedDigits,solvedNumbers,color=(0, 255, 0))

    ##### marge the solution image with question image #####

    pts2 = np.float32(biggest)
    pts1 =  np.float32([[0, 0],[image_width, 0], [0, image_height],[image_width, image_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = image.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (image_width, image_height))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, image, 0.5, 1)
    imgDetectedDigits = draw_grid(imgDetectedDigits)
    imgSolvedDigits = draw_grid(imgSolvedDigits)
    
    ##### Concat Unsolved and solved Image #####
    im_h = cv2.hconcat([image.copy(), inv_perspective])
    
    cv2.imwrite(result_path+'Solution_'+os.path.basename(image_path),im_h)