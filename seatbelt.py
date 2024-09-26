print("bisho")


import os
import cv2
# import yaml
import boto3
import shutil
import mediapipe as mp
from decimal import Decimal
from ultralytics import YOLO
from datetime import datetime
import numpy as np


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) 



model=YOLO('cjdarcl_seatbelt.pt')
model_person=YOLO('yolov8n.pt')


def preprocess_image(image_path, target_width, target_height):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return (None, None)  # or raise an exception as per your requirement
    
    # Read the left half of the image
    half_width = image.shape[1] // 2
    left_half_image = image[:, :480,:]
    
    return image,left_half_image


def isbelt(results,seatbelt_visibility_threshold):

    for r in results:
        if r.probs.data[0] > seatbelt_visibility_threshold: 
            return True
        else:
            return False



output_folder_seatbelt='/home/hitech/novus/project1/output_img_for_seatbelt/'
output_folder_noseatbelt='/home/hitech/novus/project1/output_img_for_noseatbelt/'

dirs_folder= '/home/hitech/novus/project1/input_img_for_seatbelt'   
seatbelt_visibility_threshold = 0.05
count=0


for eachDir in os.listdir(dirs_folder):
        images_path = os.path.join(dirs_folder, eachDir)
        count=count+1
        print("Images Folder: ", images_path)

        image, left_cabin_img = preprocess_image(images_path, target_width=480, target_height=720)
        
        results = model(left_cabin_img, stream=True)  # predict on a frame
        seatbelt_visibility_threshold = 0.05
        val=isbelt(results,seatbelt_visibility_threshold)

        if not os.path.isfile(images_path):
            print(f"The image path {images_path} does not exist.")
            break

        # Get the base name of the image (file name)
        image_name = os.path.basename(images_path)

        # Create a full path for the new image in the output folder

        if val is True:
            results = model_person(left_cabin_img, stream=True, save=False)  # predict on a frame
            for r in results:
                predicted_labels =[int(i) for i in  r.boxes.cls.tolist()]
                print(predicted_labels)
                if 0 in predicted_labels:
                    person_index=predicted_labels.index(0)
                    bbox=r.boxes.xywh[person_index].tolist()
                    x,y,w,h=bbox
                    x=int(x)
                    y=int(y)
                    w=int(w)
                    h=int(h)
                    cv2.rectangle(left_cabin_img,(x-int(w/2),y-int(h/2)),(x+int(w/2),y+int(h/2)),(0,0,255),2)
                    cv2.imshow("abc",left_cabin_img)
                    cv2.waitKey(0)
                    print(r.boxes.xywh.tolist())

                # bbox = r.xywh.tolist()

                filename = image_name
                new_image_path = os.path.join(output_folder_seatbelt, image_name)

                cv2.imwrite(new_image_path+filename,left_cabin_img )

            
        else:
            new_image_path = os.path.join(output_folder_noseatbelt, image_name)
            filename = image_name

            cv2.imwrite(new_image_path+filename,left_cabin_img )

        