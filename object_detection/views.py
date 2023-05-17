from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
# Create your views here.
# %matplotlib inline

def generate_response(message, detections=None):
    response = {
        "message": message,
        "data": {
            "detections": detections
        } if detections else None
    }
    return response


def generate_error_response(error_message):
    response = {
        "error": error_message,
        "data": None
    }
    return response

class ObjectDetectionApi(APIView):
    # give permissions
    permission_classes = []
    parser_classes = [MultiPartParser, FormParser]
    
    def get(self, request):
        return Response({'message': 'Hello, World!'})

    def post(self, request):
        try:
            image_file = request.FILES.get('image')
            if image_file is None:
                error_response = generate_error_response("Error: No image file found.")
                return Response(error_response)

            
            if image_file is not None:
                # Load the label map into memory
                with open('./object_detection/labelmap.txt', 'r') as f:
                    labels = [line.strip() for line in f.readlines()]

                # Load the Tensorflow Lite model into memory
                interpreter = Interpreter(model_path="./object_detection/detect.tflite")
                interpreter.allocate_tensors()   

                # Get model details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                height = input_details[0]['shape'][1]
                width = input_details[0]['shape'][2]

                float_input = (input_details[0]['dtype'] == np.float32)

                input_mean = 127.5
                input_std = 127.5

                min_conf=0.6
                txt_only=True

                image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imH, imW, _ = image.shape 
                image_resized = cv2.resize(image_rgb, (width, height))
                input_data = np.expand_dims(image_resized, axis=0)


                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if float_input:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()

                # Retrieve detection results
                boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
                classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
                scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

                detections = []

                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        
                        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                        # Draw label
                        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                        detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

                
                # All the results have been drawn on the image, now display the image
                output=[]
                if txt_only == False: # "text_only" controls whether we want to display the image results or just save them in .txt files
                    for detection in detections:
                        # if confidence score is above 80 then append
                        # output.append([detection[0], detection[1]*100])
                        if detection[1]*100 > 70:
                            output.append({"item":detection[0],"confidence": detection[1]*100})
                    
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(12,16))
                    plt.imshow(image)
                    plt.show()
                
                # Save detection results in .txt files (for calculating mAP)
                elif txt_only == True:

                    # print all the detected objects and their confidence scores in the image
                    for detection in detections:
                        # if confidence score is above 80 then append
                        # output.append([detection[0], detection[1]*100])
                        if detection[1]*100 > 70:
                            output.append({"item":detection[0],"confidence": detection[1]*100})
                

                item_counts = {}
                for entry in output:
                    item = entry["item"]
                    if item in item_counts:
                        item_counts[item] += 1
                    else:
                        item_counts[item] = 1

                output_data = []
                for item in item_counts:
                    count = item_counts[item]
                    output_data.append({"item": item, "count": count})

                response = generate_response("Success: Image processing completed.", output_data)

                return Response(response,status=200)
            else:
                error_response = generate_error_response("Error: No image file found.")
                return Response(error_response)
            
        except Exception as e:
            error_response = generate_error_response(f"Error: {str(e)}")
            return Response(error_response,status=400)