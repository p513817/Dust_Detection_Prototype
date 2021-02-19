#
#  -------------------------------------------------------------
#   Copyright (c) Cavedu.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow Lite export package from Lobe.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite

import cv2

def get_prediction(image, interpreter, signature):
    """
    Predict with the TFLite interpreter!
    """
    # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
    signature_inputs = signature.get("inputs")
    input_details = {detail.get("name"): detail for detail in interpreter.get_input_details()}
    model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
    signature_outputs = signature.get("outputs")
    output_details = {detail.get("name"): detail for detail in interpreter.get_output_details()}
    model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}

    if "Image" not in model_inputs:
        raise ValueError("Tensorflow Lite model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")

    # process image to be compatible with the model
    input_data = process_image(image, model_inputs.get("Image").get("shape"))

    # set the input to run
    interpreter.set_tensor(model_inputs.get("Image").get("index"), input_data)
    interpreter.invoke()

    # grab our desired outputs from the interpreter!
    # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
    outputs = {key: interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}
    # postprocessing! convert any byte strings to normal strings with .decode()
    for key, val in outputs.items():
        if isinstance(val, bytes):
            outputs[key] = val.decode()

    return outputs


def process_image(image, input_shape):
    """
    Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
    """
    width, height = image.size
    # ensure image type is compatible with model and convert if not
    if image.mode != "RGB":
        image = image.convert("RGB")
    # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
    if width != height:
        square_size = min(width, height)
        left = (width - square_size) / 2
        top = (height - square_size) / 2
        right = (width + square_size) / 2
        bottom = (height + square_size) / 2
        # Crop the center of the image
        image = image.crop((left, top, right, bottom))
    # now the image is square, resize it to be the right shape for the model input
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.reshape(input_shape).astype(np.float32)

class modify_lobe():

    def __init__(self, model_path):
    
        self.model_path = model_path

        with open( model_path + "/signature.json", "r") as f:
            self.signature = json.load(f)

        self.model_file = self.signature.get("filename")

        self.interpreter = tflite.Interpreter(self.model_path + '/' + self.model_file)
        self.interpreter.allocate_tensors()
        self.count = 0

        self.all_confid, self.res, self.val = 0, 0, 0 

    def detect(self, frame):

        self.count = self.count + 1

        if self.count % 10:
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

            prediction = get_prediction(image, self.interpreter, self.signature)
            
            self.all_confid = prediction['Confidences']
            self.res = self.signature['classes']['Label'][prediction['Confidences'].index(max(prediction['Confidences']))]
            self.val = str(max(prediction['Confidences']))
            
            # print(Label_name)
            # print('Confidences = ' + str(max(prediction['Confidences'])) )
            return self.all_confid, self.res, self.val

        else:
            return self.all_confid, self.res, self.val

def main():
    """
    Load the model and signature files, start the TF Lite interpreter, and run prediction on the image.

    Output prediction will be a dictionary with the same keys as the outputs in the signature.json file.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='Model path of .tflite file and .json file', required=True)
    args = parser.parse_args()
    
    with open( args.model + "/signature.json", "r") as f:
        signature = json.load(f)

    model_file = signature.get("filename")

    interpreter = tflite.Interpreter(args.model + '/' + model_file)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0)
    #擷取畫面 寬度 設定為640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    #擷取畫面 高度 設定為480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    key_detect = 0
    times=1
    while (key_detect==0):
        ret,frame =cap.read()

        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        if (times==1):
            prediction = get_prediction(image, interpreter, signature)

        # print('Result = '+ prediction["Prediction"])
        print(prediction)

        Label_name = signature['classes']['Label'][prediction['Confidences'].index(max(prediction['Confidences']))]
        print(Label_name)
        print('Confidences = ' + str(max(prediction['Confidences'])) )
 
        cv2.putText(frame, Label_name + " " +
                    str(round(max(prediction['Confidences']),3)),
                    (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow('Detecting....',frame)

        times=times+1
        if (times >= 10):
            times=1

        read_key = cv2.waitKey(1)
        if ((read_key & 0xFF == ord('q')) or (read_key == 27) ):
            key_detect = 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
