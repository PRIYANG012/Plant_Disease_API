from config import app, logger
from flask import request, jsonify
from flask_restful import Resource

import datetime

import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.models import Sequential, save_model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


import base64
from io import BytesIO
from PIL import Image




def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# image loading and converting to array
def load_image_for_general(img_path, show=False):
    img = cv2.resize(img_path, (128, 128))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
   
    return img_tensor



# image loading and converting to array
def load_image(img_path, show=False):
    img = cv2.resize(img_path, (224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
   
    return img_tensor




# Prediction Class
class Plant_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            
            # model = load_model("Model/Final_Model_Plant.h5")
            model2 = load_model("Model/Adam_Final11.h5")
            

            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)
            # im_arr is one-dim Numpy array
            im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  
            # converted to image object not stored locally
            img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            # image path
            img_path = img 
           

            # load a single image
            new_image = load_image_for_general(img)

           
            # check prediction
            pred2 = model2.predict(new_image,batch_size=1)

            print(pred2)


            preds=np.argmax(pred2, axis=1)


            if data["plant"]=="Apple":
                # do something
                if preds==0:
                    preds="Apple scab"
                elif preds==1:
                    preds="Apple Black Rot"
                elif preds==2:
                    preds="Cedar Apple Rust"
                elif preds==3:
                    preds="Healthy Apple"
            elif data["plant"]=="Blueberry":
                if preds==4:
                    preds="Healthy Blueberry"
                 
                #do something
            elif data["plant"]=="Cherry":
                if preds==5:
                    preds="Healthy Cherry"
                elif preds==6:
                    preds="Cherry Powdery Mildew"
                # do something
            elif data["plant"]=="Corn":
                if preds==7:
                    preds="Corn (maize) Cercospora leaf spot Gray leaf spot"
                elif preds==8:
                    preds="Corn Comman Rust"
                elif preds==9:
                    preds="Healthy Corn"
                elif preds==10:
                    preds="Corn Northen leaf blight"
                # do something
            elif data["plant"]=="Grape":
                if preds==11:
                    preds="Grape black Rot"
                elif preds==12:
                    preds="Grape Esca (Black Meales)"
                elif preds==13:
                    preds="Grape Leaf Blight"
                elif preds==14:
                    preds="Healthy Grape"
                #do something
            elif data["plant"]=="Orange":
                if preds==15:
                    preds="Orange Haunglongbing (Citrus_greening)"
               
            elif data["plant"]=="Peach":
                if preds==16:
                    preds="Bacterial spot"
                elif preds==17:
                    preds="Healthy Peach"
                #do something
            elif data["plant"]=="Pepper":
                if preds==18:
                    preds="Bacterial spot"
                elif preds==19:
                    preds="Healthy Pepper Bell"
                # do something
            elif data["plant"]=="Potato":
                if preds==20:
                    preds="Healthy Potato"
                elif preds==21:
                    preds="Early blight"
                elif preds==22:
                    preds="Late blight"
                #do something
            elif data["plant"]=="Raspberry":
                if preds==23:
                    preds="Healthy Raspberry"
                
                # do something
            elif data["plant"]=="Soyabean":
                if preds==24:
                    preds="Soyabean Healthy"
                
                #do something
            elif data["plant"]=="Squash":
                if preds==25:
                    preds="Squash Powdery mildew"
                   
                # do something
            elif data["plant"]=="Strawberry":
                if preds==26:
                    preds="Healthy Strawberry"
                elif preds==27:
                    preds="Strawberry Leaf Scorch"
                #do something
            else:   #for Tomato
                if preds==28:
                    preds="Bacterial_spot"
                elif preds==29:
                    preds="Early_blight"
                elif preds==30:
                    preds="Late_blight"
                elif preds==31:
                    preds="Leaf_Mold"
                elif preds==32:
                    preds="Septoria_leaf_spot"
                elif preds==33:
                    preds="Spider_mites Two-spotted_spider_mite"
                elif preds==34:
                    preds="Target_Spot"
                elif preds==35:
                    preds="Tomato_Yellow_Leaf_Curl_Virus"
                elif preds==36:
                    preds="Tomato_mosaic_virus"
                else:
                    preds="Healthy"


                # do something
            



            return{
                "Predicted result":preds
               
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500




#Tomato Prediction Class
class Tomato_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            
            # model = load_model("Model/Final_Model_Plant.h5")
            model2 = load_model("Model/model_inception.h5")
            

            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)
            # im_arr is one-dim Numpy array
            im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  
            # converted to image object not stored locally
            img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
            # image path
            img_path = img 
           

            # load a single image
            new_image = load_image(img)

           
            # check prediction
            pred2 = model2.predict(new_image,batch_size=1)

            print(pred2)


            preds=np.argmax(pred2, axis=1)
            if preds==0:
                preds="Bacterial_spot"
                remedies="Once leaf spot has infected your plants, spray with copper fungicide for seven to 10 days. After that, continue to treat every 10 days when weather is dry or every five to seven days when weather is wet. Copper fungicide can also be used preventively after sowing seeds but before moving plants into the garden.Preventive treatments are recommended, as the loss resulting from a bacterial leaf spot infection can be devastating. In addition to preventive copper fungicide treatments, gardeners should ensure their seeds are certified disease-free and soil is sterile, whether you sterilize your own soil or purchase commercial soils. If seeds aren’t sterile, soak them in 1.3% sodium hypochlorite for one minute to sterilize them on your own. Crop rotation and avoiding too-wet conditions are other strategies to prevent leaf spot. Opt for drip irrigation, or water plants at their base instead of from overhead, and do your watering in the morning instead of later in the day."
            elif preds==1:
                preds="Early_blight"
            elif preds==2:
                preds="Late_blight"
            elif preds==3:
                preds="Leaf_Mold"
            elif preds==4:
                preds="Septoria_leaf_spot"
            elif preds==5:
                preds="Spider_mites Two-spotted_spider_mite"
            elif preds==6:
                preds="Target_Spot"
            elif preds==7:
                preds="Tomato_Yellow_Leaf_Curl_Virus"
            elif preds==8:
                preds="Tomato_mosaic_virus"
            else:
                preds="Healthy"
                remedies="If you’re trying to grow the world’s biggest tomato and you have the time to remove all the suckers as your tomato plant grows, then you might want to go ahead and sucker your plants (and be sure to sanitize your tools as you go so you don’t spread diseases)."


            return{
                "Predicted result":preds,
                "remedies":remedies
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500

