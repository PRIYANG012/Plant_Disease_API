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
from keras.preprocessing.image import img_to_array,save_img
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO
from PIL import Image

#Tomato Prediction Class
class Tomato_Prediction(Resource):
    def get(self):
        try:
            msg="It's working perfectly"
            return{
                "Status":msg,
                
            },201
        
        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500
        
    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Tomato_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                preds="Tomato___Bacterial_spot"
            elif preds==1:
                preds="Tomato___Early_blight"
            elif preds==2:
                preds="Tomato___Late_blight"
            elif preds==3:
                preds="Tomato___healthy"
            elif preds==4:
                preds="Tomato___Leaf_Mold"
            elif preds==5:
                preds="Tomato___Septoria_leaf_spot"
            elif preds==6:
                preds="Tomato___Spider_mites Two-spotted_spider_mite"
            elif preds==7:
                preds="Tomato___Target_Spot"
            elif preds==8:
                preds="Tomato___Tomato_mosaic_virus"
            else:
                preds="Tomato___Tomato_Yellow_Leaf_Curl_Virus"

            return{
                "Predicted result":preds,
                
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500




#Apple Prediction Class
class Apple_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Apple_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Apple scab"
            elif preds==1:
                pass_preds="Apple Black_rot"
            elif preds==2:
                pass_preds="Cedar apple rust"
            else:
                pass_preds="Healthy"


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500





#Cherry Prediction Class
class Cherry_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Cherry_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Cherry_(including_sour)___Powdery_mildew"
            elif preds==1:
                pass_preds="Cherry_(including_sour)___healthy"
            


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500





#Corn Prediction Class
class Corn_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Corn_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
            elif preds==1:
                pass_preds="Corn_(maize)___Northern_Leaf_Blight"
            elif preds==2:
                pass_preds="Corn_(maize)___healthy"
            elif preds==3:
                pass_preds="Corn_(maize)___Common_rust_"
            


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500




#Grape Prediction Class
class Grape_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Grape_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Grape___Esca_(Black_Measles)"
            elif preds==1:
                pass_preds="Grape___healthy"
            elif preds==2:
                pass_preds="Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
            elif preds==3:
                pass_preds="Grape___Black_rot"
            
            


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500



#Peach Prediction Class
class Peach_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Peach_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Peach___Bacterial_spot"
            elif preds==1:
                pass_preds="Peach___healthy"
            


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500




#Pepper Prediction Class
class Pepper_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Pepper_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Pepper,_bell___Bacterial_spot"
            elif preds==1:
                pass_preds="Pepper,_bell___healthy"
            


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500



#Potato Prediction Class
class Potato_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Potato_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Potato___Early_blight"
            elif preds==1:
                pass_preds="Potato___healthy"
            elif preds==2:
                pass_preds="Potato___Late_blight"
            


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500



#Strawberry Prediction Class
class Strawberry_Prediction(Resource):

    def post(self):
        try:

            data = request.json
            Model_Apple = load_model("Model/Strawberry_model_inception.h5")
            im_b64=data["image"]
            im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
            im_file = BytesIO(im_bytes)  # convert image to file-like object
            img = Image.open(im_file)   # img is now PIL Image object
            img = img.resize((224, 224)) 
            # Preprocessing the image
            x = image.img_to_array(img)
            ## Scaling
            x=x/255
            x = np.expand_dims(x, axis=0)
            preds = Model_Apple.predict(x)
            preds=np.argmax(preds, axis=1)

            if preds==0:
                pass_preds="Strawberry___Leaf_scorch"
            elif preds==1:
                pass_preds="Strawberry___healthy"
            


            return{
                "Predicted result":pass_preds
                
            },201
            

        except Exception as e:
            logger.exception(e)
            return {
                "msg": "Internal Error"
            }, 500

