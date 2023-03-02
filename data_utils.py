import time
import torch
import cv2
import numpy as np

def load_image(filepath:str):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess(input_data:np.ndarray, input_size:tuple=None):
    # resize img to desired shape
    if input_size is not None:
        img_data = cv2.resize(input_data, dsize=(input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    img_data = img_data.astype(np.float32)
    img_data /= 255.0
    img_data = img_data.transpose((2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

def postprocess(pred_data:np.ndarray):
    if len(pred_data.shape)==4:
        t = pred_data.squeeze(axis=0)
    
    t = np.clip(t, 0.0, 1.0)
    t *= 255.0
    t = np.round(t).astype(np.uint8)
    t = np.transpose(t, axes=[1, 2, 0])
    t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    return t
    