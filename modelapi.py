#import libraries
import cv2
import numpy as np 
import os
import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#vars
xs = []
ys = []
n_ys = []

unicode = "2306 2310 2311 2312 2313 2315 2319 2322 2325 2327 2328 2330 2331 2332 2334 2335 2336 2337 2338 2340 2341 2342 2343 2344 2346 2347 2348 2349 2350 2351 2352 2354 2357 2358 2359 2360 2361 2362 2363 2364 2366 2367 2368 2369 2370 2375 2376 2379 2380 2381 2387 2399 2404 2405 2406 2407 2408 2409 2410 2411 2412 2413 2416 2429".split(' ')
unicode = [int(a) for a in unicode] #convert to int
from collections import defaultdict
unicode2idx = defaultdict(int)

idx = 0 
for uni in unicode : 
    unicode2idx[uni] = idx
    idx += 1

idx2unicode = defaultdict(int)
for i in range(idx) :
    idx2unicode[i] = unicode[i]
# print("idx2unicode[1] ", idx2unicode[1])


def process(imagename) : 
    image = cv2.imread(imagename, 0)
    bilateral = cv2.bilateralFilter(image, 0, 15, 23)
    median = cv2.medianBlur(bilateral, 5)
    _, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #padding to 150, 150, [96, x]
    #image.shape[0] = height
    #image.shape[1] = width
    rows, cols = image.shape[0:2]

    #difference to pad up to get square image
    diff = abs(rows - cols)
    one = diff // 2
    other = diff // 2    #integer division
    if diff % 2 != 0 :   #131 - 128 square // squaring k liye
        other += 1

    if rows > cols : 
        pad = cv2.copyMakeBorder(otsu, 0, 0, one, other, cv2.BORDER_CONSTANT, 0)
    else : 
        pad = cv2.copyMakeBorder(otsu, one, other, 0, 0, cv2.BORDER_CONSTANT, 0)

    #bring to uniform size
    size = max(rows, cols)

    if size >= 64 : 
        clean = cv2.resize(pad, (64, 64))
    else : 
        pixel_pad = 64 - size
        one = pixel_pad // 2
        other = pixel_pad // 2
        if pixel_pad % 2 == 1 : 
            other += 1
        clean = cv2.copyMakeBorder(pad, one, other, one, other, cv2.BORDER_CONSTANT, 0)
    return clean

def predict(imagename) : 
    global model, n_model
    global idx2unicode, unicode2idx, unicode
    clean = process(imagename)
    
    clean = clean.reshape(1, 64, 64, 1)
    clean = clean.astype('float32')
    clean /= 255
    
    n_probs = n_model.predict(clean)[0]
    n_predictions = np.argmax(n_probs) + 1
    
    predictions = model.predict(clean)[0]
    top_n = np.argsort(predictions)[-n_predictions:] #list of toppers (reversed)
    
#     print(result)
    #print(top_n)
    result = []
    #convert idx to unicode 
    for i in top_n : 
        #print(imagename, i, idx2unicode[i])
        result.append(idx2unicode[i])
    result.reverse()
    

    return result


def load_model(model_name) : 
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded {} from disk".format(model_name))
    return loaded_model

model_name = "model"
n_model_name = "n_model"

model = load_model(model_name)
n_model = load_model(n_model_name)
